"""사진에서 얼굴마다 스마트안경 착용 여부(와 제품)를 판정한다.

사용: python src/detect.py <사진 또는 폴더> [--weights models/release] [--thr 0.3]
정면에 가깝고(고개 20도 이내) 얼굴 너비가 200픽셀 이상인 얼굴만 판정한다. 그 밖은 '판정 불가'로 알린다.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crop import IMG_EXT, crops_for_image, make_detector  # noqa: E402
from train import MEAN, STD, Net, cut_corners  # noqa: E402

SKIP = {"small": "얼굴이 너무 작음", "pose": "정면이 아님"}


def load_release(d):
    d = Path(d)
    cfg = json.load(open(d / "config.json"))
    m = Net(cfg["backbone"], pretrained=False, n_prod=len(cfg["products"]))
    m.load_state_dict({k: v.float() for k, v in load_file(str(d / "model.safetensors")).items()})
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    return m.to(dev).eval(), dev, cfg


def judge(model, dev, band_bgr):
    """정렬된 띠(BGR) 한 장 → (스마트안경 확률, 제품별 확률)."""
    norm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    band = Image.fromarray(cv2.cvtColor(band_bgr, cv2.COLOR_BGR2RGB))
    l, r = cut_corners(band)
    b = norm(band)[None].to(dev)
    l = norm(l)[None].to(dev)
    r = norm(r.transpose(Image.FLIP_LEFT_RIGHT))[None].to(dev)
    with torch.no_grad():
        o1, q1 = model.both(b, l, r)
        o2, q2 = model.both(torch.flip(b, dims=[3]), r, l)
    p = float((torch.softmax(o1, 1)[0, 1] + torch.softmax(o2, 1)[0, 1]) / 2)
    q = ((torch.softmax(q1, 1) + torch.softmax(q2, 1)) / 2)[0].cpu().tolist() if q1 is not None else []
    return p, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--weights", default=str(Path(__file__).resolve().parent.parent / "models/release"))
    ap.add_argument("--thr", type=float, default=None)
    a = ap.parse_args()
    model, dev, cfg = load_release(a.weights)
    thr = a.thr if a.thr is not None else cfg["threshold"]
    det = make_detector()
    p = Path(a.path)
    files = sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT) if p.is_dir() else [p]
    for f in files:
        bgr = cv2.imread(str(f))
        if bgr is None:
            print(f"{f.name}: 읽을 수 없음")
            continue
        faces = crops_for_image(det, bgr)
        if not faces:
            print(f"{f.name}: 얼굴 없음")
        for band, _, _, info in faces:
            if band is None:
                print(f"{f.name} 얼굴{info['face']}: 판정 불가({SKIP[info['skip']]})")
                continue
            prob, q = judge(model, dev, band)
            names = cfg["products"]
            nocam = set(cfg.get("no_camera_products", []))
            line = f"{f.name} 얼굴{info['face']}: 카메라 스마트안경 확률 {prob * 100:.1f}% → "
            if prob >= thr:
                # 카메라 달린 제품 가운데 가장 그럴듯한 것
                cand = [(q[i], n) for i, n in enumerate(names) if n != "none" and n not in nocam] if q else []
                line += "카메라 스마트안경"
                if cand:
                    c, n = max(cand)
                    line += f" (제품 추정 {n} {c * 100:.0f}%)"
            else:
                c, n = max((q[i], n) for i, n in enumerate(names)) if q else (0, "none")
                if n in nocam and c >= 0.6:
                    line += f"카메라 없는 스마트안경 (제품 추정 {n} {c * 100:.0f}%)"
                else:
                    line += "아님"
            print(line)


if __name__ == "__main__":
    main()
