"""얼굴 사진에서 안경 띠(눈 주변)와 테 양쪽 모서리 조각을 정렬해 잘라낸다.

사용: python src/crop.py <입력 폴더> <출력 폴더> [--label 이름] [--max-yaw 20] [--min-face 200]

- 두 눈이 수평이 되게 돌리고 얼굴 너비로 크기를 맞춘다.
- 고개 각도(좌우·상하)가 크거나 얼굴이 너무 작으면 버린다.
- 사진 한 장에 얼굴이 여럿이면 조건을 통과한 얼굴마다 따로 저장한다.
- 결과: <출력>/band/*.jpg (512x192), <출력>/corner/*_L.jpg, *_R.jpg (160x160), <출력>/meta.jsonl
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "face_landmarker.task"

# 478점 랜드마크 번호
EYE_OUT_A, EYE_OUT_B = 33, 263      # 양쪽 눈꼬리
FACE_EDGE_A, FACE_EDGE_B = 127, 356  # 눈높이의 얼굴 양쪽 가장자리(관자놀이)

BAND_W, BAND_H = 512, 192
FACE_W_OUT = 400.0          # 정렬 뒤 얼굴 너비(픽셀)
CORNER = 160                # 모서리 조각 한 변
CORNER_SRC = 0.26           # 모서리 조각이 덮는 범위(얼굴 너비 대비)
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def make_detector(max_faces: int = 5):
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        output_facial_transformation_matrixes=True,
        num_faces=max_faces,
        min_face_detection_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def pose_deg(m: np.ndarray):
    r = np.asarray(m)[:3, :3]
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -r[2, 0]))))
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    return yaw, pitch


def warp(img, center, angle_deg, scale, out_w, out_h):
    """center 가 출력 한가운데 오도록 돌리고 키워서 자른다."""
    m = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle_deg, scale)
    m[0, 2] += out_w / 2 - center[0]
    m[1, 2] += out_h / 2 - center[1]
    return cv2.warpAffine(img, m, (out_w, out_h), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_REPLICATE)


def crops_for_image(det, bgr, max_yaw=20.0, max_pitch=20.0, min_face=200.0):
    """통과한 얼굴마다 (band, cornerL, cornerR, info) 를 돌려준다. 탈락 사유는 info['skip']."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    out = []
    for i, lms in enumerate(res.face_landmarks):
        p = np.array([[l.x * w, l.y * h] for l in lms], dtype=np.float32)
        ea, eb = p[EYE_OUT_A], p[EYE_OUT_B]
        fa, fb = p[FACE_EDGE_A], p[FACE_EDGE_B]
        if ea[0] > eb[0]:
            ea, eb, fa, fb = eb, ea, fb, fa
        face_w = float(np.linalg.norm(fb - fa))
        yaw, pitch = pose_deg(res.facial_transformation_matrixes[i])
        info = {"face": i, "face_w": round(face_w, 1), "yaw": round(yaw, 1), "pitch": round(pitch, 1)}
        if face_w < min_face:
            info["skip"] = "small"
        elif abs(yaw) > max_yaw or abs(pitch) > max_pitch:
            info["skip"] = "pose"
        if "skip" in info:
            out.append((None, None, None, info))
            continue
        angle = math.degrees(math.atan2(eb[1] - ea[1], eb[0] - ea[0]))
        scale = FACE_W_OUT / face_w
        mid = (ea + eb) / 2
        band = warp(bgr, mid, angle, scale, BAND_W, BAND_H)
        cs = CORNER / (CORNER_SRC * face_w)   # 모서리는 더 크게 확대
        # 테 끝은 눈꼬리와 얼굴 가장자리 사이~바깥에 있다: 가장자리 쪽으로 치우친 점을 중심으로
        ca = ea + (fa - ea) * 0.75
        cb = eb + (fb - eb) * 0.75
        cl = warp(bgr, ca, angle, cs, CORNER, CORNER)
        cr = warp(bgr, cb, angle, cs, CORNER, CORNER)
        info["upscale"] = round(scale, 2)
        out.append((band, cl, cr, info))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--label", default="")
    ap.add_argument("--max-yaw", type=float, default=20)
    ap.add_argument("--max-pitch", type=float, default=20)
    ap.add_argument("--min-face", type=float, default=200)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    (dst / "band").mkdir(parents=True, exist_ok=True)
    (dst / "corner").mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in src.rglob("*") if f.suffix.lower() in IMG_EXT)
    if a.limit:
        files = files[: a.limit]
    det = make_detector()
    stats = {"files": len(files), "saved": 0, "noface": 0, "small": 0, "pose": 0, "unreadable": 0}
    with open(dst / "meta.jsonl", "a") as meta:
        for n, f in enumerate(files):
            bgr = cv2.imread(str(f))
            if bgr is None:
                stats["unreadable"] += 1
                continue
            results = crops_for_image(det, bgr, a.max_yaw, a.max_pitch, a.min_face)
            if not results:
                stats["noface"] += 1
            for band, cl, cr, info in results:
                if band is None:
                    stats[info["skip"]] += 1
                    continue
                stem = f"{f.stem}_f{info['face']}"
                cv2.imwrite(str(dst / "band" / f"{stem}.jpg"), band, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(dst / "corner" / f"{stem}_L.jpg"), cl, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(dst / "corner" / f"{stem}_R.jpg"), cr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                info.update({"id": stem, "src": str(f), "label": a.label})
                meta.write(json.dumps(info, ensure_ascii=False) + "\n")
                stats["saved"] += 1
            if (n + 1) % 200 == 0:
                print(n + 1, stats, file=sys.stderr, flush=True)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
