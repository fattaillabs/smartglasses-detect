"""영상의 프레임마다 스마트안경을 판정해 상자와 제품 이름을 그린 영상을 만든다.

사용: python src/demo_video.py <입력 영상> <출력 영상> [--weights models/release] [--start 0] [--seconds 0]
- 카메라 스마트안경: 빨간 상자 + 제품 이름 + 확률
- 카메라 없는 스마트안경: 노란 상자 + 제품 이름
- 판정했지만 해당 없는 얼굴: 회색 상자 + "no camera glasses"
- 정면이 아니거나 너무 작은 얼굴: 표시하지 않음
확률은 같은 얼굴에 대해 프레임 사이에 부드럽게 이어 붙인다(깜빡임 방지). 소리는 원본에서 그대로 옮긴다.
"""
import argparse
import math
import subprocess
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crop import (BAND_H, BAND_W, EYE_OUT_A, EYE_OUT_B, FACE_EDGE_A, FACE_EDGE_B, FACE_W_OUT, make_detector,  # noqa: E402
                  pose_deg, warp)
from detect import load_release  # noqa: E402
from train import MEAN, STD, cut_corners  # noqa: E402

LABEL = {"wayfarer": "Ray-Ban Meta Wayfarer", "headliner": "Ray-Ban Meta Headliner", "skyler": "Ray-Ban Meta Skyler",
         "adventurer": "Meta Adventurer", "fury": "Meta Fury", "nova": "Meta Nova", "oakley_hstn": "Oakley Meta HSTN",
         "vanguard": "Oakley Meta Vanguard", "display": "Meta Ray-Ban Display", "rokid": "Rokid Glasses",
         "samsung": "Samsung Intelligent Eyewear", "xiaomi": "Xiaomi AI Glasses", "eveng2": "Even Realities G2",
         "xreal": "XREAL"}
RED, YELLOW, GRAY, WHITE = (48, 48, 235), (40, 200, 245), (120, 120, 120), (255, 255, 255)   # BGR


def faces_in(det, bgr, max_yaw=20.0, max_pitch=20.0, min_face=200.0):
    """판정할 수 있는 얼굴마다 (정렬된 띠, 원본 좌표의 상자, 중심)."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    out = []
    for i, lms in enumerate(res.face_landmarks):
        p = np.array([[l.x * w, l.y * h] for l in lms], dtype=np.float32)
        ea, eb, fa, fb = p[EYE_OUT_A], p[EYE_OUT_B], p[FACE_EDGE_A], p[FACE_EDGE_B]
        if ea[0] > eb[0]:
            ea, eb, fa, fb = eb, ea, fb, fa
        face_w = float(np.linalg.norm(fb - fa))
        yaw, pitch = pose_deg(res.facial_transformation_matrixes[i])
        if face_w < min_face or abs(yaw) > max_yaw or abs(pitch) > max_pitch:
            continue
        angle = math.degrees(math.atan2(eb[1] - ea[1], eb[0] - ea[0]))
        scale = FACE_W_OUT / face_w
        mid = (ea + eb) / 2
        band = warp(bgr, mid, angle, scale, BAND_W, BAND_H)
        # 띠가 원본에서 차지하는 범위(안경 영역)를 상자로
        bw, bh = BAND_W / scale * 0.86, BAND_H / scale * 0.78
        x0, y0 = int(mid[0] - bw / 2), int(mid[1] - bh / 2)
        out.append((band, (x0, y0, int(x0 + bw), int(y0 + bh)), (float(mid[0]), float(mid[1])), face_w))
    return out


def score(model, dev, norm, band_bgr):
    band = Image.fromarray(cv2.cvtColor(band_bgr, cv2.COLOR_BGR2RGB))
    l, r = cut_corners(band)
    b = norm(band)[None].to(dev)
    l = norm(l)[None].to(dev)
    r = norm(r.transpose(Image.FLIP_LEFT_RIGHT))[None].to(dev)
    with torch.no_grad():
        o1, q1 = model.both(b, l, r)
        o2, q2 = model.both(torch.flip(b, dims=[3]), r, l)
    p = float((torch.softmax(o1, 1)[0, 1] + torch.softmax(o2, 1)[0, 1]) / 2)
    q = ((torch.softmax(q1, 1) + torch.softmax(q2, 1)) / 2)[0].cpu().numpy()
    return p, q


def draw_box(img, box, color, text, thick=3):
    x0, y0, x1, y1 = box
    k = img.shape[0] / 1080           # 화면 높이에 맞춰 글자와 선 굵기를 키운다
    thick = max(1, round(thick * 1.6 * k))
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thick, cv2.LINE_AA)
    if not text:
        return
    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 1.25 * k, max(2, round(3 * k))
    (tw, th), base = cv2.getTextSize(text, font, fs, ft)
    pad = round(10 * k)
    ty = y0 - pad if y0 - th - 2 * pad > 0 else y1 + th + 2 * pad
    cv2.rectangle(img, (x0 - thick // 2, ty - th - pad), (x0 + tw + 2 * pad, ty + base), color, -1, cv2.LINE_AA)
    cv2.putText(img, text, (x0 + pad, ty - 2), font, fs, WHITE if color != YELLOW else (20, 20, 20), ft, cv2.LINE_AA)


def make_banner(text, height):
    """왼쪽 아래에 얹을 자막 띠(BGR, 불투명도)를 만든다."""
    from PIL import ImageDraw, ImageFont
    k = height / 1080
    font = ImageFont.truetype("/System/Library/Fonts/AppleSDGothicNeo.ttc", round(40 * k), index=6)
    pad = round(18 * k)
    w = int(font.getlength(text)) + pad * 2
    h = round(40 * k) + pad * 2
    im = Image.new("RGB", (w, h), (18, 18, 18))
    ImageDraw.Draw(im).text((pad, pad - round(4 * k)), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def put_banner(frame, banner):
    h, w = frame.shape[:2]
    bh, bw = banner.shape[:2]
    m = round(40 * h / 1080)
    bw = min(bw, w - 2 * m)
    roi = frame[h - m - bh:h - m, m:m + bw]
    roi[:] = (roi * 0.18 + banner[:, :bw] * 0.82).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--weights", default=str(Path(__file__).resolve().parent.parent / "models/release"))
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--caption", default="", help="왼쪽 아래에 띄울 글(예: 정답 제품 이름)")
    a = ap.parse_args()
    model, dev, cfg = load_release(a.weights)
    names, nocam, thr = cfg["products"], set(cfg.get("no_camera_products", [])), cfg["threshold"]
    cam_idx = [i for i, n in enumerate(names) if n != "none" and n not in nocam]
    norm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    det = make_detector()
    cap = cv2.VideoCapture(a.src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = int(a.start * fps)
    last = min(total, first + int(a.seconds * fps)) if a.seconds else total
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    tmp = str(Path(a.dst).with_suffix(".silent.mp4"))
    ff = subprocess.Popen(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "medium",
                           "-crf", "20", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    banner = make_banner(a.caption, h) if a.caption else None
    tracks = []      # 얼굴마다 {c: 중심, p: 부드럽게 이은 확률, q: 제품 확률, age: 못 본 프레임 수}
    stats = {"frames": 0, "judged": 0, "camera": 0, "nocam": 0}
    log = []
    for n in range(first, last):
        ok, frame = cap.read()
        if not ok:
            break
        found = faces_in(det, frame)
        for t in tracks:
            t["age"] += 1
        for band, box, c, fw in found:
            p, q = score(model, dev, norm, band)
            t = min(tracks, key=lambda t: (t["c"][0] - c[0]) ** 2 + (t["c"][1] - c[1]) ** 2, default=None)
            if t is None or math.hypot(t["c"][0] - c[0], t["c"][1] - c[1]) > fw * 0.6 or t["age"] > 8 or t.get("used") == n:
                t = {"p": p, "q": q}
                tracks.append(t)
            else:
                # 확률은 빠르게, 제품 이름은 천천히 따라가게 한다(이름이 프레임마다 튀지 않도록)
                t["p"] = t["p"] * 0.55 + p * 0.45
                t["q"] = t["q"] * 0.88 + q * 0.12
            t.update(c=c, age=0, used=n)
            stats["judged"] += 1
            best = int(np.argmax(t["q"]))
            if t["p"] >= thr:
                ci = max(cam_idx, key=lambda i: t["q"][i])
                draw_box(frame, box, RED, f"{LABEL[names[ci]]} {t['p']:.2f}")
                stats["camera"] += 1
                log.append((n, "camera", names[ci], round(t["p"], 3)))
            elif names[best] in nocam and t["q"][best] >= 0.6:
                draw_box(frame, box, YELLOW, f"{LABEL[names[best]]} (no camera)")
                stats["nocam"] += 1
                log.append((n, "nocam", names[best], round(t["p"], 3)))
            else:
                draw_box(frame, box, GRAY, f"no camera glasses {t['p']:.2f}", thick=1.6)
                log.append((n, "none", "", round(t["p"], 3)))
        tracks = [t for t in tracks if t["age"] <= 15]
        if banner is not None:
            put_banner(frame, banner)
        ff.stdin.write(frame.tobytes())
        stats["frames"] += 1
        if stats["frames"] % 900 == 0:
            print(stats, flush=True)
    ff.stdin.close()
    ff.wait()
    cap.release()
    # 원본 소리를 옮겨 붙인다
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", tmp, "-ss", f"{a.start}", "-i", a.src,
           "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-shortest", a.dst]
    subprocess.run(cmd, check=True)
    Path(tmp).unlink(missing_ok=True)
    Path(a.dst).with_suffix(".log.tsv").write_text("\n".join("\t".join(map(str, x)) for x in log) + "\n")
    print("done", stats, "->", a.dst)


if __name__ == "__main__":
    main()
