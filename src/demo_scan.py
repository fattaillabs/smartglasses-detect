"""데모 모음에 쓸 구간을 영상마다 자동으로 고른다.

사용: python src/demo_scan.py sources/demo_plan.txt [--len 6]
영상을 초당 두 장씩 훑어, 판정할 수 있는 정면 얼굴에 안경을 쓴 프레임이 가장 많은 구간을 고른다.
안경 착용 여부는 공개 분류기(glasses-detector)로 본다. 스마트안경 판별 모델의 출력은 고르는 데 쓰지 않는다
(모델이 맞힌 구간만 골라 보여 주는 일을 막기 위해서다).
결과: eval/demo_scan.json, eval/sheets/demo_scan.jpg(구간 가운데 프레임 모음, 사람이 정답을 확인하는 용도)
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
from glasses_detector import GlassesClassifier
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crop import make_detector  # noqa: E402
from demo_video import faces_in  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STEP = 0.75    # 초


def scan(det, clf, path: Path):
    """[(초, 판정 가능한 얼굴이 있나, 안경 확률)]"""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    out, t = [], 0.0
    with tempfile.TemporaryDirectory() as td:
        while t < dur - 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, frame = cap.read()
            if not ok:
                break
            found = faces_in(det, frame)
            g = -1.0
            if found:
                band = max(found, key=lambda x: x[3])[0]      # 가장 큰 얼굴
                f = Path(td) / "b.jpg"
                cv2.imwrite(str(f), band)
                g = float(clf.predict(str(f), format="proba"))
            out.append((round(t, 2), bool(found), round(g, 3)))
            t += STEP
    cap.release()
    return out


def best_window(samples, length, want_glasses=True):
    n = int(length / STEP)
    best, at = -1, 0
    for i in range(0, max(1, len(samples) - n)):
        w = samples[i:i + n]
        s = sum(1 for _, ok, g in w if ok and ((g >= 0.5) if want_glasses else (0 <= g < 0.2)))
        if s > best:
            best, at = s, i
    return samples[at][0], best, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--len", type=float, default=6)
    ap.add_argument("--out", default="eval/demo_scan", help="결과 이름(.json 과 시트 .jpg)")
    a = ap.parse_args()
    det = make_detector()
    clf = GlassesClassifier(kind="anyglasses", size="medium")
    rows = []
    for line in open(a.plan):
        if not line.strip() or line.startswith("#"):
            continue
        kind, vid = line.split()
        path = ROOT / "data/raw/demo" / f"{vid}.mp4"
        if not path.exists():
            print("missing", vid, flush=True)
            continue
        s = scan(det, clf, path)
        start, score, n = best_window(s, a.len, True)
        rows.append({"vid": vid, "kind": kind, "start": start, "len": a.len, "score": score, "of": n})
        # 같은 영상에서 안경을 벗은 구간도 하나 찾아 둔다(맨얼굴 예시 후보)
        b0, bs, _ = best_window(s, a.len, False)
        if bs >= n * 0.75 and abs(b0 - start) > a.len:
            rows.append({"vid": vid, "kind": "no_glasses", "start": b0, "len": a.len, "score": bs, "of": n})
        print(vid, kind, "start", start, "score", f"{score}/{n}", "| bare", b0, f"{bs}/{n}", flush=True)
    json.dump(rows, open(ROOT / f"{a.out}.json", "w"), indent=1)
    # 확인용 시트: 구간 가운데 프레임
    cols, cw, ch = 4, 480, 270
    sheet = Image.new("RGB", (cols * cw, ((len(rows) + cols - 1) // cols) * (ch + 18)), "white")
    d = ImageDraw.Draw(sheet)
    for k, r in enumerate(rows):
        cap = cv2.VideoCapture(str(ROOT / "data/raw/demo" / f"{r['vid']}.mp4"))
        cap.set(cv2.CAP_PROP_POS_MSEC, (r["start"] + r["len"] / 2) * 1000)
        ok, frame = cap.read()
        cap.release()
        if ok:
            im = Image.fromarray(cv2.cvtColor(cv2.resize(frame, (cw, ch)), cv2.COLOR_BGR2RGB))
            sheet.paste(im, ((k % cols) * cw, (k // cols) * (ch + 18)))
        d.text(((k % cols) * cw + 3, (k // cols) * (ch + 18) + ch + 2),
               f"#{k} {r['kind']} {r['vid']} @{r['start']}s {r['score']}/{r['of']}", fill="black")
    sheet.save(ROOT / f"{a.out}.jpg", quality=88)
    print("rows", len(rows))


if __name__ == "__main__":
    main()
