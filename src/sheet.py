"""잘라낸 안경 띠(와 모서리 조각)를 한 장에 모아 눈으로 검수하는 시트를 만든다.

사용: python src/sheet.py <crops 폴더(band/ corner/ 가 있는 곳)> <출력.jpg> [--n 36] [--cols 3] [--seed 0] [--even]
--even: 무작위 대신 파일 이름 순서로 고르게 뽑는다(영상이면 시간 순).
"""
import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

BW, BH, C = 384, 144, 72


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--even", action="store_true")
    ap.add_argument("--page", type=int, default=-1, help="이름 순서로 n장씩 끊어 page번째 묶음(검수용, 번호 표시)")
    a = ap.parse_args()
    src = Path(a.src)
    bands = sorted((src / "band").glob("*.jpg"))
    if not bands:
        raise SystemExit("no crops")
    base = 0
    if a.page >= 0:
        base = a.page * a.n
        bands = bands[base:base + a.n]
    elif len(bands) > a.n:
        if a.even:
            step = len(bands) / a.n
            bands = [bands[int(i * step)] for i in range(a.n)]
        else:
            bands = sorted(random.Random(a.seed).sample(bands, a.n))
    cell_w, cell_h = BW + C, BH + 14
    rows = (len(bands) + a.cols - 1) // a.cols
    sheet = Image.new("RGB", (cell_w * a.cols, cell_h * rows), "white")
    d = ImageDraw.Draw(sheet)
    for k, b in enumerate(bands):
        x, y = (k % a.cols) * cell_w, (k // a.cols) * cell_h
        sheet.paste(Image.open(b).resize((BW, BH)), (x, y))
        for j, side in enumerate("LR"):
            c = src / "corner" / f"{b.stem}_{side}.jpg"
            if c.exists():
                sheet.paste(Image.open(c).resize((C, C)), (x + BW, y + j * C))
        d.text((x + 2, y + BH + 1), f"#{base + k}  {b.stem[-24:]}", fill="black")
    sheet.save(a.out, quality=88)
    print(a.out, len(bands))


if __name__ == "__main__":
    main()
