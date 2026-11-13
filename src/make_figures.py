"""README 에 싣는 그림을 만든다: 제품별 판정 예시, 틀린 사례, 입력 조각 설명.

사용: python src/make_figures.py eval/test2_run4.json      → docs/*.en.png, docs/*.ko.png
예시는 테스트 묶음(학습에 쓰지 않은 것)에서만 고른다. 쉬운 것만 고르지 않도록 종류마다 확신이 중간쯤인 것을 쓴다.
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import cut_corners  # noqa: E402

FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
W, H, PAD, CAP = 512, 192, 14, 62
OK, WARN, BAD, INK = (22, 110, 60), (176, 110, 0), (190, 30, 30), (40, 40, 40)
NOCAM = ("eveng2", "xreal")

NAME = {
    "ko": {"none": "스마트안경 아님", "wayfarer": "레이밴 메타 웨이페어러", "headliner": "레이밴 메타 헤드라이너",
           "skyler": "레이밴 메타 스카일러", "adventurer": "메타 어드밴처러", "fury": "메타 퓨리", "nova": "메타 노바",
           "oakley_hstn": "오클리 메타 HSTN", "vanguard": "오클리 메타 뱅가드", "display": "메타 레이밴 디스플레이",
           "rokid": "로키드", "samsung": "삼성 인텔리전트 아이웨어", "xiaomi": "샤오미", "eveng2": "이븐 G2",
           "xreal": "엑스리얼", "rayban_meta": "레이밴 메타", "oakley_meta": "오클리 메타 HSTN",
           "google_glass": "구글 글래스", "plain_glasses": "일반 안경", "no_glasses": "안경 없음",
           "other_smart": "스마트안경(제품 미상)", "scriber": "레이밴 메타 스크라이버"},
    "en": {"none": "not smart glasses", "wayfarer": "Ray-Ban Meta Wayfarer", "headliner": "Ray-Ban Meta Headliner",
           "skyler": "Ray-Ban Meta Skyler", "adventurer": "Meta Adventurer", "fury": "Meta Fury", "nova": "Meta Nova",
           "oakley_hstn": "Oakley Meta HSTN", "vanguard": "Oakley Meta Vanguard", "display": "Meta Ray-Ban Display",
           "rokid": "Rokid Glasses", "samsung": "Samsung Intelligent Eyewear", "xiaomi": "Xiaomi AI Glasses",
           "eveng2": "Even Realities G2", "xreal": "XREAL", "rayban_meta": "Ray-Ban Meta",
           "oakley_meta": "Oakley Meta HSTN", "google_glass": "Google Glass", "plain_glasses": "ordinary glasses",
           "no_glasses": "no glasses", "other_smart": "smart glasses (unknown model)", "scriber": "Ray-Ban Meta Scriber"},
}
TXT = {
    "ko": {"truth": "정답", "pred": "판정", "cam": "카메라 스마트안경", "nocam": "카메라 없는 스마트안경",
           "not": "카메라 스마트안경 아님", "p": "카메라 확률", "band": "눈 주변 띠 (512x192)", "l": "왼쪽 구석", "r": "오른쪽 구석"},
    "en": {"truth": "Truth", "pred": "Predicted", "cam": "camera glasses", "nocam": "smart glasses without a camera",
           "not": "no camera glasses", "p": "camera prob.", "band": "eye band (512x192)", "l": "left corner", "r": "right corner"},
}


def verdict(r, thr=0.5):
    """(종류, 카메라 판정이 맞았나, 제품까지 맞았나)"""
    cam_ok = (r["p"] >= thr) == bool(r["y"])
    prod_ok = r.get("prod") is None or r.get("pred_prod") == r.get("prod") or (r["y"] == 0 and r.get("prod") == "none")
    if r["p"] >= thr:
        return "cam", cam_ok, prod_ok
    if r.get("pred_prod") in NOCAM:
        return "nocam", cam_ok, prod_ok
    return "not", cam_ok, prod_ok


def cell(r, lang, f1, fb):
    n, t = NAME[lang], TXT[lang]
    im = Image.new("RGB", (W, H + CAP), "white")
    im.paste(Image.open(Path(r["dir"]) / "band" / f"{r['id']}.jpg"), (0, 0))
    d = ImageDraw.Draw(im)
    kind, cam_ok, prod_ok = verdict(r)
    d.text((2, H + 6), f"{t['truth']}  {n.get(r['kind'], r['kind'])}", font=f1, fill=INK)
    line = f"{t['pred']}  {t[kind]} {r['p'] * 100:.0f}%" if kind == "cam" else f"{t['pred']}  {t[kind]}"
    if kind in ("cam", "nocam") and r.get("pred_prod"):
        line += f" · {n.get(r['pred_prod'], r['pred_prod'])}"
    if kind != "cam":
        line += f" ({t['p']} {r['p'] * 100:.0f}%)"
    d.text((2, H + 32), line, font=fb, fill=BAD if not cam_ok else (OK if prod_ok else WARN))
    return im


def sheet(rows, cols, out, lang, f1, fb):
    R = (len(rows) + cols - 1) // cols
    s = Image.new("RGB", (cols * W + (cols + 1) * PAD, R * (H + CAP) + (R + 1) * PAD), "white")
    for k, r in enumerate(rows):
        s.paste(cell(r, lang, f1, fb), (PAD + (k % cols) * (W + PAD), PAD + (k // cols) * (H + CAP + PAD)))
    s.save(out, optimize=True)
    print(out, s.size)


def main():
    res = json.load(open(sys.argv[1]))
    Path("docs").mkdir(exist_ok=True)
    order = ["wayfarer", "headliner", "skyler", "display", "oakley_hstn", "vanguard", "rokid", "xiaomi", "samsung",
             "fury", "eveng2", "xreal", "plain_glasses", "no_glasses"]
    pick = []
    for k in order:
        if k in ("plain_glasses", "no_glasses"):
            c = [r for r in res if r["kind"] == k and verdict(r)[1]]
            # 일반 안경은 웹 사진(화질이 고른 것)에서 고른다
            c = [r for r in c if r["source"] == "web_test"] or c
        else:
            c = [r for r in res if r.get("prod") == k and verdict(r)[1]]
        c = sorted(c, key=lambda r: r["p"])
        pick.append(c[len(c) // 2])
    err = [r for r in res if not verdict(r)[1]]
    for lang in ("en", "ko"):
        f1 = ImageFont.truetype(FONT, 19, index=0)
        f2 = ImageFont.truetype(FONT, 16, index=0)
        fb = ImageFont.truetype(FONT, 18, index=6)
        sheet(pick, 2, f"docs/results.{lang}.png", lang, f1, fb)
        sheet(err, 2, f"docs/errors.{lang}.png", lang, f1, fb)
        r = pick[6]
        b = Image.open(Path(r["dir"]) / "band" / f"{r['id']}.jpg")
        l, rr = cut_corners(b)
        s = Image.new("RGB", (512 + 224 * 2 + PAD * 4, 224 + PAD * 2 + 34), "white")
        d = ImageDraw.Draw(s)
        s.paste(b, (PAD, PAD + 16)); s.paste(l, (512 + PAD * 2, PAD)); s.paste(rr, (512 + 224 + PAD * 3, PAD))
        t = TXT[lang]
        d.text((PAD, 224 + PAD + 6), t["band"], font=f2, fill=INK)
        d.text((512 + PAD * 2, 224 + PAD + 6), t["l"], font=f2, fill=INK)
        d.text((512 + 224 + PAD * 3, 224 + PAD + 6), t["r"], font=f2, fill=INK)
        s.save(f"docs/input.{lang}.png", optimize=True)


if __name__ == "__main__":
    main()
