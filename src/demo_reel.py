"""여러 영상의 짧은 구간에 판정을 그려 하나의 데모 영상으로 이어 붙인다.

사용: python src/demo_reel.py eval/demo_reel.json <출력 폴더> [en|ko|en,ko]
입력 json: [{"vid": 영상ID, "start": 초, "len": 초, "truth": 제품 키 또는 종류}, ...]
  영상은 data/raw/demo/<영상ID>.mp4 에 있어야 한다(sources/demo_plan.txt, src/demo_scan.py 참고).
결과: <출력 폴더>/demo_reel.<언어>.mp4
구간마다 왼쪽 아래에 사람이 확인한 정답을 띄우고, 상자와 이름은 모델이 그 자리에서 판정한 것이다.
"""
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
W, H, FPS = 1920, 1080, 30

NAME = {
    "en": {"wayfarer": "Ray-Ban Meta Wayfarer", "headliner": "Ray-Ban Meta Headliner", "skyler": "Ray-Ban Meta Skyler",
           "adventurer": "Meta Adventurer", "fury": "Meta Fury", "oakley_hstn": "Oakley Meta HSTN",
           "vanguard": "Oakley Meta Vanguard", "display": "Meta Ray-Ban Display", "rokid": "Rokid Glasses",
           "samsung": "Samsung Intelligent Eyewear", "xiaomi": "Xiaomi AI Glasses",
           "eveng2": "Even Realities G2 (no camera)", "xreal": "XREAL (no camera)",
           "plain_glasses": "ordinary glasses", "no_glasses": "no glasses"},
    "ko": {"wayfarer": "레이밴 메타 웨이페어러", "headliner": "레이밴 메타 헤드라이너", "skyler": "레이밴 메타 스카일러",
           "adventurer": "메타 어드밴처러", "fury": "메타 퓨리", "oakley_hstn": "오클리 메타 HSTN",
           "vanguard": "오클리 메타 뱅가드", "display": "메타 레이밴 디스플레이", "rokid": "로키드 글래스",
           "samsung": "삼성 인텔리전트 아이웨어", "xiaomi": "샤오미 AI 글래스",
           "eveng2": "이븐 리얼리티스 G2 (카메라 없음)", "xreal": "엑스리얼 (카메라 없음)",
           "plain_glasses": "일반 안경", "no_glasses": "안경 없음"},
}
TXT = {
    "en": {"truth": "Actual: ", "title": "Camera smart glasses detection",
           "sub": ["Clips from videos the model never saw in training.",
                   "Bottom left: what the person is actually wearing.",
                   "Box and name: what the model says, frame by frame.",
                   "Red = camera glasses.  Yellow = smart glasses without a camera.  Gray = no camera glasses."],
           "end": "github.com/fattaillabs/smartglasses-detect", "end2": "Model, code and dataset. Non-commercial use."},
    "ko": {"truth": "실제: ", "title": "카메라 스마트안경 탐지",
           "sub": ["모델이 학습 때 본 적 없는 영상의 구간들입니다.",
                   "왼쪽 아래: 실제로 쓰고 있는 것.",
                   "상자와 이름: 모델이 프레임마다 내린 판정.",
                   "빨강 = 카메라 안경.  노랑 = 카메라 없는 스마트안경.  회색 = 카메라 안경 아님."],
           "end": "github.com/fattaillabs/smartglasses-detect", "end2": "모델, 코드, 데이터셋. 비상업 용도."},
}


def card(lines, out, seconds):
    """글자만 있는 화면(시작·끝)을 영상으로 만든다. lines = [(글, 크기, 굵게)]"""
    im = Image.new("RGB", (W, H), (16, 16, 16))
    d = ImageDraw.Draw(im)
    total = sum(sz + 26 for _, sz, _ in lines)
    y = (H - total) // 2
    for text, sz, bold in lines:
        f = ImageFont.truetype(FONT, sz, index=6 if bold else 0)
        d.text(((W - f.getlength(text)) // 2, y), text, font=f, fill=(255, 255, 255) if bold else (205, 205, 205))
        y += sz + 26
    png = out.with_suffix(".png")
    im.save(png)
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-loop", "1", "-t", str(seconds), "-i", str(png),
                    "-f", "lavfi", "-t", str(seconds), "-i", "anullsrc=r=44100:cl=stereo", "-r", str(FPS),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)], check=True)
    png.unlink()


def main():
    clips = json.load(open(sys.argv[1]))
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    work.mkdir(exist_ok=True)
    langs = sys.argv[3].split(",") if len(sys.argv) > 3 else ["en"]   # 기본은 영어판만, "en,ko" 로 둘 다
    for lang in langs:
        t = TXT[lang]
        parts = [work / f"intro.{lang}.mp4"]
        card([(t["title"], 88, True)] + [(x, 44, False) for x in t["sub"]], parts[0], 4.5)
        for k, c in enumerate(clips):
            raw = work / f"clip{k}.{lang}.raw.mp4"
            cap = t["truth"] + NAME[lang][c["truth"]]
            subprocess.run([PY, str(ROOT / "src/demo_video.py"), str(ROOT / "data/raw/demo" / f"{c['vid']}.mp4"), str(raw),
                            "--start", str(c["start"]), "--seconds", str(c["len"]), "--caption", cap],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            fixed = work / f"clip{k}.{lang}.mp4"
            # 크기·초당 프레임·소리 형식을 통일해야 이어 붙일 수 있다(소리가 없는 원본은 무음을 넣는다)
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(raw),
                            "-f", "lavfi", "-t", str(c["len"]), "-i", "anullsrc=r=44100:cl=stereo",
                            "-filter_complex",
                            f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={FPS},setsar=1[v]",
                            "-map", "[v]", "-map", "0:a?", "-map", "1:a", "-c:v", "libx264", "-crf", "19",
                            "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "44100", "-ac", "2", "-shortest", str(fixed)],
                           check=True)
            parts.append(fixed)
            print(lang, k, c["vid"], c["truth"], flush=True)
        parts.append(work / f"end.{lang}.mp4")
        card([(t["end"], 64, True), (t["end2"], 38, False)], parts[-1], 3.5)
        lst = work / f"list.{lang}.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in parts))
        final = out / f"demo_reel.{lang}.mp4"
        # 구간마다 소리 줄 수가 다를 수 있으니(원본 소리+무음) 첫 소리 줄만 써서 다시 부호화한다
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-map", "0:v:0", "-map", "0:a:0", "-c:v", "libx264", "-crf", "20", "-preset", "medium",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(final)], check=True)
        print("->", final, flush=True)


if __name__ == "__main__":
    main()
