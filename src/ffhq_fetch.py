"""FFHQ 원본(1024px)에서 안경 쓴 얼굴/안 쓴 얼굴을 무작위로 골라 받는다.

사용: python src/ffhq_fetch.py --glasses 3000 --noglasses 1500
결과: data/raw/ffhq/glasses/*.png, data/raw/ffhq/noglasses/*.png
안경 여부는 FFHQ 안경 확장판 주석(Zenodo 14252074)에 번호가 있는지로 정한다.
"""
import argparse
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "data/raw/ffhq_eyeglasses/labels/FFHQ_Eyeglasses_Detection_labels"
BASE = "https://huggingface.co/datasets/marcosv/ffhq-dataset/resolve/main"


def url(i: int) -> str:
    return f"{BASE}/Part{i // 10000 + 1}/{i:05d}.png"


def fetch(job):
    i, dst = job
    f = dst / f"{i:05d}.png"
    if f.exists() and f.stat().st_size > 100_000:
        return True
    for attempt in range(4):
        try:
            req = urllib.request.Request(url(i), headers={"User-Agent": "python-urllib"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) < 100_000:
                raise IOError("too small")
            f.write_bytes(data)
            return True
        except Exception as e:  # noqa: BLE001
            time.sleep(2 + attempt * 5)
    print("FAIL", i, flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glasses", type=int, default=3000)
    ap.add_argument("--noglasses", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    with_g = sorted(int(p.stem) for p in LABELS.glob("*.json"))
    gset = set(with_g)
    without = [i for i in range(70000) if i not in gset]
    rnd = random.Random(a.seed)
    jobs = []
    for name, pool, n in (("glasses", with_g, a.glasses), ("noglasses", without, a.noglasses)):
        dst = ROOT / "data/raw/ffhq" / name
        dst.mkdir(parents=True, exist_ok=True)
        jobs += [(i, dst) for i in rnd.sample(pool, n)]
    rnd.shuffle(jobs)
    ok = 0
    with ThreadPoolExecutor(6) as ex:
        for n, r in enumerate(ex.map(fetch, jobs), 1):
            ok += bool(r)
            if n % 250 == 0:
                print(n, "/", len(jobs), "ok", ok, flush=True)
    print("done", ok, "/", len(jobs))


if __name__ == "__main__":
    main()
