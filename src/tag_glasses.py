"""잘라낸 띠마다 '안경을 쓰고 있을 확률'을 매겨 <crops 폴더>/glasses.json 에 저장한다.

사용: python src/tag_glasses.py <crops 폴더> [<crops 폴더> ...]
공개 분류기 glasses-detector(anyglasses)를 쓴다. 이미 매긴 것은 건너뛴다.
"""
import json
import sys
from pathlib import Path

from glasses_detector import GlassesClassifier


def main():
    clf = GlassesClassifier(kind="anyglasses", size="medium")
    for d in map(Path, sys.argv[1:]):
        out = d / "glasses.json"
        done = json.load(open(out)) if out.exists() else {}
        files = [f for f in sorted((d / "band").glob("*.jpg")) if f.stem not in done]
        for k in range(0, len(files), 64):
            batch = files[k:k + 64]
            ps = clf.predict([str(f) for f in batch], format="proba")
            for f, p in zip(batch, ps):
                done[f.stem] = round(float(p), 4)
        json.dump(done, open(out, "w"))
        v = list(done.values())
        print(d, "n", len(v), "glasses", sum(p > 0.6 for p in v), "none", sum(p < 0.1 for p in v),
              "unsure", sum(0.1 <= p <= 0.6 for p in v), flush=True)


if __name__ == "__main__":
    main()
