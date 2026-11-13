"""묶음 단위 판정(labels/clusters/<이름>_*.txt)을 프레임 단위 정답으로 풀어 각 영상 폴더에 저장한다.

사용: python src/apply_clusters.py <이름>      (eval/<이름>.json 과 labels/clusters/<이름>_*.txt 를 읽는다)
결과: <영상 crops 폴더>/frame_labels.json = {프레임 id: "S"|"O"|"P"|"N"|"X"}
  S=이 영상의 제품(스마트안경) O=다른 스마트안경 P=일반 안경 N=안경 없음 X=잡음·불분명(학습·평가에서 뺌)
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
name = sys.argv[1]
clusters = json.load(open(ROOT / f"eval/{name}.json"))
code = {}
for f in sorted((ROOT / "labels/clusters").glob(f"{name}_*.txt")):
    for line in f.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        c, *nums = line.split()
        for n in nums:
            code[int(n)] = c
missing = [c["n"] for c in clusters if c["n"] not in code]
if missing:
    raise SystemExit(f"판정이 없는 묶음: {missing[:20]} ... ({len(missing)}개)")
per = defaultdict(dict)
for c in clusters:
    for i in c["ids"]:
        per[c["dir"]][i] = code[c["n"]]
tot = Counter()
for d, m in per.items():
    json.dump(m, open(Path(d) / "frame_labels.json", "w"))
    tot.update(m.values())
# 저장소에 올릴 수 있게 판정 결과를 영상별로 한 파일에 모아 둔다
json.dump({d: m for d, m in per.items()}, open(ROOT / f"labels/clusters/{name}_frames.json", "w"))
print(len(per), "videos", dict(tot))
