"""안경 착용 판정기(models/wear)용 학습 목록을 만든다. 정답: 1 = 안경 착용, 0 = 맨얼굴.

FFHQ 의 안경 착용/미착용 조각과, 검수를 마친 웹 스마트안경 조각(전부 착용)을 쓴다.
사용: python src/build_wear_list.py && python src/train.py --list data/wear_list.jsonl --out models/wear --epochs 4
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def bucket(k: str) -> int:
    return int(hashlib.sha1(k.encode()).hexdigest(), 16) % 100


rows = []
for lab, y in (("glasses", 1), ("noglasses", 0)):
    d = ROOT / "data/crops/ffhq" / lab
    for f in sorted((d / "band").glob("*.jpg")):
        rows.append({"id": f.stem, "dir": str(d), "y": y, "group": f.stem, "source": "ffhq_" + lab,
                     "split": "val" if bucket(f.stem) < 15 else "train"})
d = ROOT / "data/crops/web/rbmeta"
if (d / "relabel.json").exists():
    for k in json.load(open(d / "relabel.json")):
        rows.append({"id": k, "dir": str(d), "y": 1, "group": k, "source": "web_rbmeta",
                     "split": "val" if bucket(k) < 15 else "train"})
out = ROOT / "data/wear_list.jsonl"
out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print(len(rows), "->", out)
