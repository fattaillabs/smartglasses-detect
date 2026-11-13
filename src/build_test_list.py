"""테스트 묶음(data/test_list.jsonl)을 만든다. 학습에 전혀 쓰지 않는 영상·사진만 들어간다.

- 1차 테스트 묶음(labels/test_list_v1.json): 초기 제품(레이밴 메타·오클리 HSTN)과 일반 안경·맨얼굴. 사람이 확인한 정답.
- 확장분: data/crops/yt_test 아래에서 묶음 검수를 거친 영상. 영상·판정 종류마다 고르게 최대 20장.
제품(prod): "none" = 스마트안경 아님, None = 스마트안경이지만 제품 미상(제품 평가에서 뺌).
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_list import BY_VIDEO, LABEL2PROD, NOCAM, PRODUCTS, ROOT  # noqa: E402

PER = 20


def even(w, per):
    return w if len(w) <= per else [w[int(i * len(w) / per)] for i in range(per)]


rows = []
for r in json.load(open(ROOT / "labels/test_list_v1.json")):
    r["prod"] = "none" if r["y"] == 0 else (r.get("prod") if r.get("prod") in PRODUCTS else None)
    rows.append(r)
old_dirs = {r["dir"] for r in rows}
for vd in sorted((ROOT / "data/crops/yt_test").glob("*/*")):
    rel = str(vd.relative_to(ROOT))
    if rel in old_dirs or not (vd / "frame_labels.json").exists():
        continue
    label = vd.parent.name
    fl = json.load(open(vd / "frame_labels.json"))
    prod_s = BY_VIDEO.get(vd.name) or LABEL2PROD.get(label)
    for code in "SOPN":
        ids = sorted(k for k, c in fl.items() if c == code)
        for sid in even(ids, PER):
            if code == "S":
                y = 0 if label in NOCAM else 1
                prod = prod_s if prod_s in PRODUCTS else None
                kind = prod or label
            elif code == "O":
                y, prod, kind = 1, None, "other_smart"
            else:
                y, prod, kind = 0, "none", ("plain_glasses" if code == "P" else "no_glasses")
            rows.append({"id": sid, "dir": rel, "y": y, "kind": kind, "group": vd.name, "source": "yt_test",
                         "prod": prod})
# 갤러리에서 사람이 다시 보고 고친 정답(묶음 대표 한 장만 보고 붙여 틀린 것). 값이 null 이면 제외.
ov = json.load(open(ROOT / "labels/test_overrides.json")) if (ROOT / "labels/test_overrides.json").exists() else {}
rows = [({**r, **ov[r["id"]]} if r["id"] in ov else r) for r in rows if not (r["id"] in ov and ov[r["id"]] is None)]
out = ROOT / "data/test_list.jsonl"
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
json.dump(rows, open(ROOT / "labels/test_list.json", "w"), ensure_ascii=False, indent=0)
c = Counter((r["y"], r["kind"]) for r in rows)
for (y, k), n in sorted(c.items(), key=lambda x: (-x[0][0], x[0][1])):
    vids = len({r["group"] for r in rows if r["kind"] == k and r["source"] == "yt_test"})
    print(f"  y={y} {k:14s} {n:4d} 장  (영상 {vids}편)")
print("total", len(rows), "->", out)
