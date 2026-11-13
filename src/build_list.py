"""수집한 조각들을 모아 학습/검증 목록(data/train_list.jsonl)을 만든다.

정답(y): 1 = 카메라 달린 스마트안경 착용, 0 = 그 밖(일반 안경, 안경 없음, 카메라 없는 스마트안경).
제품(prod): labels/products.json 의 이름 중 하나. "none" = 스마트안경 아님. None = 스마트안경이지만 제품은 모름(제품 학습에서 뺌).

출처별 규칙
- 묶음 검수를 거친 영상(frame_labels.json 있음): 프레임마다 S/O/P/N/X
    S=그 영상의 제품  O=다른 스마트안경(제품 모름)  P=일반 안경  N=안경 없음  X=제외
- 초기 수집 영상(rbmeta, oakmeta): 착용 판정기 확률로 착용/맨얼굴을 가르고 제품은 영상 단위 표(product_by_video.json)
- 일반 안경 영상(rbplain): 전부 0
- 웹 사진: rbmeta·rbplain·plain2 는 검수 파일(relabel.json, exclude.txt) 반영, p_<제품> 은 검수한 것만
- FFHQ: 전부 0
같은 영상·같은 사진이 학습과 검증에 걸치지 않게 영상 단위·사진 단위로 나눈다.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CROPS = ROOT / "data/crops"
G_HI, G_LO = 0.6, 0.1
PRODUCTS = json.load(open(ROOT / "labels/products.json"))
NOCAM = {"eveng2", "xreal"}                 # 카메라 없는 스마트안경: y=0, 제품은 알아본다
YT_PLAIN = ("rbplain",)
# 영상 폴더 라벨 → 제품. 여기 없는 라벨(mixed, clone, blazer, scriber ...)은 제품을 붙이지 않는다.
LABEL2PROD = {"wayfarer": "wayfarer", "oakmeta": "oakley_hstn", "adventurer": "adventurer", "fury": "fury",
              "nova": "nova", "vanguard": "vanguard", "display": "display", "rokid": "rokid",
              "samsung": "samsung", "xiaomi": "xiaomi", "eveng2": "eveng2", "xreal": "xreal", "skyler": "skyler"}
BY_VIDEO = json.load(open(ROOT / "labels/product_by_video.json")) if (ROOT / "labels/product_by_video.json").exists() else {}


def ahash(path: Path) -> int:
    a = np.asarray(Image.open(path).convert("L").resize((16, 8), Image.BILINEAR), dtype=np.float32)
    bits = (a > a.mean()).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def bucket(key: str, mod: int = 100) -> int:
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % mod


def load_dir(d: Path):
    g = json.load(open(d / "glasses.json")) if (d / "glasses.json").exists() else {}
    excl = set((d / "exclude.txt").read_text().split()) if (d / "exclude.txt").exists() else set()
    rel = json.load(open(d / "relabel.json")) if (d / "relabel.json").exists() else {}
    for f in sorted((d / "band").glob("*.jpg")):
        if f.stem in excl or f.stem not in g:
            continue
        yield f.stem, g[f.stem], rel.get(f.stem)


def frame_rows(vd: Path, label: str, source: str):
    """묶음 검수 결과(S/O/P/N/X)를 프레임별 정답으로 바꾼다."""
    fl = json.load(open(vd / "frame_labels.json"))
    prod_s = BY_VIDEO.get(vd.name) or LABEL2PROD.get(label)
    for sid, c in fl.items():
        if c == "X" or not (vd / "band" / f"{sid}.jpg").exists():
            continue
        if c == "S":
            y, prod = (0 if label in NOCAM else 1), (prod_s if prod_s in PRODUCTS else None)
        elif c == "O":
            y, prod = 1, None
        else:
            y, prod = 0, "none"
        yield {"id": sid, "dir": str(vd), "y": y, "group": vd.name, "source": source,
               "wearing": int(c != "N"), "prod": prod, "code": c}


def main():
    rows = []
    # 유튜브
    for ld in sorted((CROPS / "yt").glob("*")):
        label = ld.name
        vids = sorted(p for p in ld.glob("*") if (p / "band").exists() and any((p / "band").iterdir()))
        for k, vd in enumerate(vids):
            split = "val" if k % 4 == 3 else "train"
            if (vd / "frame_labels.json").exists():
                for r in frame_rows(vd, label, f"yt_{label}"):
                    rows.append({**r, "split": split})
                continue
            if label not in ("rbmeta", "oakmeta") + YT_PLAIN:
                continue      # 검수하지 않은 새 영상은 쓰지 않는다
            for sid, g, rel in load_dir(vd):
                if G_LO <= g <= G_HI:
                    continue
                wearing = g > G_HI
                y = rel if rel is not None else int(wearing and label not in YT_PLAIN)
                prod = "none" if y == 0 else (BY_VIDEO.get(vd.name) or LABEL2PROD.get(label))
                rows.append({"id": sid, "dir": str(vd), "y": y, "group": vd.name, "source": f"yt_{label}",
                             "wearing": int(wearing), "split": split, "prod": prod if prod in PRODUCTS else None})
    # 웹: 같은 보도 사진이 여러 주소로 올라오므로 비슷한 것은 하나만 남긴다(제품 정답이 붙은 p_* 를 먼저)
    seen = []
    web = sorted(p.name for p in (CROPS / "web").glob("p_*")) + ["rbmeta", "rbplain", "plain2"]
    for label in web:
        d = CROPS / "web" / label
        if not (d / "band").exists():
            continue
        manual_only = label.startswith("p_")
        if manual_only and not (d / "relabel.json").exists():
            continue
        for sid, g, rel in load_dir(d):
            if manual_only and rel is None:
                continue
            if G_LO <= g <= G_HI and rel is None:
                continue
            wearing = g > G_HI or rel is not None
            if label == "rbmeta" and not wearing and rel is None:
                continue
            h = ahash(d / "band" / f"{sid}.jpg")
            if any(bin(h ^ o).count("1") <= 8 for o in seen):
                continue
            seen.append(h)
            y = rel if rel is not None else int(wearing and label == "rbmeta")
            prod = "none" if y == 0 else (label[2:] if manual_only and label[2:] in PRODUCTS else None)
            group = sid.rsplit("_f", 1)[0]
            rows.append({"id": sid, "dir": str(d), "y": y, "group": group, "source": f"web_{label}",
                         "wearing": int(wearing), "split": "val" if bucket(group) < 40 else "train", "prod": prod})
    # FFHQ
    for label in ("glasses", "noglasses"):
        d = CROPS / "ffhq" / label
        if not (d / "band").exists():
            continue
        for sid in sorted(f.stem for f in (d / "band").glob("*.jpg")):
            rows.append({"id": sid, "dir": str(d), "y": 0, "group": sid, "source": f"ffhq_{label}",
                         "wearing": int(label == "glasses"), "split": "val" if bucket(sid) < 15 else "train",
                         "prod": "none"})
    out = ROOT / "data/train_list.jsonl"
    for r in rows:
        r["dir"] = str(Path(r["dir"]).relative_to(ROOT))   # 저장소 기준 상대 경로
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("y=1", sum(r["y"] for r in rows), "y=0", sum(1 - r["y"] for r in rows), "total", len(rows))
    prod = Counter(r["prod"] for r in rows)
    vids = {p: len({r["group"] for r in rows if r["prod"] == p and r["source"].startswith("yt")}) for p in PRODUCTS}
    for p in PRODUCTS:
        print(f"  {p:12s} {prod.get(p, 0):5d} 장  영상 {vids[p]}편")
    print(f"  (제품 모름)  {prod.get(None, 0):5d} 장")
    print("->", out)


if __name__ == "__main__":
    main()
