"""테스트 목록(data/test_list.jsonl)으로 모델을 평가한다.

사용: python src/evaluate.py <모델 폴더> [--list data/test_list.jsonl] [--thr 0.5] [--out eval/test_<이름>.json]
종류별(레이밴 메타·오클리 메타·일반 안경·맨얼굴) 정답률과 전체 재현율·오탐률을 내고,
조각별 확률을 저장한다(갤러리와 오답 분석에 쓴다).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import load_model  # noqa: E402
from train import MEAN, STD, load_triplet  # noqa: E402


def run(model, dev, rows, bs=64):
    norm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    mode = getattr(model, "corner_mode", "file")
    out, prod = [], []
    for k in range(0, len(rows), bs):
        chunk = rows[k:k + bs]
        trip = [load_triplet(r, mode) for r in chunk]
        b = torch.stack([norm(t[0]) for t in trip]).to(dev)
        l = torch.stack([norm(t[1]) for t in trip]).to(dev)
        r = torch.stack([norm(t[2].transpose(Image.FLIP_LEFT_RIGHT)) for t in trip]).to(dev)
        with torch.no_grad():
            o1, q1 = model.both(b, l, r)
            o2, q2 = model.both(torch.flip(b, dims=[3]), r, l)
        p = (torch.softmax(o1, 1)[:, 1] + torch.softmax(o2, 1)[:, 1]) / 2
        out += p.cpu().tolist()
        if q1 is not None:
            prod += ((torch.softmax(q1, 1) + torch.softmax(q2, 1)) / 2).argmax(1).cpu().tolist()
        else:
            prod += [-1] * len(chunk)
    return out, prod


def summarize(rows, ps, thr):
    by = defaultdict(lambda: [0, 0])
    tp = fp = fn = tn = 0
    for r, p in zip(rows, ps):
        pred = p >= thr
        ok = pred == bool(r["y"])
        by[(r["source"], r["kind"])][0] += ok
        by[(r["source"], r["kind"])][1] += 1
        if r["y"]:
            tp += pred; fn += not pred
        else:
            fp += pred; tn += not pred
    return {"n": len(rows), "acc": round((tp + tn) / len(rows), 4), "recall": round(tp / max(1, tp + fn), 4),
            "fpr": round(fp / max(1, fp + tn), 4), "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)}, \
           {f"{s}/{k}": f"{a}/{n}" for (s, k), (a, n) in sorted(by.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--list", default="data/test_list.jsonl")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = [json.loads(x) for x in open(a.list)]
    model, dev = load_model(a.model)
    ps, pr = run(model, dev, rows)
    m, by = summarize(rows, ps, a.thr)
    print(json.dumps(m))
    for k, v in by.items():
        print(f"  {k:28s} {v}")
    names = getattr(model, "products", [])
    if names:
        # 제품 구분: 제품 정답이 있는 스마트안경만, 표(행=정답, 열=예측)
        conf = defaultdict(lambda: defaultdict(int))
        for r, q in zip(rows, pr):
            if r.get("prod") in names:
                conf[r["prod"]][names[q]] += 1
        tot = sum(sum(v.values()) for v in conf.values())
        hit = sum(conf[n][n] for n in names)
        print(f"product acc {hit}/{tot}")
        for n in names:
            print(f"  {n:12s}", {m: conf[n][m] for m in names})
    out = a.out or f"eval/test_{Path(a.model).name}.json"
    json.dump([{**r, "p": round(p, 4), "pred_prod": (names[q] if names and q >= 0 else None)}
               for r, p, q in zip(rows, ps, pr)], open(out, "w"), ensure_ascii=False)
    print("saved", out)


if __name__ == "__main__":
    main()
