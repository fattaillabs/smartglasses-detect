"""스마트안경 판별 모델 학습.

입력: 안경 띠(512x192) 1장 + 테 모서리 조각(160x160) 2장.
출력: 스마트안경일 확률(2분류).

사용: python src/train.py --list data/train_list.jsonl --out models/run1 [--epochs 12]
train_list.jsonl 한 줄 = {"id","dir","y"(1=스마트안경,0=아님),"group"(같은 영상·같은 사람 묶음),"split"("train"|"val")}

출처 차이(조명·압축·해상도)를 배우지 않게 해상도 떨어뜨리기·JPEG 재압축·색 흔들기를 강하게 넣는다.
"""
import argparse
import io
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
BAND = (192, 512)   # (h, w)
CORNER = 160


def degrade(img: Image.Image, rnd: random.Random) -> Image.Image:
    """해상도와 압축 품질을 무작위로 떨어뜨린다(출처별 화질 차이를 지운다)."""
    w, h = img.size
    if rnd.random() < 0.7:
        s = rnd.uniform(0.35, 1.0)
        img = img.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR).resize((w, h), Image.BICUBIC)
    if rnd.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rnd.uniform(0.3, 1.2)))
    if rnd.random() < 0.7:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rnd.randint(35, 92))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img


CORNER_BOX = 168      # 띠에서 모서리를 잘라낼 때 한 변(띠 왼쪽 위·오른쪽 위 구석)
CORNER_OUT = 224      # 잘라낸 모서리를 키우는 크기


def cut_corners(band: Image.Image):
    """정렬된 띠의 양쪽 위 구석(테 끝, 카메라·LED 자리)을 잘라 키운다."""
    w, h = band.size
    c = min(CORNER_BOX, h)
    l = band.crop((0, 0, c, c)).resize((CORNER_OUT, CORNER_OUT), Image.BICUBIC)
    r = band.crop((w - c, 0, w, c)).resize((CORNER_OUT, CORNER_OUT), Image.BICUBIC)
    return l, r


def load_triplet(r, mode="file"):
    """(띠, 왼쪽 모서리, 오른쪽 모서리). mode='band' 면 모서리를 띠에서 잘라낸다."""
    d = Path(r["dir"])
    b = Image.open(d / "band" / f"{r['id']}.jpg").convert("RGB")
    if mode == "band":
        l, rr = cut_corners(b)
    else:
        l = Image.open(d / "corner" / f"{r['id']}_L.jpg").convert("RGB")
        rr = Image.open(d / "corner" / f"{r['id']}_R.jpg").convert("RGB")
    return b, l, rr


class Crops(Dataset):
    def __init__(self, rows, train: bool, mode: str = "file"):
        self.rows, self.train, self.mode = rows, train, mode
        self.color = T.ColorJitter(0.35, 0.35, 0.3, 0.04)
        self.norm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        b, l, rr = load_triplet(r, self.mode)
        if self.train:
            rnd = random.Random()
            flip = rnd.random() < 0.5
            fn_idx, bf, cf, sf, hf = T.ColorJitter.get_params(self.color.brightness, self.color.contrast,
                                                              self.color.saturation, self.color.hue)

            def cj(im):
                for f in fn_idx:
                    if f == 0: im = T.functional.adjust_brightness(im, bf)
                    elif f == 1: im = T.functional.adjust_contrast(im, cf)
                    elif f == 2: im = T.functional.adjust_saturation(im, sf)
                    else: im = T.functional.adjust_hue(im, hf)
                return im
            gray = rnd.random() < 0.08

            def aug(im, jit):
                im = cj(im)
                if gray:
                    im = im.convert("L").convert("RGB")
                dx, dy = rnd.randint(-jit, jit), rnd.randint(-jit, jit)
                im = T.functional.affine(im, angle=rnd.uniform(-3, 3), translate=(dx, dy),
                                         scale=rnd.uniform(0.94, 1.06), shear=0)
                return degrade(im, rnd)
            if self.mode == "band":
                # 띠에 변형을 먼저 주고 모서리는 그 띠에서 잘라 서로 어긋나지 않게 한다
                if flip:
                    b = b.transpose(Image.FLIP_LEFT_RIGHT)
                b = aug(b, 10)
                l, rr = cut_corners(b)
            else:
                if flip:   # 좌우 뒤집기: 띠를 뒤집고 모서리는 서로 바꾼다
                    b = b.transpose(Image.FLIP_LEFT_RIGHT)
                    l, rr = rr.transpose(Image.FLIP_LEFT_RIGHT), l.transpose(Image.FLIP_LEFT_RIGHT)
                b, l, rr = aug(b, 10), aug(l, 8), aug(rr, 8)
        # 오른쪽 모서리는 뒤집어 왼쪽과 같은 방향으로 맞춘다(가중치 공유)
        rr = rr.transpose(Image.FLIP_LEFT_RIGHT)
        pi = PRODUCTS.index(r["prod"]) if r.get("prod") in PRODUCTS else -1
        return self.norm(b), self.norm(l), self.norm(rr), torch.tensor(r["y"]), i, torch.tensor(pi)


# 제품 출력이 가르는 종류. "none" = 스마트안경 아님. 목록은 labels/products.json 에서 읽는다.
_PF = Path(__file__).resolve().parent.parent / "labels" / "products.json"
PRODUCTS = json.load(open(_PF)) if _PF.exists() else ["wayfarer", "headliner", "oakley_hstn"]


class Net(nn.Module):
    def __init__(self, backbone="resnet18.a1_in1k", pretrained=True, n_prod=len(PRODUCTS)):
        super().__init__()
        self.band = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.corner = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        f = self.band.num_features
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(f * 3, 2))
        # 제품 출력: 스마트안경일 때 어느 제품인지. n_prod=0 이면 만들지 않는다(예전 가중치 호환).
        self.prod = nn.Sequential(nn.Dropout(0.3), nn.Linear(f * 3, n_prod)) if n_prod else None

    def feats(self, b, l, r):
        return torch.cat([self.band(b), self.corner(l), self.corner(r)], 1)

    def forward(self, b, l, r):
        return self.head(self.feats(b, l, r))

    def both(self, b, l, r):
        f = self.feats(b, l, r)
        return self.head(f), (self.prod(f) if self.prod is not None else None)


def evaluate(model, dl, dev):
    model.eval()
    ps, ys, idx = [], [], []
    with torch.no_grad():
        for b, l, r, y, i, _ in dl:
            p = torch.softmax(model(b.to(dev), l.to(dev), r.to(dev)), 1)[:, 1]
            ps += p.cpu().tolist(); ys += y.tolist(); idx += i.tolist()
    return np.array(ps), np.array(ys), idx


def report(ps, ys, rows, idx, thr=0.5):
    pred = ps >= thr
    tp = int((pred & (ys == 1)).sum()); fp = int((pred & (ys == 0)).sum())
    fn = int((~pred & (ys == 1)).sum()); tn = int((~pred & (ys == 0)).sum())
    by = defaultdict(lambda: [0, 0])
    for p, y, i in zip(pred, ys, idx):
        k = f"{rows[i].get('source', '?')}/{rows[i]['group']}"
        by[k][0] += int(p == bool(y)); by[k][1] += 1
    return {"acc": round((tp + tn) / max(1, len(ys)), 4),
            "recall": round(tp / max(1, tp + fn), 4), "fpr": round(fp / max(1, fp + tn), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}, {k: f"{a}/{n}" for k, (a, n) in sorted(by.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--backbone", default="resnet18.a1_in1k")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--corner", default="file", choices=["file", "band"], help="모서리 조각: 따로 잘라 둔 파일 / 띠에서 잘라내기")
    ap.add_argument("--final", action="store_true", help="검증 몫까지 전부 학습에 쓰고 마지막 회차를 저장(테스트 묶음은 따로 있음)")
    ap.add_argument("--samples", type=int, default=6000, help="한 회에 뽑는 표본 수")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in open(a.list)]
    tr = rows if a.final else [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    print("train", len(tr), "pos", sum(r["y"] for r in tr), "| val", len(va), "pos", sum(r["y"] for r in va), flush=True)

    # 표본 가중치: 양·음 균형 + 같은 묶음(영상) 안의 비슷한 프레임이 지배하지 않게
    cnt_g = defaultdict(int)
    for r in tr:
        cnt_g[(r["y"], r.get("source"), r["group"])] += 1
    n_groups = defaultdict(int)
    for (y, s, g) in cnt_g:
        n_groups[y] += 1
    cap = 60   # 묶음 하나가 받는 가중치 총량 상한(프레임 수 기준)
    w = []
    for r in tr:
        c = cnt_g[(r["y"], r.get("source"), r["group"])]
        w.append(min(c, cap) / c)
    tot = defaultdict(float)
    for r, x in zip(tr, w):
        tot[r["y"]] += x
    w = [x / tot[r["y"]] for r, x in zip(tr, w)]
    sampler = WeightedRandomSampler(w, num_samples=min(len(tr), a.samples), replacement=True)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dl_tr = DataLoader(Crops(tr, True, a.corner), batch_size=a.bs, sampler=sampler, num_workers=a.workers, drop_last=True,
                       persistent_workers=a.workers > 0)
    dl_va = DataLoader(Crops(va, False, a.corner), batch_size=64, shuffle=False, num_workers=a.workers,
                       persistent_workers=a.workers > 0)
    model = Net(a.backbone).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    # 제품별 장수가 크게 달라(웨이페어러가 대부분) 적은 제품에 가중치를 준다
    cnt = [max(1, sum(1 for r in tr if r.get("prod") == p)) for p in PRODUCTS]
    pw = torch.tensor([max(cnt) / c for c in cnt], dtype=torch.float32).clamp(max=6).to(dev)
    prodf = nn.CrossEntropyLoss(weight=pw, ignore_index=-1, label_smoothing=0.05)
    best = -1
    for ep in range(a.epochs):
        model.train(); run = 0.0
        for k, (b, l, r, y, _, pi) in enumerate(dl_tr):
            opt.zero_grad()
            lo, lp = model.both(b.to(dev), l.to(dev), r.to(dev))
            loss = lossf(lo, y.to(dev))
            if (pi >= 0).any():
                loss = loss + 0.5 * prodf(lp, pi.to(dev))
            loss.backward(); opt.step(); sched.step()
            run += loss.item()
            if (k + 1) % 100 == 0:
                print(f"  ep{ep} step {k + 1}/{len(dl_tr)} loss {run / (k + 1):.4f}", flush=True)
        ps, ys, idx = evaluate(model, dl_va, dev)
        m, by = report(ps, ys, va, idx)
        score = m["recall"] - m["fpr"]
        print(f"ep{ep} loss {run / len(dl_tr):.4f} val {m}", flush=True)
        if a.final:
            score = ep   # 고르지 않고 마지막 회차를 남긴다
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "backbone": a.backbone, "corner": a.corner, "products": PRODUCTS}, out / "best.pt")
            json.dump({"epoch": ep, "metrics": m, "by_group": by}, open(out / "best.json", "w"), ensure_ascii=False, indent=1)
            np.save(out / "val_probs.npy", np.stack([ps, ys, np.array(idx)]))
    print("best", json.load(open(out / "best.json"))["metrics"])


if __name__ == "__main__":
    main()
