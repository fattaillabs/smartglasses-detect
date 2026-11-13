"""영상마다 비슷한 프레임끼리 묶어, 묶음 단위로 정답을 붙일 수 있게 시트를 만든다.

프레임을 한 장씩 보기에는 많으므로, 모델 특징으로 영상 안의 프레임을 몇 묶음으로 나누고
묶음마다 대표 한 장을 시트에 싣는다. 사람이 묶음 번호별로 종류를 정하면 그 묶음의 프레임 전체에 적용한다.

사용: python src/cluster_label.py <모델 폴더> <출력 이름> <영상 crops 폴더> [...]
결과: eval/<출력 이름>.json (묶음 → 프레임 목록), eval/sheets/<출력 이름>_<쪽>.jpg
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import load_model  # noqa: E402
from train import MEAN, STD, load_triplet  # noqa: E402

BW, BH = 456, 171


def features(model, dev, d: Path, ids, bs=64):
    norm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    mode = getattr(model, "corner_mode", "file")
    out = []
    for k in range(0, len(ids), bs):
        trip = [load_triplet({"dir": str(d), "id": i}, mode) for i in ids[k:k + bs]]
        b = torch.stack([norm(t[0]) for t in trip]).to(dev)
        l = torch.stack([norm(t[1]) for t in trip]).to(dev)
        r = torch.stack([norm(t[2].transpose(Image.FLIP_LEFT_RIGHT)) for t in trip]).to(dev)
        with torch.no_grad():
            f = model.feats(b, l, r)
        out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1))


def kmeans(x, k, iters=25, seed=0):
    rnd = np.random.RandomState(seed)
    c = x[rnd.choice(len(x), k, replace=False)]
    for _ in range(iters):
        a = (x @ c.T).argmax(1)
        for j in range(k):
            if (a == j).any():
                c[j] = x[a == j].mean(0)
                c[j] /= np.linalg.norm(c[j]) + 1e-9
    return (x @ c.T).argmax(1), c


def main():
    model_dir, name, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
    model, dev = load_model(model_dir)
    clusters = []
    for d in map(Path, dirs):
        ids = [f.stem for f in sorted((d / "band").glob("*.jpg"))]
        if not ids:
            continue
        x = features(model, dev, d, ids)
        k = max(1, min(6, len(ids) // 12))
        a, c = kmeans(x, k) if k > 1 else (np.zeros(len(ids), dtype=int), x.mean(0, keepdims=True))
        for j in range(k):
            m = np.where(a == j)[0]
            if len(m) == 0:
                continue
            rep = m[(x[m] @ c[j]).argmax()]       # 묶음 한가운데에 가장 가까운 프레임
            clusters.append({"dir": str(d), "label": d.parent.name, "video": d.name, "rep": ids[rep],
                             "ids": [ids[i] for i in m]})
    for n, cl in enumerate(clusters):
        cl["n"] = n
    json.dump(clusters, open(f"eval/{name}.json", "w"))
    per, cols = 42, 3
    for page in range((len(clusters) + per - 1) // per):
        part = clusters[page * per:(page + 1) * per]
        rows = (len(part) + cols - 1) // cols
        sheet = Image.new("RGB", (BW * cols, (BH + 14) * rows), "white")
        dr = ImageDraw.Draw(sheet)
        for k, cl in enumerate(part):
            x0, y0 = (k % cols) * BW, (k // cols) * (BH + 14)
            sheet.paste(Image.open(Path(cl["dir"]) / "band" / f"{cl['rep']}.jpg").resize((BW, BH)), (x0, y0))
            test = "T " if "yt_test" in cl["dir"] else ""
            dr.text((x0 + 2, y0 + BH + 1), f"#{cl['n']} {test}{cl['label']} {cl['video'][:11]} x{len(cl['ids'])}", fill="black")
        sheet.save(f"eval/sheets/{name}_{page}.jpg", quality=88)
    print(len(clusters), "clusters", "pages", (len(clusters) + per - 1) // per)


if __name__ == "__main__":
    main()
