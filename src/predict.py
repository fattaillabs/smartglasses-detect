"""학습된 모델로 crops 폴더의 조각마다 확률을 매겨 <폴더>/<이름>.json 에 저장한다.

사용: python src/predict.py <모델 폴더(best.pt)> <저장 이름> <crops 폴더> [<crops 폴더> ...]
예: python src/predict.py models/wear glasses data/crops/yt/rbmeta/*   → 각 폴더에 glasses.json
"""
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import MEAN, STD, Net, load_triplet  # noqa: E402


def load_model(model_dir):
    ck = torch.load(Path(model_dir) / "best.pt", map_location="cpu")
    m = Net(ck["backbone"], pretrained=False, n_prod=len(ck.get("products", [])))
    m.products = ck.get("products", [])
    m.load_state_dict(ck["model"])
    m.corner_mode = ck.get("corner", "file")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    return m.to(dev).eval(), dev


def predict_dir(model, dev, d: Path, bs=64):
    norm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    ids = [f.stem for f in sorted((d / "band").glob("*.jpg"))]
    mode = getattr(model, "corner_mode", "file")
    out = {}
    for k in range(0, len(ids), bs):
        chunk = ids[k:k + bs]
        trip = [load_triplet({"dir": str(d), "id": i}, mode) for i in chunk]
        b = torch.stack([norm(t[0]) for t in trip])
        l = torch.stack([norm(t[1]) for t in trip])
        r = torch.stack([norm(t[2].transpose(Image.FLIP_LEFT_RIGHT)) for t in trip])
        with torch.no_grad():
            b, l, r = b.to(dev), l.to(dev), r.to(dev)
            p1 = torch.softmax(model(b, l, r), 1)[:, 1]
            # 좌우를 뒤집은 것도 함께 보고 평균(뒤집으면 왼쪽·오른쪽 모서리가 서로 바뀐다)
            p2 = torch.softmax(model(torch.flip(b, dims=[3]), r, l), 1)[:, 1]
            p = ((p1 + p2) / 2).cpu().tolist()
        out.update({i: round(x, 4) for i, x in zip(chunk, p)})
    return out


def main():
    model_dir, name, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
    model, dev = load_model(model_dir)
    for d in map(Path, dirs):
        if not (d / "band").exists():
            continue
        res = predict_dir(model, dev, d)
        json.dump(res, open(d / f"{name}.json", "w"))
        v = list(res.values())
        print(d, "n", len(v), ">0.6:", sum(p > 0.6 for p in v), "<0.1:", sum(p < 0.1 for p in v),
              "between:", sum(0.1 <= p <= 0.6 for p in v), flush=True)


if __name__ == "__main__":
    main()
