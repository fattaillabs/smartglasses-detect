"""학습된 모델(best.pt)을 공개용 형식(safetensors 반정밀도 + config.json)으로 내보낸다.

사용: python src/export.py <모델 폴더> [--thr 0.5] [--out models/release]
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--thr", type=float, default=0.5)
ap.add_argument("--out", default="models/release")
a = ap.parse_args()
ck = torch.load(Path(a.model) / "best.pt", map_location="cpu")
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
# 반정밀도로 저장해 크기를 절반으로(배치 정규화의 정수 카운터는 그대로 둔다)
save_file({k: (v.half() if v.is_floating_point() else v) for k, v in ck["model"].items()}, str(out / "model.safetensors"))
json.dump({"backbone": ck["backbone"], "corner": ck["corner"], "products": ck["products"],
           "no_camera_products": ["eveng2", "xreal"], "threshold": a.thr,
           "input": {"band": [192, 512], "face_width_px": 400, "corner_box": 168, "corner_out": 224,
                     "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
           "license": "CC BY-NC-SA 4.0"}, open(out / "config.json", "w"), indent=1)
print(out, round((out / "model.safetensors").stat().st_size / 1e6, 1), "MB", "products", len(ck["products"]))
