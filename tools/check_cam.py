"""给 4 张示例图生成原图 + Grad-CAM 叠加，检查热区是否落在试样上。"""
import glob
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.demo_app import predict_one  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(r"C:/Users/22390/AppData/Local/Temp/amcheck")
OUT.mkdir(exist_ok=True)

imgs = sorted(glob.glob(str(ROOT / "samples/D1_test/*/*.jpg")))
run = ROOT / "results/runs/mobilenet_v2__D1__adam__s42"

tiles = []
for p in imgs:
    r = predict_one(run, p)
    cam = r["cam"]
    # 热区质心（相对坐标），看它落在画面哪一块
    ys, xs = np.unravel_index(np.argmax(cam), cam.shape)
    print(f"{Path(p).name:16s} p_defect={r['p_defect']:.3f}  CAM argmax at "
          f"({xs}/{cam.shape[1]}, {ys}/{cam.shape[0]})  cam max={cam.max():.2f}")
    tiles.append((Path(p).name, Image.open(p).resize((224, 224)).convert("RGB"), r["overlay"]))

w = 224 * len(tiles)
canvas = Image.new("RGB", (w, 224 * 2), "white")
for i, (name, orig, over) in enumerate(tiles):
    canvas.paste(orig, (i * 224, 0))
    canvas.paste(over, (i * 224, 224))
canvas.save(OUT / "cam_check.png")
print("saved", OUT / "cam_check.png")
