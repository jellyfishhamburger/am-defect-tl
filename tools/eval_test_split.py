"""用保存的 checkpoint 复跑 D1 测试集（54 张），核对 metrics.json 里的 accuracy。

为什么需要：demo 上传图时出现判错，需要判断是"图难"还是"checkpoint 与报告值不一致"。
本脚本按训练同款预处理（data_prepare：draft → 224 → JPEG q92）逐张预测。

用法：
    python tools/eval_test_split.py --models vgg16,mobilenet_v2 --dataset D1
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = "https://raw.githubusercontent.com/mahsan2/AMdataset/main/dataset/{}/{}.JPG"
CACHE = Path(r"C:/Users/22390/AppData/Local/Temp/amcheck/testcache")


def prep(src_bytes: bytes, size: int = 224, quality: int = 92) -> Image.Image:
    img = Image.open(io.BytesIO(src_bytes))
    img.draft("RGB", (size * 2, size * 2))
    return img.convert("RGB").resize((size, size), Image.BILINEAR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="vgg16,mobilenet_v2,resnet50")
    ap.add_argument("--dataset", default="D1")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader((ROOT / "splits" / f"{args.dataset}.csv").open(encoding="utf-8"))
            if r["split"] == "test"]
    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"test rows = {len(rows)}", flush=True)

    paths = []
    for r in rows:
        folder = "badcylinder" if r["label"] == "defect" else "goodcylinder"
        dst = CACHE / f"{r['label']}_{r['source_id']}.jpg"
        if not dst.exists():
            raw = None
            for attempt in range(4):
                try:
                    raw = urllib.request.urlopen(RAW.format(folder, r["source_id"]), timeout=60).read()
                    break
                except Exception as exc:
                    print(f"  retry {attempt+1} {r['source_id']}: {type(exc).__name__}", flush=True)
                    time.sleep(2)
            if raw is None:
                print(f"  SKIP {r['source_id']} (download failed)", flush=True)
                continue
            prep(raw).save(dst, "JPEG", quality=92)
        paths.append((dst, r["label"], r["source_id"]))
    print("images ready", flush=True)

    from analysis.demo_app import predict_one  # noqa: E402

    for name in [m.strip() for m in args.models.split(",")]:
        run_dir = next((ROOT / "results" / "runs").glob(f"{name}__{args.dataset}__adam__*"), None)
        if run_dir is None:
            print(f"{name}: no checkpoint")
            continue
        ok, preds = 0, []
        for path, label, sid in paths:
            p = predict_one(run_dir, str(path))["p_defect"]
            pred = "defect" if p >= 0.5 else "nondefect"
            ok += pred == label
            preds.append((sid, label, pred, p))
        meta = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        claimed = meta["metrics"]["test"]["accuracy"]
        print(f"{name:16s} reproduced {ok}/{len(paths)} = {ok/len(paths):.3f} | "
              f"metrics.json says {claimed:.3f}", flush=True)
        wrong = [x for x in preds if x[1] != x[2]]
        if wrong:
            print("   wrong: " + ", ".join(f"{s}({l}->{p},{v:.2f})" for s, l, p, v in wrong), flush=True)


if __name__ == "__main__":
    main()
