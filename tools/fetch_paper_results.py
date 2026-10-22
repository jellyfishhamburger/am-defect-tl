# Author: 地基组 | Task: 从作者仓库抽取 99 份结果表，生成对标基线 | PR: #<编号>
# Status: 本组原创实现（作者仓库只有 xlsx，没有代码可复用）
"""把作者 AM3d 仓库里 99 份 xlsx（3 研究 × 3 优化器 × 11 模型）抽成一张对照表。

为什么需要它：
  - 8 个人跑出来的 66 个 run 需要和论文数值逐格对比，手抄一周也抄不完；
  - 论文正文（付费墙）没有全表，但作者仓库的 xlsx 里有完整指标。

输出：docs/paper_results.csv
  列：dataset,optimizer,model,split,accuracy,precision,recall,f1,mcc,kappa,auc,map,sensitivity,specificity,source_url
  其中 dataset 映射：Studyone→D1, Studytwo→D2, Studythree→D3(作者未公开数据)

用法：
    python tools/fetch_paper_results.py            # 需联网，约 1–2 分钟
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO = "mahsan2/AM3d"
BRANCH = "main"
API = f"https://api.github.com/repos/{REPO}/contents"
RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/"
UA = {"User-Agent": "am-defect-tl"}

STUDY_MAP = {"Studyone": "D1", "Studytwo": "D2", "Studythree": "D3"}
OPT_MAP = {"Adam": "adam", "Sgd": "sgd", "rmsprop": "rmsprop"}
SPLIT_MAP = {"Train": "train", "Validation": "val", "Test": "test"}

# 文件名前缀 → configs/*.yaml 里的 model 字段
MODEL_PATTERNS = [
    ("inceptionresnet", "inception_resnet_v2"),
    ("mobilenet", "mobilenet_v2"),
    ("nasnetlargemodel", "nasnet_large"),
    ("nasnetlarge", "nasnet_large"),
    ("netb0", "efficientnet_b0"),
    ("netb7", "efficientnet_b7"),
    ("v2m", "efficientnet_v2m"),
    ("resnet101", "resnet101"),
    ("resnet50", "resnet50"),
    ("vgg16", "vgg16"),
    ("vgg19", "vgg19"),
    ("xception", "xception"),
]

METRIC_ORDER = [
    "accuracy", "precision", "recall", "f1", "mcc", "kappa",
    "auc", "map", "sensitivity", "specificity",
]


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=60).read()


def list_dir(path: str) -> list[dict]:
    return json.loads(_get(f"{API}/{path}?ref={BRANCH}"))


def model_key(filename: str) -> str | None:
    stem = re.sub(r"model_results\.xlsx$", "", filename, flags=re.I).lower()
    for pattern, key in MODEL_PATTERNS:
        if stem.startswith(pattern):
            return key
    return None


def parse_xlsx(blob: bytes) -> list[dict]:
    """读 Metrics sheet，返回 3 条记录（train / val / test）。"""
    try:
        import openpyxl
    except ImportError:  # pragma: no cover
        sys.exit("需要 openpyxl：pip install openpyxl")

    wb = openpyxl.load_workbook(io.BytesIO(blob), data_only=True)
    if "Metrics" not in wb.sheetnames:
        return []
    ws = wb["Metrics"]

    header = [str(c.value).strip() if c.value else "" for c in ws[1]]
    # 表头形如 ['', 'Accuracy', 'Precision', ...]，建立 列名 → 列索引
    col = {}
    for idx, name in enumerate(header):
        key = name.lower().replace(" ", "").replace("-", "")
        alias = {
            "f1score": "f1", "accuracy": "accuracy", "precision": "precision",
            "recall": "recall", "mcc": "mcc", "kappa": "kappa", "auc": "auc",
            "map": "map", "sensitivity": "sensitivity", "specificity": "specificity",
        }
        if key in alias:
            col[alias[key]] = idx

    rows_out = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or row[0] not in SPLIT_MAP:
            continue
        rec = {"split": SPLIT_MAP[str(row[0]).strip()]}
        for metric in METRIC_ORDER:
            idx = col.get(metric)
            value = row[idx] if idx is not None and idx < len(row) else None
            try:
                rec[metric] = round(float(value), 4)
            except (TypeError, ValueError):
                rec[metric] = ""
        rows_out.append(rec)
    return rows_out


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "docs" / "paper_results.csv"
    records: list[dict] = []
    missing: list[str] = []

    for study_dir, dataset in STUDY_MAP.items():
        for opt_dir, optimizer in OPT_MAP.items():
            listing = list_dir(f"{study_dir}/{opt_dir}")
            for item in listing:
                name = item["name"]
                if not name.lower().endswith("model_results.xlsx"):
                    continue
                model = model_key(name)
                if model is None:
                    missing.append(f"{study_dir}/{opt_dir}/{name}")
                    continue
                blob = _get(RAW + f"{study_dir}/{opt_dir}/{name}")
                for rec in parse_xlsx(blob):
                    records.append({
                        "dataset": dataset, "optimizer": optimizer, "model": model,
                        **rec, "source_url": f"https://github.com/{REPO}/blob/{BRANCH}/{study_dir}/{opt_dir}/{name}",
                    })

    fields = ["dataset", "optimizer", "model", "split", *METRIC_ORDER, "source_url"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    print(f"wrote {out} : {len(records)} rows "
          f"(期望 3 研究 × 3 优化器 × 11 模型 × 3 split = 297)")
    if missing:
        print("未识别的文件（需补充 MODEL_PATTERNS）:")
        for m in missing:
            print("  -", m)


if __name__ == "__main__":
    main()
