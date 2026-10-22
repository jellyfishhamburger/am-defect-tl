# Author: <姓名> | Task: 数据预处理 | PR: #<编号>
# Status: 本组原创实现。参考：torchvision.transforms 文档（仅调用，未复制代码）
"""数据预处理：原始图 → 224×224 → data/processed/ + data/manifest.csv

规则（见 docs/DATA.md §3）：
  D1: 5184×3456，保持长宽比短边→224，中心裁剪 224×224
  D2: 128×128，直接双线性放大到 224

用法：
    python -m src.data_prepare --raw data/raw/AMdataset --out data/processed --size 224
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_manifest(raw_dir: Path) -> list[dict]:
    """扫描 raw_dir，返回 manifest 记录列表。

    每条记录字段：path, dataset, label, orig_w, orig_h, sha256, source_id
    - D1: dataset="D1", source_id = 文件名去扩展名（如 IMG_9078）
    - D2: dataset="D2", source_id = ""（作者已丢失增强溯源）
    """
    raise NotImplementedError("地基组实现：目录结构见 docs/DATA.md §1")


def preprocess(records: list[dict], out_dir: Path, size: int = 224) -> None:
    """按规则 resize/crop 并写入 out_dir/<dataset>/<label>/，同时回填 processed 路径。"""
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser(description="AMdataset 预处理")
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    records = build_manifest(args.raw)
    preprocess(records, args.out, args.size)
    # TODO: 写出 data/manifest.csv，并打印每个 dataset/label 的计数供人工核对
    #       期望 D1: 136/135，D2: 947/2070


if __name__ == "__main__":
    main()
