# Author: <姓名> | Task: 数据划分（契约文件） | PR: #<编号>
# Status: 本组原创实现。参考：scikit-learn train_test_split 文档（仅调用）
"""生成 splits/{D1,D2,D2p_group}.csv —— 全组唯一一份划分，冻结后不得修改。

列：path,label,split,source_id（定义见 splits/README.md）

用法：
    python -m src.splits --dim D1  --seed 42
    python -m src.splits --dim D2  --seed 42
    python -m src.splits --dim D2p --group-by source_id --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

SPLIT_RATIO = (0.7, 0.1, 0.2)  # train / val / test，对齐作者


def load_manifest(manifest_csv: Path, dataset: str) -> list[dict]:
    """从 data/manifest.csv 取某个 dataset 的记录。"""
    raise NotImplementedError


def make_split(records: list[dict], *, stratify: bool = True, group_by: str | None = None,
               seed: int = 42) -> list[dict]:
    """分层划分 70/10/20。

    group_by=None → 普通分层抽样；
    group_by="source_id" → GroupShuffleSplit，同一 source_id 只出现在一个 split 里
                          （泄漏对照实验的核心，不能省）。
    """
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser(description="生成数据划分")
    ap.add_argument("--dim", required=True, choices=["D1", "D2", "D2p"])
    ap.add_argument("--group-by", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("splits"))
    args = ap.parse_args()

    records = load_manifest(args.manifest, args.dim)
    rows = make_split(records, group_by=args.group_by, seed=args.seed)
    # TODO: 写 csv（不含 index，utf-8，lf 换行），并打印三个 split 的类别分布


if __name__ == "__main__":
    main()
