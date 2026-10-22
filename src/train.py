# Author: <姓名> | Task: 训练循环 | PR: #<编号>
# Status: 本组原创实现
"""单个 run 的训练入口。结果写入 results/runs/<run_id>/（字段见 docs/RESULTS.md）。

用法：
    python -m src.train --config configs/vgg16.yaml
    python -m src.train --config configs/vgg16.yaml --optimizer sgd --dataset D2
"""

from __future__ import annotations

import argparse
from pathlib import Path


def load_config(path: Path) -> dict:
    """读 yaml 并做基本校验（必填字段、取值合法性）。"""
    raise NotImplementedError


def set_seed(seed: int) -> None:
    """固定 python / numpy / torch / cuda 的随机性，保证可复现。"""
    raise NotImplementedError


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None) -> dict:
    raise NotImplementedError


def evaluate(model, loader, device) -> tuple[dict, list[list[int]]]:
    """返回 (指标字典, 混淆矩阵)。指标口径见 docs/RESULTS.md §4，positive = defect。"""
    raise NotImplementedError


def run(config: dict) -> Path:
    """完整流程：数据 → 模型 → 训练 → 评估 → 保存 4 个产物 → 返回 run 目录。"""
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser(description="训练一个 run")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--optimizer", default=None, help="覆盖 config")
    ap.add_argument("--dataset", default=None, help="覆盖 config")
    ap.add_argument("--seed", type=int, default=None, help="覆盖 config")
    ap.add_argument("--dry-run", action="store_true", help="只构建模型与前 2 个 batch，用于冒烟")
    args = ap.parse_args()

    config = load_config(args.config)
    # TODO: 命令行覆盖 → set_seed → run() → 打印 run_id 与关键指标


if __name__ == "__main__":
    main()
