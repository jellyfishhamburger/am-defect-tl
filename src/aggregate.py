# Author: <姓名> | Task: 结果汇总与论文图 | PR: #<编号>
# Status: 本组原创实现
"""汇总 results/runs/*/metrics.json → results/leaderboard.csv + results/figures/*

用法：
    python -m src.aggregate            # 生成表与全部图
    python -m src.aggregate --check    # 只做完整性检查（66 个 run 是否齐全、字段是否合法）
"""

from __future__ import annotations

from pathlib import Path

REQUIRED_RUNS = 66  # 11 models × 3 optimizers × 2 datasets

# 作者 xlsx 的数值对照表：<model>__<dataset>__<optimizer> → (acc, f1)
# 从 https://github.com/mahsan2/AM3d 的 99 个 xlsx 读取；Study Three 的行留空
PAPER_RESULTS_CSV = Path("docs/paper_results.csv")


def collect_runs(runs_dir: Path = Path("results/runs")) -> list[dict]:
    """读所有 metrics.json，做 schema 校验，返回记录列表。"""
    raise NotImplementedError


def load_paper_results(path: Path = PAPER_RESULTS_CSV) -> dict:
    """读作者数值（需要有人先把 99 个 xlsx 抽成一张 csv，见 docs/RESULTS.md §6）。"""
    raise NotImplementedError


def write_leaderboard(records: list[dict], out: Path = Path("results/leaderboard.csv")) -> None:
    """列定义见 docs/RESULTS.md §6。包含 delta_acc = 我们的 accuracy - 作者 accuracy。"""
    raise NotImplementedError


def make_all_figures(records: list[dict], out_dir: Path = Path("results/figures")) -> None:
    """生成 docs/RESULTS.md §7 要求的 6 组必备图。"""
    raise NotImplementedError
