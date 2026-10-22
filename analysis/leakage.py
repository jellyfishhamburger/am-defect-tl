# Author: <姓名> | Task: 数据泄漏对照实验（本组自主贡献） | PR: #<编号>
# Status: 本组原创实现
"""泄漏对照实验：D2 随机划分 vs D2′ 按原图分组划分。

背景（docs/DATA.md §5-1）：作者把 D1 增强成 D2 后按 70/10/20 随机划分，同一张原图的
多个增强副本可能同时出现在 train 与 test → 准确率被高估。我们无法还原作者的映射，
所以自己做一份保留 source_id 的 D2′，做 A/B 对照。

预期结论：分组划分后准确率下降 → 定量证明原设定存在泄漏。

用法：
    python -m analysis.leakage --models vgg16,mobilenet_v2 --save-json
"""

from __future__ import annotations

from pathlib import Path


def build_d2prime(processed_d1: Path, out_dir: Path, aug_per_image: int = 10, seed: int = 42) -> Path:
    """从 D1 重做增强，保留 source_id。

    增强策略（需写进报告，作者未公开其增强细节）：
      旋转 ±15°、水平翻转、亮度/对比度抖动、缩放 0.9–1.1、高斯噪声
    输出 data/processed/D2p/<label>/ + 更新到 manifest（source_id = 原图名）
    """
    raise NotImplementedError


def run_ablation(models: list[str], split_a: Path, split_b: Path) -> list[dict]:
    """同一模型、同一超参，只换划分文件，跑两次并对比 10 个指标。"""
    raise NotImplementedError


def main() -> None:
    """结果写 results/leakage.csv 与 results/figures/fig_leakage.pdf。"""
    raise NotImplementedError


if __name__ == "__main__":
    main()
