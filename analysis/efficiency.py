# Author: <姓名> | Task: 效率测量 | PR: #<编号>
# Status: 本组原创实现
"""效率测量：参数量 / FLOPs / 单张推理时延 → results/efficiency.csv

为什么值得做：题目背景是"3D 打印在线质检"，只有准确率不够，还要证明轻量模型可部署。
这块直接支撑报告 3.x 的"调参讨论"与口头报告的效率对比页。

用法：
    python -m analysis.efficiency --models all --device cpu
"""

from __future__ import annotations

from pathlib import Path


def measure_params(model) -> tuple[int, int]:
    """(总参数量, 可训练参数量)"""
    raise NotImplementedError


def measure_flops(model, input_size: int = 224) -> float:
    """GFLOPs。可用 torch.utils.flop_counter（PyTorch ≥2.1）或 thop，后者需登记 NOTICE.md。"""
    raise NotImplementedError


def measure_latency(model, input_size: int = 224, device: str = "cpu",
                    warmup: int = 10, iters: int = 50) -> dict:
    """返回 {mean_ms, p95_ms}，必须 warmup 且多次取均值。"""
    raise NotImplementedError


def main() -> None:
    """对 11 个骨干跑一遍，写 results/efficiency.csv（列：model,n_params,n_params_trainable,gflops,cpu_ms,gpu_ms）。"""
    raise NotImplementedError


if __name__ == "__main__":
    main()
