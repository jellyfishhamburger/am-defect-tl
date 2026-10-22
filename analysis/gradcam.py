# Author: <姓名> | Task: Grad-CAM 可解释性 | PR: #<编号>
# Status: 本组原创实现（建议自己写，约 40 行；若改用 grad-cam 库须在 NOTICE.md 登记）
"""Grad-CAM：对每个模型输出"模型在看哪里"的热力图，回答"它是不是真的在看缺陷处"。

用法：
    python -m analysis.gradcam --run results/runs/vgg16__D1__adam__s42 --n 8
"""

from __future__ import annotations

from pathlib import Path


def gradcam(model, image_tensor, target_layer, target_class: int = 1):
    """返回 CAM 热力图（H×W，0–1）。步骤：前向 → 取目标层梯度 → 全局平均 → 加权求和 → ReLU。"""
    raise NotImplementedError


def overlay_heatmap(image, cam, alpha: float = 0.5):
    raise NotImplementedError


def main() -> None:
    """对指定 run 抽取 N 张测试图，输出 results/figures/gradcam_<model>_<i>.png，并拼成一张总图。"""
    raise NotImplementedError


if __name__ == "__main__":
    main()
