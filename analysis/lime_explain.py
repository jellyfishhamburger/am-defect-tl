# Author: <姓名> | Task: LIME 可解释性（对齐论文） | PR: #<编号>
# Status: 使用 lime 库（开源，见 NOTICE.md），调用与封装为本组编写
"""LIME：论文用的是 LIME（arXiv:2310.08645 摘要、ESWA 版亦同），我们至少复现一组图。

用法：
    python -m analysis.lime_explain --run results/runs/vgg16__D1__adam__s42 --n 6
"""

from __future__ import annotations

from pathlib import Path


def build_explainer(predict_fn, train_images, num_samples: int = 1000):
    """封装 lime.lime_image.LimeImageExplainer。"""
    raise NotImplementedError


def explain_one(explainer, image_path: Path, num_features: int = 10):
    raise NotImplementedError


def main() -> None:
    """输出 results/figures/lime_<model>_<i>.png（原图 + 正/负超像素掩码）。"""
    raise NotImplementedError


if __name__ == "__main__":
    main()
