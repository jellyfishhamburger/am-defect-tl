# Author: <姓名> | Task: 评估与指标 | PR: #<编号>
# Status: 本组原创实现。使用 sklearn.metrics 计算指标（仅调用，见 NOTICE.md）
"""评估：10 个指标 + 3 个混淆矩阵 + ROC/PR 曲线。

指标口径见 docs/RESULTS.md §4，**positive = defect（缺陷）** —— 正类定义错会让全部指标翻转。
"""

from __future__ import annotations

METRIC_NAMES = [
    "accuracy", "precision", "recall", "f1", "mcc", "kappa",
    "auc", "map", "sensitivity", "specificity",
]


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """计算 10 个指标，全部返回 0–1 的 float。

    y_prob: 正类（defect）的预测概率；y_true: 0=nondefect, 1=defect
    """
    raise NotImplementedError


def confusion_matrix_2x2(y_true, y_pred) -> list[list[int]]:
    """返回 [[TN, FP], [FN, TP]]，与 docs/RESULTS.md §5 一致。"""
    raise NotImplementedError


def plot_confusion(cm: list[list[int]], title: str, out_path):
    raise NotImplementedError


def plot_curves(train_log, out_path):
    """训练/验证的 accuracy 与 loss 曲线。"""
    raise NotImplementedError


def plot_roc_pr(y_true, y_prob, out_path):
    raise NotImplementedError
