# Author: 组长 + agent | Task: 评估与指标 | PR: 待提
# Status: 本组原创实现。使用 sklearn.metrics 计算指标（仅调用，见 NOTICE.md）
"""评估：10 个指标 + 混淆矩阵。

口径见 docs/RESULTS.md §4。**positive = defect（缺陷）**，定义反了所有指标会翻转。
混淆矩阵约定：[[TN, FP], [FN, TP]]，行=真实，列=预测，顺序 nondefect → defect。
"""

from __future__ import annotations

METRIC_NAMES = [
    "accuracy", "precision", "recall", "f1", "mcc",
    "kappa", "auc", "map", "sensitivity", "specificity",
]


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """y_true: 0/1 的 array-like；y_prob: defect 的预测概率。返回 10 个 0–1 指标。"""
    import numpy as np
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 cohen_kappa_score, f1_score,
                                 matthews_corrcoef, precision_score,
                                 recall_score, roc_auc_score)

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0
    try:
        ap = average_precision_score(y_true, y_prob)
    except ValueError:
        ap = 0.0
    # 只出现一个类别时 sklearn 会返回 nan（冒烟/极小测试集常见），统一记 0.0 并在日志里可见
    if not np.isfinite(auc):
        auc = 0.0
    if not np.isfinite(ap):
        ap = 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true.tolist())) > 1 else 0.0,
        "kappa": float(cohen_kappa_score(y_true, y_pred)) if len(set(y_true.tolist())) > 1 else 0.0,
        "auc": float(auc),
        "map": float(ap),
        "sensitivity": float(recall),
        "specificity": float(specificity),
    }


def confusion_matrix_2x2(y_true, y_prob, threshold: float = 0.5) -> list[list[int]]:
    """返回 [[TN, FP], [FN, TP]]。"""
    import numpy as np

    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return [[tn, fp], [fn, tp]]


def plot_confusion(cm: list[list[int]], title: str, out_path) -> None:
    """画 2×2 混淆矩阵，用于报告图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["nondefect", "defect"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_curves(train_log: list[dict], out_path) -> None:
    """训练/验证的 loss 与 accuracy 曲线。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in train_log]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(epochs, [r["train_loss"] for r in train_log], label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in train_log], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, [r["train_acc"] for r in train_log], label="train")
    axes[1].plot(epochs, [r["val_acc"] for r in train_log], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
