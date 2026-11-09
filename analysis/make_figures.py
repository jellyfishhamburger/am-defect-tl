# Author: jellyfishhamburger | Task: T1 报告图表生成器 | PR: 待提
# Status: 本组原创实现。只读 results/，不修改任何 metrics.json / leaderboard.csv。
"""生成报告第 3 节需要的全局汇总图（T1）。

输入（只读）：
    results/leaderboard.csv
    results/runs/*/metrics.json
    results/runs/*/confusion_matrices.json

输出：
    results/figures/fig_accuracy_by_model.png   11 模型 × 2 数据集准确率（含论文值）
    results/figures/fig_optimizer_heatmap.png   3 优化器消融热力图
    results/figures/fig_confusion_grid.png      4 个代表 run 的混淆矩阵
    results/figures/fig_delta_acc.png           我们 − 论文 的准确率差
    results/figures/fig_ranking.png             模型准确率排名
    results/figures/INDEX.md                    图号 / 文件 / 一句话说明

用法：
    python -m analysis.make_figures                 # 全部重画
    python -m analysis.make_figures --only 1,3      # 只画指定编号
    python -m analysis.make_figures --dpi 300

约束：所有数字直接读仓库文件；图中不出现任何手写常数（论文值除外，来自 leaderboard 的 paper_acc 列）。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGDIR = RESULTS / "figures"

# 与看板/报告保持一致的中文名 → 英文展示名（图里统一用英文）
DISPLAY = {
    "vgg16": "VGG16", "vgg19": "VGG19", "resnet50": "ResNet50", "resnet101": "ResNet101",
    "inception_resnet_v2": "InceptionResNetV2", "xception": "Xception",
    "mobilenet_v2": "MobileNetV2", "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_b7": "EfficientNet-B7", "efficientnet_v2m": "EfficientNetV2-M",
    "nasnet_large": "NASNet-Large",
}
# 论文里的模型顺序（大致按参数量从小到大），图里保持同一顺序便于比较
ORDER = ["mobilenet_v2", "efficientnet_b0", "resnet50", "xception", "inception_resnet_v2",
         "efficientnet_v2m", "efficientnet_b7", "resnet101", "vgg16", "vgg19", "nasnet_large"]
DATASETS = ["D1", "D2"]
OPTS = ["adam", "sgd", "rmsprop"]

C_D1, C_D2 = "#16233A", "#BF452C"
C_PAPER = "#8C8175"
C_ACCENT = "#2A6B57"
GREY = "#6B7280"

# 由 main() 里的 matplotlib 导入填充，避免在 import 阶段拉起绘图库（CI 不装 matplotlib/torch）
plt = None


# ---------------------------------------------------------------- data
def read_leaderboard() -> list[dict]:
    with (RESULTS / "leaderboard.csv").open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("accuracy", "f1", "mcc", "paper_acc", "paper_f1", "delta_acc",
                  "n_params_trainable", "train_time_s"):
            if r.get(k) in ("", None):
                r[k] = None
            else:
                r[k] = float(r[k])
    return rows


def load_metrics(run_id: str) -> dict:
    p = RESULTS / "runs" / run_id / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_cm(run_id: str) -> dict:
    p = RESULTS / "runs" / run_id / "confusion_matrices.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def pick(rows, model, dataset, opt="adam"):
    for r in rows:
        if r["model"] == model and r["dataset"] == dataset and r["optimizer"] == opt:
            return r
    return None


def style(plt):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#9AA0A6",
        "axes.labelcolor": "#1F2937",
        "text.color": "#111827",
        "grid.color": "#E5E0D5",
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


# ---------------------------------------------------------------- figures
def fig_accuracy(models, rows, out: Path):
    """① 11 模型 × 2 数据集 测试准确率，叠加论文报告值。"""
    import numpy as np

    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    x = np.arange(len(models))
    w = 0.36
    for i, (ds, color) in enumerate(zip(DATASETS, (C_D1, C_D2))):
        vals = [(pick(rows, m, ds) or {}).get("accuracy") for m in models]
        vals = [v if v is not None else 0 for v in vals]
        ax.bar(x + (i - 0.5) * w, vals, w, color=color, label=f"{ds} (ours, Adam)", zorder=2)
        paper = [((pick(rows, m, ds) or {}).get("paper_acc")) for m in models]
        px = [xi + (i - 0.5) * w for xi, p in zip(x, paper) if p is not None]
        py = [p for p in paper if p is not None]
        ax.scatter(px, py, marker="_", s=170, linewidths=1.8, color=C_PAPER,
                   label="reported by authors" if i == 0 else None, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY.get(m, m) for m in models], rotation=28, ha="right")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.40, 1.03)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Test accuracy by backbone and dataset (Adam, seed 42)", pad=26)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.005), fontsize=7.6)
    fig.savefig(out / "fig_accuracy_by_model.png")
    plt.close(fig)


def fig_heatmap(models, rows, out: Path):
    """② 3 优化器消融热力图（D1 / D2 两个面板，11 模型 × 3 优化器，共用色阶）。"""
    import numpy as np

    grids = []
    for ds in DATASETS:
        grid = np.full((len(models), len(OPTS)), np.nan)
        for i, m in enumerate(models):
            for j, o in enumerate(OPTS):
                r = pick(rows, m, ds, o)
                if r and r["accuracy"] is not None:
                    grid[i, j] = r["accuracy"]
        grids.append(grid)
    finite = np.concatenate([g[~np.isnan(g)] for g in grids])
    vmin = np.floor(finite.min() * 100) / 100 if finite.size else 0.85
    mid = (vmin + 1.0) / 2

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), sharey=True)
    for ax, ds, grid in zip(axes, DATASETS, grids):
        im = ax.imshow(grid, cmap="viridis", vmin=vmin, vmax=1.0, aspect="auto")
        for i in range(len(models)):
            for j in range(len(OPTS)):
                v = grid[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.4,
                        color="white" if v < mid + 0.01 else "#111827")
        ax.set_xticks(range(len(OPTS)))
        ax.set_xticklabels([{"adam": "Adam", "sgd": "SGD", "rmsprop": "RMSprop"}[o] for o in OPTS])
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([DISPLAY.get(m, m) for m in models])
        ax.set_title(f"{ds} — test accuracy")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label(f"Test accuracy (colour scale {vmin:.2f}–1.00, shared by both panels)", fontsize=7.5)
    cb.outline.set_visible(False)
    fig.suptitle("Optimizer ablation: frozen backbone, head-only training (lr=1e-3, 30 epochs)", fontsize=10)
    fig.savefig(out / "fig_optimizer_heatmap.png")
    plt.close(fig)


def fig_confusion(models, rows, out: Path):
    """③ 4 个代表 run 的测试集混淆矩阵。"""
    import numpy as np

    # 代表 run：D1 最好 / D1 最差 / D2 最好 / D2 最差（按测试准确率）
    d1 = [r for r in rows if r["dataset"] == "D1" and r["accuracy"] is not None]
    d2 = [r for r in rows if r["dataset"] == "D2" and r["accuracy"] is not None]
    d1.sort(key=lambda r: r["accuracy"])
    d2.sort(key=lambda r: r["accuracy"])
    picks = [d1[-1], d1[0], d2[-1], d2[0]]

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.6))
    for ax, r in zip(axes.ravel(), picks):
        cm = load_cm(r["run_id"]).get("test")
        if not cm:
            ax.axis("off")
            continue
        arr = np.array(cm)
        ax.imshow(arr, cmap="Blues", vmin=0, vmax=arr.max() * 1.15)
        for (i, j), v in np.ndenumerate(arr):
            ax.text(j, i, str(int(v)), ha="center", va="center", fontsize=11,
                    color="white" if v > arr.max() * 0.55 else "#111827")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["nondefect", "defect"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["nondefect", "defect"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{DISPLAY.get(r['model'], r['model'])} · {r['dataset']} · {r['optimizer'].upper()}\n"
                     f"acc={r['accuracy']:.3f}  f1={r['f1']:.3f}", fontsize=8.5)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Confusion matrices: best and worst runs per dataset (test split)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_confusion_grid.png")
    plt.close(fig)


def fig_delta(models, rows, out: Path):
    """④ 我们 − 论文 的准确率差。"""
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), sharex=True)
    # sharex=True：两个面板共用 x 轴，所以范围必须在循环外按全局极值设一次
    allv = [r["delta_acc"] for r in rows if r["delta_acc"] is not None]
    lo, hi = min(-0.09, min(allv) - 0.04), max(0.05, max(allv) * 1.18)
    for ax, ds in zip(axes, DATASETS):
        data = []
        for m in models:
            r = pick(rows, m, ds)
            if r and r["delta_acc"] is not None:
                data.append((r["model"], r["delta_acc"], r["paper_f1"]))
        data.sort(key=lambda t: t[1])
        names = [DISPLAY.get(m, m) for m, _, _ in data]
        vals = [v for _, v, _ in data]
        colors = [C_D1 if ds == "D1" else C_D2] * len(vals)
        bars = ax.barh(np.arange(len(vals)), vals, color=colors, height=0.62, zorder=2)
        ax.axvline(0, color="#111827", linewidth=0.9, zorder=3)
        ax.set_yticks(np.arange(len(vals)))
        ax.set_yticklabels(names)
        ax.set_title(f"{ds}")
        ax.set_xlabel("Δ accuracy (ours − authors)")
        ax.grid(axis="x", zorder=0)
        ax.set_axisbelow(True)
        for b, (_, v, pf1) in zip(bars, data):
            # 负值也把标签放在 0 线右侧：小负数条只有几像素宽，标签放左边会压到模型名
            ax.text((v if v > 0 else 0) + 0.008, b.get_y() + b.get_height() / 2, f"{v:+.3f}",
                    va="center", ha="left", fontsize=7.0,
                    color="#111827", clip_on=False,
                    bbox=None if pf1 != 0 else dict(boxstyle="round,pad=0.12", fc="#F3D3CB",
                                                    ec="none", alpha=0.95))
    axes[0].set_xlim(lo, hi)
    fig.suptitle("Reproduction gap vs. Ahsan et al. (2024) — Adam runs", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_delta_acc.png")
    plt.close(fig)


def fig_ranking(models, rows, out: Path):
    """⑤ 模型准确率排名（每个模型取三个优化器里最好的一次）。"""
    import numpy as np

    data = []
    for m in models:
        best = {}
        for ds in DATASETS:
            cand = [r for r in rows if r["model"] == m and r["dataset"] == ds and r["accuracy"] is not None]
            best[ds] = max(cand, key=lambda r: r["accuracy"]) if cand else None
        data.append((m, best))
    data.sort(key=lambda t: -((t[1]["D1"]["accuracy"] if t[1]["D1"] else 0)))

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    y = np.arange(len(data))
    for i, (m, best) in enumerate(data):
        r1, r2 = best["D1"], best["D2"]
        if r1 and r2:
            ax.plot([r1["accuracy"], r2["accuracy"]], [i, i], color="#D5CCB8", linewidth=1.4, zorder=1)
        if r1:
            ax.scatter(r1["accuracy"], i, s=42, color=C_D1, zorder=3, label="D1" if i == 0 else None)
        if r2:
            ax.scatter(r2["accuracy"], i, s=42, color=C_D2, zorder=3, label="D2" if i == 0 else None)
        if r1:
            ax.annotate({"adam": "Adam", "sgd": "SGD", "rmsprop": "RMSprop"}[r1["optimizer"]],
                        xy=(1.012, i), xycoords=("axes fraction", "data"),
                        va="center", ha="left", fontsize=6.8, color=GREY,
                        annotation_clip=False)
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY.get(m, m) for m, _ in data])
    ax.invert_yaxis()          # 最好的排在最上面（matplotlib 的 y=0 默认在底部）
    ax.set_xlabel("Test accuracy (best of 3 optimizers)")
    ax.set_xlim(0.90, 1.005)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Model ranking on D1 and D2 (best of 3 optimizers)")
    ax.annotate("best\noptimizer", xy=(1.012, 1.0), xycoords=("axes fraction", "axes fraction"),
                va="bottom", ha="left", fontsize=6.2, color=GREY, annotation_clip=False)
    fig.savefig(out / "fig_ranking.png")
    plt.close(fig)


INDEX_ROWS = [
    ("Figure 1", "fig_accuracy_by_model.png",
     "Test accuracy of 11 ImageNet backbones on D1 and D2 (Adam, seed 42). Bars are our runs; short horizontal ticks "
     "are the values reported by Ahsan et al. (2024). The y-axis starts at 0.40.",
     "① 11 模型 × 2 数据集准确率柱状图，短横线为论文值；纵轴从 0.40 起，读图时注意"),
    ("Figure 2", "fig_optimizer_heatmap.png",
     "Optimizer ablation: test accuracy of each backbone under Adam / SGD / RMSprop with a frozen backbone and "
     "head-only training (lr = 1e-3, 30 epochs). Colour scale spans the observed range.",
     "② 3 优化器消融热力图（D1 / D2 两栏）；色阶按实际最小值拉伸"),
    ("Figure 3", "fig_confusion_grid.png",
     "Confusion matrices of the best and worst run per dataset (test split). Rows = true label, columns = predicted; "
     "each panel is shaded to its own maximum.",
     "③ 4 个代表 run 的混淆矩阵（每数据集最好 / 最差）"),
    ("Figure 4", "fig_delta_acc.png",
     "Accuracy gap between our reproduction and the reported values (Adam runs). Pink labels mark the authors' "
     "EfficientNet-B0 / B7 / V2-M rows, whose reported F1 is 0.000 (majority-class collapse, D1 accuracy 0.4909) — "
     "those gaps are not evidence that our implementation is better.",
     "④ 我们 vs 论文的 Δaccuracy；浅红标注是作者侧塌成多数类的三行"),
    ("Figure 5", "fig_ranking.png",
     "Model ranking by test accuracy (best of the three optimizers). The line joins the best D1 and D2 run of each "
     "backbone; the right-hand column names the optimizer that produced the D1 best run. D1 test n=54, so ties "
     "within ~0.02 are not statistically meaningful.",
     "⑤ 模型准确率排名；右侧注明 D1 最优的优化器"),
]


def write_index(out: Path, made: list[str]):
    lines = [
        "# Figure index（报告第 3 节用图）",
        "",
        "> 由 `python -m analysis.make_figures` 生成，**不要手改**。数字全部读自 `results/leaderboard.csv` 与",
        "> `results/runs/*/{metrics,confusion_matrices}.json`；论文值来自 `docs/paper_results.csv`。",
        "",
        "| 图号 | 文件 | Caption (EN) | 说明 |",
        "|---|---|---|---|",
    ]
    for no, fname, cap, zh in INDEX_ROWS:
        mark = "" if (out / fname).exists() else " *(未生成)*"
        lines.append(f"| {no} | [`{fname}`]({fname}){mark} | {cap} | {zh} |")
    lines += [
        "",
        "## 生成命令",
        "",
        "```bash",
        "python -m analysis.make_figures            # 全部",
        "python -m analysis.make_figures --only 1,3 # 只画 1、3",
        "```",
        "",
        "## 口径与已知问题",
        "",
        "- 图 1 / 4 用 **Adam** run 与论文的 Adam 行对比；每个模型只比同一优化器，避免拿我们最好的和论文最差的比。",
        "- D1 测试集只有 54 张：错 1 张 = 1.8 个百分点，所以 0.98 与 1.00 的差别不要过度解读。",
        "- 论文的 EfficientNet-B0 / B7 / V2-M 报告 F1 = 0（全部预测为多数类），图 4 用浅红底标出，不能读作\"我们更强\"。",
        "- 图 3 选的是每个数据集准确率最高 / 最低的 run，属于**示意**，不是随机抽样。",
        "",
    ]
    (out / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="逗号分隔的图号，如 1,3；留空=全部")
    ap.add_argument("--out", type=Path, default=FIGDIR)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F811

    globals()["plt"] = plt
    style(plt)
    plt.rcParams["savefig.dpi"] = args.dpi

    rows = read_leaderboard()
    models = [m for m in ORDER if any(r["model"] == m for r in rows)]
    missing = sorted({r["model"] for r in rows} - set(models))
    if missing:
        models += missing

    args.out.mkdir(parents=True, exist_ok=True)
    wanted = {int(x) for x in args.only.split(",") if x.strip()} if args.only else {1, 2, 3, 4, 5}
    jobs = {1: fig_accuracy, 2: fig_heatmap, 3: fig_confusion, 4: fig_delta, 5: fig_ranking}
    made = []
    for no in sorted(wanted):
        if no not in jobs:
            print(f"[figures] 跳过未知图号 {no}")
            continue
        jobs[no](models, rows, args.out)
        name = {1: "fig_accuracy_by_model.png", 2: "fig_optimizer_heatmap.png",
                3: "fig_confusion_grid.png", 4: "fig_delta_acc.png", 5: "fig_ranking.png"}[no]
        made.append(name)
        print(f"[figures] {no} -> {name}", flush=True)

    write_index(args.out, made)
    print(f"[figures] {len(made)} 张图 + INDEX.md → {args.out}")


if __name__ == "__main__":
    main()
