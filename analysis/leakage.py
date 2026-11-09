# Author: Kenneth | Task: 数据泄漏对照实验（自主贡献） | PR: 待提
# Status: 本组原创实现
"""数据泄漏对照实验：**随机划分 vs 按原图分组划分**。

背景（见 docs/DATA.md §5）：
    作者把 D1（271 张真实照片）做数据增强得到 D2（3,017 张），然后随机划分 70/10/20。
    同一张原图的多个增强副本因此可能同时出现在 train 与 test → 测试集含"训练集的近亲"
    → 准确率被系统性高估。作者的增强文件名（aug_<k>_<随机数>）已丢失溯源信息，无法还原。

本实验的做法：
    1. 从 D1 重新生成一份增强数据集 **D2′**，文件名里保留 `source_id`（原图名）→ 溯源可控
    2. 对同一份 D2′ 生成两套划分：
         - `D2p_random.csv`：随机分层 70/10/20（复刻作者的泄漏情形）
         - `D2p_group.csv` ：按 `source_id` 分组划分（同一原图的增强副本不跨集合）
    3. 同一模型、同一超参，分别在上面两套划分上训练 → 对比 test 指标
    4. Δ(acc) = random − group，即"泄漏带来的虚高幅度"

用法：
    python -m analysis.leakage --data-root <数据根> --out results \
        --models vgg16,resnet50,mobilenet_v2 --per-image 11 --models-per-run 1
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

DATASET = "D2p"
LABELS = ["defect", "nondefect"]
SIZE = 224
RATIO = (0.7, 0.1, 0.2)


# ---------------------------------------------------------------- 增强 & 数据集
def _augment(img, rng: random.Random):
    """单张图的随机增强：旋转 / 翻转 / 亮度 / 对比度 / 缩放 / 噪声。"""
    import numpy as np
    from PIL import Image, ImageEnhance

    angle = rng.uniform(-15, 15)
    im = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.8, 1.2))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.8, 1.2))

    scale = rng.uniform(0.9, 1.1)
    w, h = im.size
    im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    left = max(0, (im.width - SIZE) // 2)
    top = max(0, (im.height - SIZE) // 2)
    im = im.crop((left, top, left + SIZE, top + SIZE))
    if im.size != (SIZE, SIZE):
        im = im.resize((SIZE, SIZE), Image.BILINEAR)

    arr = np.asarray(im, dtype=np.float32)
    arr = np.clip(arr + np.float32(rng.gauss(0, 3.0)), 0, 255).astype("uint8")
    return Image.fromarray(arr)


def build_d2prime(data_root: Path, work_dir: Path, per_image: int = 11, seed: int = 42) -> Path:
    """从 D1 生成 D2′（保留 source_id），返回「伪数据根」（其下有 data/D2p/...）。"""
    from PIL import Image

    d1_dir = Path(data_root) / "data" / "D1"
    out_root = Path(work_dir)
    rows: list[dict] = []
    written = 0

    for label in LABELS:
        src_dir = d1_dir / label
        if not src_dir.is_dir():
            raise FileNotFoundError(f"找不到 D1 目录：{src_dir}")
        dst_dir = out_root / "data" / DATASET / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in src_dir.iterdir() if p.is_file())
        for f in files:
            base = Image.open(f).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
            for k in range(per_image):
                rng = random.Random(f"{seed}-{f.stem}-{k}")
                aug = _augment(base, rng)
                name = f"{f.stem}__aug{k:02d}.jpg"
                aug.save(dst_dir / name, "JPEG", quality=88)
                rows.append({
                    "path": (Path("data") / DATASET / label / name).as_posix(),
                    "label": label,
                    "source_id": f.stem,
                })
                written += 1
        print(f"[leakage] {label}: {len(files)} 张原图 → {len(files) * per_image} 张增强图", flush=True)

    print(f"[leakage] D2′ 生成完成：{written} 张 → {out_root / 'data' / DATASET}", flush=True)
    return out_root


# ---------------------------------------------------------------- 两套划分
def _write_split(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "split", "source_id"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def split_random(rows: list[dict], seed: int = 42) -> list[dict]:
    """随机分层 70/10/20 —— 复刻作者的泄漏情形。"""
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    out = []
    for offset, label in enumerate(LABELS):
        items = by_label[label][:]
        random.Random(seed + offset).shuffle(items)
        n = len(items)
        n_train = int(round(n * RATIO[0]))
        n_val = int(round(n * RATIO[1]))
        for i, r in enumerate(items):
            r = dict(r)
            r["split"] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
            out.append(r)
    return out


def split_group(rows: list[dict], seed: int = 42) -> list[dict]:
    """按 source_id 分组划分：同一原图的所有增强副本只进同一个集合。"""
    by_label = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_label[r["label"]][r["source_id"]].append(r)

    out = []
    for offset, label in enumerate(LABELS):
        groups = sorted(by_label[label].keys())
        random.Random(seed + offset).shuffle(groups)
        n = len(groups)
        n_train = int(round(n * RATIO[0]))
        n_val = int(round(n * RATIO[1]))
        for i, sid in enumerate(groups):
            split = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
            for r in by_label[label][sid]:
                r = dict(r)
                r["split"] = split
                out.append(r)
    return out


def _summary(rows: list[dict]) -> str:
    c = defaultdict(int)
    for r in rows:
        c[r["split"]] += 1
    return f"train={c['train']} val={c['val']} test={c['test']}"


def subsample_train(rows: list[dict], fraction: float, seed: int = 42) -> list[dict]:
    """只保留一部分训练样本，val/test 不动。

    目的：原始实验里两套划分都顶到 ~100%（天花板效应），看不出泄漏差异。
    砍掉训练数据后准确率下降，泄漏才有空间显现。
    对 group 划分，砍掉部分副本不会破坏"同源不跨集合"的性质。
    """
    if fraction >= 1.0:
        return rows
    keep: set[str] = set()
    for offset, label in enumerate(LABELS):
        items = [r for r in rows if r["split"] == "train" and r["label"] == label]
        n_keep = max(1, int(round(len(items) * fraction)))
        idx = list(range(len(items)))
        random.Random(seed + offset).shuffle(idx)
        keep.update(items[i]["path"] for i in idx[:n_keep])
    out = []
    for r in rows:
        if r["split"] != "train" or r["path"] in keep:
            out.append(r)
    return out


# ---------------------------------------------------------------- 实验
def run_ablation(data_root: Path, splits_dir: Path, out_root: Path,
                 models: list[str], smoke: bool = False) -> list[dict]:
    """同一模型、同一超参，分别在 random / group 两套划分上训练，对比 test 指标。"""
    from src.batch import DATASET_HP
    from src.train import load_config, run

    template = Path("configs/_template.yaml")
    base = load_config(template)
    results = []

    for split_name in ("random", "group"):
        for model in models:
            cfg = dict(base)
            cfg.update({"model": model, "dataset": f"{DATASET}_{split_name}",
                        "optimizer": "adam", **DATASET_HP["D2"]})
            print(f"\n[leakage] ===== {model} / {split_name} =====", flush=True)
            run_dir = run(cfg, data_root, splits_dir, out_root, smoke=smoke,
                          force=True, cache_root=None)
            import json
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            test = metrics["metrics"]["test"]
            results.append({
                "model": model,
                "split_mode": split_name,
                "n_train": metrics["n_train"], "n_val": metrics["n_val"], "n_test": metrics["n_test"],
                "accuracy": test["accuracy"], "precision": test["precision"],
                "recall": test["recall"], "f1": test["f1"], "auc": test["auc"],
                "run_id": metrics["run_id"],
            })
    return results


def write_results(results: list[dict], out_root: Path, tag: str, train_fraction: float) -> Path:
    """写 results/leakage_<tag>.csv（含随机划分与分组划分的差值）。"""
    by_model = defaultdict(dict)
    for r in results:
        by_model[r["model"]][r["split_mode"]] = r

    fields = ["setting", "train_fraction", "model",
              "accuracy_random", "accuracy_group", "delta_accuracy",
              "f1_random", "f1_group", "delta_f1", "auc_random", "auc_group",
              "n_train_random", "n_train_group", "n_test"]
    rows = []
    for model, d in by_model.items():
        a, g = d.get("random"), d.get("group")
        if not a or not g:
            continue
        rows.append({
            "setting": tag,
            "train_fraction": train_fraction,
            "model": model,
            "accuracy_random": round(a["accuracy"], 4),
            "accuracy_group": round(g["accuracy"], 4),
            "delta_accuracy": round(a["accuracy"] - g["accuracy"], 4),
            "f1_random": round(a["f1"], 4),
            "f1_group": round(g["f1"], 4),
            "delta_f1": round(a["f1"] - g["f1"], 4),
            "auc_random": round(a["auc"], 4),
            "auc_group": round(g["auc"], 4),
            "n_train_random": a["n_train"],
            "n_train_group": g["n_train"],
            "n_test": a["n_test"],
        })

    out = Path(out_root) / f"leakage_{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"[leakage] 写出 {out}", flush=True)
    return out


def plot_results(csv_path: Path, out_dir: Path, tag: str) -> Path | None:
    """画对照柱状图：每个模型两组准确率 + 差值。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import csv as _csv

    with Path(csv_path).open(encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    if not rows:
        return None

    models = [r["model"] for r in rows]
    acc_r = [float(r["accuracy_random"]) for r in rows]
    acc_g = [float(r["accuracy_group"]) for r in rows]
    deltas = [float(r["delta_accuracy"]) for r in rows]

    import numpy as np

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(models)), 3.6))
    w = 0.36
    ax.bar(x - w / 2, acc_r, w, label="random split (leak-prone)")
    ax.bar(x + w / 2, acc_g, w, label="group split (source-disjoint)")
    for i, d in enumerate(deltas):
        ax.annotate(f"Δ{d:+.3f}", (x[i], max(acc_r[i], acc_g[i]) + 0.02),
                    ha="center", fontsize=8)
    ax.set_xticks(x, models, rotation=15)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("test accuracy")
    ax.set_title(f"Data leakage: random vs source-grouped split ({tag})")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = Path(out_dir) / f"fig_leakage_{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[leakage] 写出 {out}", flush=True)
    return out


def combine_and_plot(out_root: Path) -> None:
    """把 results/leakage_<tag>.csv 合并成 results/leakage.csv，并画 2 面板对比图。

    纯 csv + matplotlib，不需要 torch / 数据，可以在本地直接跑。
    """
    import csv as _csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    parts = [p for p in sorted(Path(out_root).glob("leakage_*.csv")) if p.name != "leakage.csv"]
    if not parts:
        raise FileNotFoundError(f"{out_root} 下没有 leakage_*.csv")
    rows = []
    for p in parts:
        with p.open(encoding="utf-8") as fh:
            rows.extend(list(_csv.DictReader(fh)))
    rows.sort(key=lambda r: (r["setting"], r["model"]))

    out_csv = Path(out_root) / "leakage.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"[leakage] 合并写出 {out_csv}（{len(rows)} 行）", flush=True)

    settings = sorted({r["setting"] for r in rows})
    fig, axes = plt.subplots(1, len(settings), figsize=(5.8 * len(settings), 3.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, settings):
        sub = [r for r in rows if r["setting"] == s]
        models = [r["model"] for r in sub]
        x = np.arange(len(models))
        width = 0.36
        acc_r = [float(r["accuracy_random"]) for r in sub]
        acc_g = [float(r["accuracy_group"]) for r in sub]
        ax.bar(x - width / 2, acc_r, width, label="random split (leak-prone)")
        ax.bar(x + width / 2, acc_g, width, label="source-grouped split")
        for i, r in enumerate(sub):
            ax.annotate(f"{float(r['delta_accuracy']):+.3f}",
                        (x[i], min(1.06, max(acc_r[i], acc_g[i]) + 0.015)),
                        ha="center", fontsize=8)
        ax.set_xticks(x, models, rotation=15)
        ax.set_ylim(0, 1.14)
        ax.set_title(f"setting = {s}  (train = {sub[0]['n_train_random']} imgs)")
        ax.set_ylabel("test accuracy")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Data leakage test on D2': random vs source-grouped split", fontsize=11)
    fig.tight_layout()
    out_fig = Path(out_root) / "figures" / "fig_leakage_combined.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)
    print(f"[leakage] 写出 {out_fig}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None,
                    help="数据根目录（--combine 模式下可省略）")
    ap.add_argument("--combine", action="store_true",
                    help="只合并已有 leakage_*.csv 并画对比图（纯本地，不需要 GPU/数据）")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--work-dir", type=Path, default=None, help="D2′ 与划分的存放目录")
    ap.add_argument("--models", default="vgg16,resnet50,mobilenet_v2")
    ap.add_argument("--per-image", type=int, default=11)
    ap.add_argument("--train-fraction", type=float, default=1.0,
                    help="只保留这部分训练样本（<1 用于打破准确率天花板，让泄漏效应显现）")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out)
    if args.combine:
        combine_and_plot(out_root)
        return
    if args.data_root is None:
        raise SystemExit("非 --combine 模式必须提供 --data-root")
    work_dir = Path(args.work_dir) if args.work_dir else out_root / "leakage_work"
    splits_dir = work_dir / "splits"
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    # 1) 生成 D2′
    d2p_root = build_d2prime(args.data_root, work_dir, args.per_image)

    # 2) 两套划分
    import csv as _csv
    rows = []
    for label in LABELS:
        d = d2p_root / "data" / DATASET / label
        for f in sorted(d.iterdir()):
            rows.append({"path": (Path("data") / DATASET / label / f.name).as_posix(),
                         "label": label, "source_id": f.name.split("__aug")[0]})

    rnd = split_random(rows)
    grp = split_group(rows)
    rnd = subsample_train(rnd, args.train_fraction)
    grp = subsample_train(grp, args.train_fraction)
    tag = "full" if args.train_fraction >= 1.0 else f"small{int(round(args.train_fraction * 100))}"
    _write_split(rnd, splits_dir / f"{DATASET}_random.csv")
    _write_split(grp, splits_dir / f"{DATASET}_group.csv")
    print(f"[leakage] 设定={tag}  train_fraction={args.train_fraction}", flush=True)
    print(f"[leakage] random 划分: {_summary(rnd)}", flush=True)
    print(f"[leakage] group  划分: {_summary(grp)}", flush=True)

    # 3) 训练对比
    results = run_ablation(d2p_root, splits_dir, out_root / f"leakage_runs_{tag}", models, smoke=args.smoke)

    # 4) 汇总 + 画图
    csv_path = write_results(results, out_root, tag, args.train_fraction)
    plot_results(csv_path, out_root / "figures", tag)
    print("[leakage] 完成", flush=True)


if __name__ == "__main__":
    main()
