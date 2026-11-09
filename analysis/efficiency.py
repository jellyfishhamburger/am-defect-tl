# Author: jellyfishhamburger | Task: T3 效率测量（CPU） | PR: 待提
# Status: 本组原创实现
"""效率测量：参数量 / FLOPs / 单张推理时延 → results/efficiency.csv + fig_efficiency.png（T3）。

为什么值得做：题目背景是"3D 打印在线质检"，只有准确率不够，还要证明轻量模型可部署。

用法：
    python -m analysis.efficiency                 # 全部 11 个骨干，CPU
    python -m analysis.efficiency --models vgg16,mobilenet_v2
    python -m analysis.efficiency --iters 30

说明：
    - 骨干第一次构建会从 timm 下载 ImageNet 权重（100–600 MB/模型），之后走缓存；
      本任务只测量**推理**成本，权重是否预训练对时延的影响可以忽略，但为了与训练时的模型一致，仍用 pretrained=True。
    - GPU 时延本机无法测（无 GPU），CSV 里写 N/A，交给 Kenneth 在 Kaggle 补。
    - **测量时必须让机器空载**：并行下载权重 / 跑 LIME 会把均值拖高 40–100%（实测 EfficientNetV2-M
      344.9 → 161.8 ms）。重测前先确认没有别的 python 进程在跑。
    - 不写任何 src/ 下的文件，不改 metrics.json。
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGDIR = RESULTS / "figures"
DISPLAY = {
    "vgg16": "VGG16", "vgg19": "VGG19", "resnet50": "ResNet50", "resnet101": "ResNet101",
    "inception_resnet_v2": "InceptionResNetV2", "xception": "Xception",
    "mobilenet_v2": "MobileNetV2", "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_b7": "EfficientNet-B7", "efficientnet_v2m": "EfficientNetV2-M",
    "nasnet_large": "NASNet-Large",
}
ORDER = ["mobilenet_v2", "efficientnet_b0", "resnet50", "xception", "inception_resnet_v2",
         "efficientnet_v2m", "efficientnet_b7", "resnet101", "vgg16", "vgg19", "nasnet_large"]

COLUMNS = ["model", "n_params", "n_params_trainable", "gflops",
           "cpu_ms_per_image", "gpu_ms_per_image"]


def measure_params(model) -> tuple[int, int]:
    """(总参数量, 可训练参数量)。冻结骨干后 trainable 只有分类头。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def measure_flops(model, input_size: int = 224) -> float:
    """GFLOPs（前向，batch=1）。优先用 torch 自带的 flop counter，避免额外依赖。"""
    import torch

    x = torch.zeros(1, 3, input_size, input_size)
    try:
        from torch.utils.flop_counter import FlopCounterMode

        counter = FlopCounterMode(display=False)
        with counter, torch.no_grad():
            model(x)
        total = counter.get_total_flops()
        return total / 1e9
    except Exception:
        return float("nan")


def measure_latency(model, input_size: int = 224, device: str = "cpu",
                    warmup: int = 5, iters: int = 20) -> dict:
    """返回 {mean_ms, p95_ms}：warmup 次后计时 iters 次。"""
    import torch

    model.eval()
    x = torch.zeros(1, 3, input_size, input_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": samples[min(len(samples) - 1, int(round(0.95 * (len(samples) - 1))))],
    }


def draw_figure(rows: list[dict], out: Path) -> None:
    """fig_efficiency.png：CPU 单张时延（横条）+ 参数量标注。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "grid.color": "#E5E0D5", "legend.frameon": False,
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })
    rows = sorted([r for r in rows if r.get("cpu_ms_per_image")], key=lambda r: r["cpu_ms_per_image"])
    names = [DISPLAY.get(r["model"], r["model"]) for r in rows]
    vals = [r["cpu_ms_per_image"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    bars = ax.barh(range(len(rows)), vals, color="#16233A", height=0.62, zorder=2)
    for b, r in zip(bars, rows):
        label = f"{r['cpu_ms_per_image']:.0f} ms · {r['n_params']/1e6:.1f} M params"
        if r.get("gflops") == r.get("gflops"):  # not NaN
            label += f" · {r['gflops']:.1f} GFLOPs"
        ax.annotate(label, xy=(b.get_width(), b.get_y() + b.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=6.8, color="#374151")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names)
    ax.set_xlabel("CPU inference time per image (ms, batch 1, 224×224)")
    ax.set_xlim(0, max(vals) * 1.55)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Inference cost of the 11 backbones (frozen, head only)")
    fig.text(0.01, -0.03,
             "Measured on the author's laptop CPU (no GPU). Absolute numbers are machine-specific; "
             "the relative ordering is what matters.",
             fontsize=6.6, color="#6B7280")
    fig.savefig(out / "fig_efficiency.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all", help="逗号分隔；all=11 个骨干")
    ap.add_argument("--iters", type=int, default=20, help="计时重复次数（默认 20）")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", type=Path, default=RESULTS / "efficiency.csv")
    ap.add_argument("--replot", action="store_true",
                    help="不重新测量，只用已有的 efficiency.csv 重画图")
    args = ap.parse_args()

    if args.replot:
        if not args.out.exists():
            raise SystemExit(f"{args.out} 不存在，先跑一次完整测量")
        with args.out.open(encoding="utf-8-sig", newline="") as fh:
            rows = []
            for r in csv.DictReader(fh):
                for k in ("n_params", "n_params_trainable", "gflops", "cpu_ms_per_image"):
                    r[k] = float(r[k]) if r.get(k) not in ("", None) else float("nan")
                rows.append(r)
        FIGDIR.mkdir(parents=True, exist_ok=True)
        draw_figure(rows, FIGDIR)
        print(f"[efficiency] 重新绘图 → {FIGDIR / 'fig_efficiency.png'}")
        return

    import torch

    from src.models import TransferModel

    wanted = None if args.models == "all" else {m.strip() for m in args.models.split(",")}
    models = [m for m in ORDER if wanted is None or m in wanted]
    if not models:
        raise SystemExit(f"没有匹配的模型：{args.models}")

    rows = []
    for i, name in enumerate(models, 1):
        t0 = time.time()
        try:
            model = TransferModel(name, pretrained=True, freeze="all_conv", input_size=224)
            model.eval()
        except Exception as exc:
            print(f"[efficiency] {i}/{len(models)} {name} 构建失败：{type(exc).__name__}: {exc}", flush=True)
            continue
        total, trainable = measure_params(model)
        gflops = measure_flops(model)
        lat = measure_latency(model, iters=args.iters, warmup=args.warmup)
        rows.append({
            "model": name,
            "n_params": total,
            "n_params_trainable": trainable,
            "gflops": round(gflops, 3) if gflops == gflops else float("nan"),
            "cpu_ms_per_image": round(lat["mean_ms"], 1),
            "gpu_ms_per_image": "N/A",
        })
        print(f"[efficiency] {i}/{len(models)} {DISPLAY.get(name, name)}: "
              f"{total/1e6:.1f}M params, {gflops:.2f} GFLOPs, "
              f"{lat['mean_ms']:.1f} ms (median {lat['median_ms']:.1f}) — {time.time()-t0:.0f}s",
              flush=True)

    if not rows:
        raise SystemExit("一个模型都没测成")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(r.get(k), float) and r.get(k) != r.get(k) else r.get(k, ""))
                        for k in COLUMNS})
    print(f"[efficiency] {len(rows)} 行 → {args.out}")

    FIGDIR.mkdir(parents=True, exist_ok=True)
    draw_figure([r for r in rows], FIGDIR)
    print(f"[efficiency] 图 → {FIGDIR / 'fig_efficiency.png'}")


if __name__ == "__main__":
    main()
