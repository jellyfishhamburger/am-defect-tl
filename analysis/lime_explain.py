# Author: jellyfishhamburger | Task: T4 LIME 可解释性（CPU） | PR: 待提
# Status: 本组原创实现；LIME 库为第三方（见 NOTICE.md），本文件只做适配与出图。
"""LIME 解释图：同一张图，模型靠哪些超像素做判断（T4）。

输入：
    results/runs/*/model.pt     分类头权重（Kenneth 已上传 5 个 D1 + 5 个 D2，均为 Adam）
    samples/D1/{defect,nondefect}/*.jpg   20 张 224×224 调试样图（仓库已带）

输出：
    results/figures/lime_<run_id>.png     每个模型一张：原图 + LIME 正/负贡献区域

用法：
    python -m analysis.lime_explain                                  # 默认 3 个模型 × 4 张图
    python -m analysis.lime_explain --models vgg16,resnet50 --dataset D1 --num-samples 120
    python -m analysis.lime_explain --dataset D2 --n-per-class 3

验收要点（T4）：≥3 个模型各出 1 张图；CPU 上单图 ≤60 秒。
    实测 VGG16 约 8–12 秒/图（num_samples=100），用 --num-samples 可调。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SAMPLES = ROOT / "samples"
FIGDIR = RESULTS / "figures"

DISPLAY = {
    "vgg16": "VGG16", "vgg19": "VGG19", "resnet50": "ResNet50", "resnet101": "ResNet101",
    "inception_resnet_v2": "InceptionResNetV2", "xception": "Xception",
    "mobilenet_v2": "MobileNetV2", "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_b7": "EfficientNet-B7", "efficientnet_v2m": "EfficientNetV2-M",
    "nasnet_large": "NASNet-Large",
}


def list_samples(samples_root: Path, dataset: str, n_per_class: int) -> list[tuple[Path, str]]:
    """从 samples/<dataset>/{defect,nondefect} 各取 n 张（按文件名排序，可复现）。"""
    out = []
    for label in ("defect", "nondefect"):
        d = samples_root / dataset / label
        if not d.exists():
            continue
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        out += [(p, label) for p in files[:n_per_class]]
    return out


def find_run(model: str, dataset: str, opt: str = "adam") -> Path | None:
    for p in sorted((RESULTS / "runs").glob(f"{model}__{dataset}__{opt}__*")):
        if (p / "model.pt").exists():
            return p
    return None


def make_predict_fn(model, size: int):
    """把模型包成 LIME 需要的 f(numpy N×H×W×3) → (N, 2) 概率。"""
    import torch
    from src.dataset import IMAGENET_MEAN, IMAGENET_STD

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[None, :, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[None, :, None, None]

    def predict(images: np.ndarray) -> np.ndarray:
        arr = np.asarray(images, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(0, 3, 1, 2)
        if t.shape[-1] != size:
            t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear",
                                                align_corners=False)
        t = (t - mean) / std
        with torch.no_grad():
            logits = model(t)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    return predict


def overlay_mask(img: np.ndarray, mask: np.ndarray, color=(220, 60, 40),
                 alpha_pos: float = 0.55) -> np.ndarray:
    """正贡献区域涂红，负贡献区域涂蓝；其余保持原图。"""
    out = img.astype(np.float32).copy()
    pos = mask > 0
    neg = mask < 0
    for sel, col in ((pos, np.array([220, 60, 40], dtype=np.float32)),
                     (neg, np.array([40, 90, 200], dtype=np.float32))):
        if sel.any():
            out[sel] = out[sel] * (1 - alpha_pos) + col * alpha_pos
    return np.clip(out, 0, 255).astype(np.uint8)


def explain_one(run_dir: Path, samples: list[tuple[Path, str]], num_samples: int,
                num_features: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from lime.lime_image import LimeImageExplainer

    from analysis.checkpoint import load_model

    model, ckpt = load_model(run_dir, device="cpu")
    size = int(ckpt.get("input_size", 224))
    model.eval()
    predict = make_predict_fn(model, size)

    fig, axes = plt.subplots(2, len(samples), figsize=(2.9 * len(samples), 6.2),
                             gridspec_kw={"hspace": 0.06, "wspace": 0.04})
    axes = np.atleast_2d(axes)

    def stamp(ax, text, y=0.03, size=7.6):
        """把说明直接压在图片底部，避免标题和上一行图片重叠。"""
        ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=size, color="white", linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none", alpha=0.55))

    for col, (path, true_label) in enumerate(samples):
        img = np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR))
        explainer = LimeImageExplainer()
        explanation = explainer.explain_instance(
            img, predict, top_labels=1, hide_color=0,
            num_samples=num_samples, random_seed=42, batch_size=16,
        )
        label = explanation.top_labels[0]
        _, mask = explanation.get_image_and_mask(
            label, positive_only=False, num_features=num_features,
            hide_rest=False, min_weight=0.0,
        )
        prob = predict(img[None, ...])[0]

        ax = axes[0, col]
        ax.imshow(img)
        ax.axis("off")
        stamp(ax, f"true = {true_label}")

        ax = axes[1, col]
        ax.imshow(overlay_mask(img, mask))
        ax.axis("off")
        stamp(ax, f"p(defect) = {prob[1]:.2f}\nLIME explains '{'defect' if label == 1 else 'nondefect'}'")

    fig.suptitle(
        f"LIME — {DISPLAY.get(ckpt.get('model', ''), ckpt.get('model', ''))} · "
        f"{run_dir.name.split('__')[1]} · {run_dir.name.split('__')[2]}  "
        f"(red = supports the decision, blue = contradicts; {num_samples} samples)",
        fontsize=9.5)
    fig.tight_layout()
    out = FIGDIR / f"lime_{run_dir.name}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="vgg16,resnet50,mobilenet_v2",
                    help="逗号分隔；默认 3 个（T4 验收要求 ≥3）")
    ap.add_argument("--dataset", default="D1", choices=["D1", "D2"])
    ap.add_argument("--samples-root", type=Path, default=SAMPLES)
    ap.add_argument("--n-per-class", type=int, default=2, help="每类取几张（默认 2，共 4 张）")
    ap.add_argument("--num-samples", type=int, default=100, help="LIME 采样数；越大越稳也越慢")
    ap.add_argument("--num-features", type=int, default=6, help="展示的超像素数量")
    args = ap.parse_args()

    samples = list_samples(args.samples_root, args.dataset, args.n_per_class)
    if not samples:
        raise SystemExit(f"{args.samples_root}/{args.dataset} 下没有样图；先确认 samples/ 是否存在")

    FIGDIR.mkdir(parents=True, exist_ok=True)
    made = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        run_dir = find_run(name, args.dataset)
        if run_dir is None:
            print(f"[lime] 跳过 {name}：没有 {args.dataset}/adam 的 model.pt")
            continue
        print(f"[lime] {DISPLAY.get(name, name)} · {run_dir.name} …", flush=True)
        out = explain_one(run_dir, samples, args.num_samples, args.num_features)
        made.append(out.name)
        print(f"[lime]   -> {out.name}", flush=True)

    if not made:
        raise SystemExit("一张图都没生成：检查模型名与 dataset")


if __name__ == "__main__":
    main()
