# Author: 组长 + agent | Task: Grad-CAM 可解释性 + 预测示例图 | PR: 待提
# Status: 本组原创实现（CAM 算法自己实现，仅用 matplotlib 上色）
"""生成两类图（报告图 + demo 素材）：
    1. results/figures/gradcam_<run_id>.png  —— 原图 + Grad-CAM 热力图叠加，标注 true/pred/置信度
    2. results/figures/predict_<run_id>.png  —— 原图 + 预测标签 + 置信度 + 对错标记

前置条件：对应 run 目录下要有 model.pt（由 `src.train --save-weights` 生成，只存分类头）。

原理（Grad-CAM）：
    让梯度流到骨干输出的最后一层特征图 F（C×H×W），
    对目标类别的 logit 求导，权重 w_c = mean(∂y/∂F_c)，
    CAM = ReLU(Σ_c w_c · F_c)，归一化到 0–1 后上采样叠加到原图。

用法：
    python -m analysis.gradcam --data-root <数据根> --cache-root /kaggle/temp/cache \
        --models vgg16,mobilenet_v2 --dataset D1 --n-per-class 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, read_split
from src.models import TransferModel


def load_trained(run_dir: Path, device):
    """从 model.pt 复原模型：骨干用 ImageNet 权重重建，再载入分类头。"""
    ckpt = torch.load(Path(run_dir) / "model.pt", map_location="cpu", weights_only=False)
    model = TransferModel(
        ckpt["model"],
        num_classes=int(ckpt.get("num_classes", 2)),
        pretrained=True,
        freeze=ckpt.get("freeze", "all_conv"),
        dense_units=int(ckpt.get("dense_units", 64)),
        dropout=float(ckpt.get("dropout", 0.5)),
        input_size=int(ckpt.get("input_size", 224)),
    )
    model.head.load_state_dict(ckpt["head_state"])
    return model.to(device).eval(), ckpt


def resolve_path(cache_root, data_root, rel_path: str) -> Path:
    rel = str(rel_path).replace("\\", "/")
    if cache_root:
        parts = rel.split("/")
        if len(parts) >= 4 and parts[0] == "data":
            fname = Path("/".join(parts[3:])).stem + ".jpg"
            return Path(cache_root) / parts[1] / parts[2] / fname
    return Path(data_root) / rel


def load_image(path: Path, size: int):
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    t = (t - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(IMAGENET_STD)[:, None, None]
    return t.unsqueeze(0), im


def compute_cam(model, x: torch.Tensor, class_idx: int = 1):
    """返回 (CAM 0–1 的 H×W numpy, softmax 概率 numpy)。"""
    x = x.clone().requires_grad_(True)  # 让梯度能流到特征图
    feat = model.net.forward_features(x)
    feat.retain_grad()
    logits = model.head(feat)
    model.zero_grad(set_to_none=True)
    logits[0, class_idx].backward()

    grads = feat.grad[0]
    feats = feat[0]
    weights = grads.mean(dim=(1, 2))                     # 每个通道的重要性
    cam = torch.relu((weights[:, None, None] * feats).sum(0))
    cam = cam.detach().cpu().numpy()
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)
    probs = torch.softmax(logits.detach(), dim=1)[0].cpu().numpy()
    return cam, probs


def heat_overlay(im: Image.Image, cam: np.ndarray, alpha: float = 0.45):
    import matplotlib.cm as cm

    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(im.size, Image.BILINEAR)
    heat = cm.jet(np.asarray(cam_img, dtype=np.float32) / 255.0)[..., :3]
    base = np.asarray(im, dtype=np.float32) / 255.0
    return np.clip((1 - alpha) * base + alpha * heat, 0, 1)


def pick_samples(rows: list[dict], n_per_class: int) -> list[dict]:
    out = []
    for label in ("defect", "nondefect"):
        out.extend([r for r in rows if r["label"] == label][:n_per_class])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, default=None)
    ap.add_argument("--splits-dir", type=Path, default=Path("splits"))
    ap.add_argument("--runs-dir", type=Path, default=Path("results/runs"))
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    ap.add_argument("--models", default="all")
    ap.add_argument("--dataset", default="D1")
    ap.add_argument("--n-per-class", type=int, default=4)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.out.mkdir(parents=True, exist_ok=True)
    rows = read_split(Path(args.splits_dir) / f"{args.dataset}.csv", "test")
    samples = pick_samples(rows, args.n_per_class)
    if not samples:
        raise RuntimeError("测试集为空，检查 splits 目录")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wanted = None if args.models == "all" else {m.strip() for m in args.models.split(",")}
    run_dirs = []
    for p in sorted(Path(args.runs_dir).glob(f"*__{args.dataset}__*")):
        if not (p / "model.pt").exists():
            continue
        if wanted is None or p.name.split("__")[0] in wanted:
            run_dirs.append(p)
    print(f"[gradcam] 找到 {len(run_dirs)} 个带 model.pt 的 run", flush=True)

    for run_dir in run_dirs:
        model, ckpt = load_trained(run_dir, device)
        size = int(ckpt.get("input_size", 224))
        ims, cams, probs, trues = [], [], [], []
        for r in samples:
            img_path = resolve_path(args.cache_root, args.data_root, r["path"])
            x, im = load_image(img_path, size)
            cam, p = compute_cam(model, x.to(device), class_idx=1)
            ims.append(im)
            cams.append(cam)
            probs.append(p)
            trues.append(r["label"])

        n = len(samples)
        # (1) Grad-CAM 图
        fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 3.0))
        axes = np.atleast_1d(axes)
        for ax, im, cam, p, t in zip(axes, ims, cams, probs, trues):
            ax.imshow(im)
            ax.imshow(heat_overlay(im, cam), alpha=0.55)
            ax.axis("off")
            pred = "defect" if p[1] >= 0.5 else "nondefect"
            ax.set_title(f"true={t}\npred={pred} (p={p[1]:.2f})", fontsize=8)
        fig.suptitle(f"{run_dir.name} — Grad-CAM (red = model attention)", fontsize=10)
        fig.tight_layout()
        fig.savefig(args.out / f"gradcam_{run_dir.name}.png", dpi=140)
        plt.close(fig)

        # (2) 预测示例图
        fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.7))
        axes = np.atleast_1d(axes)
        for ax, im, p, t in zip(axes, ims, probs, trues):
            ax.imshow(im)
            ax.axis("off")
            pred = "defect" if p[1] >= 0.5 else "nondefect"
            mark = "OK" if pred == t else "WRONG"
            ax.set_title(f"true={t}\npred={pred} ({p[1]:.2f}) {mark}", fontsize=8)
        fig.suptitle(f"{run_dir.name} — predictions", fontsize=10)
        fig.tight_layout()
        fig.savefig(args.out / f"predict_{run_dir.name}.png", dpi=140)
        plt.close(fig)
        print(f"[gradcam] {run_dir.name} -> gradcam + predict", flush=True)


if __name__ == "__main__":
    main()
