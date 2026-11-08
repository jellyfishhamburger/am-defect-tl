# Author: Leiman Pan + agent | Task: checkpoint 复原（推理 / Grad-CAM 共用） | PR: 待提
# Status: 本组原创实现。解决"存档的分类头与 src/models.py 当前定义不一致"的问题。
"""从 results/runs/<run_id>/model.pt 复原一个可推理的模型。

为什么单独写一个文件：
    仓库里 5 个 model.pt 只保存了分类头（骨干权重是 ImageNet 预训练，随时可从 timm 重建）。
    但存档时用的是"全局平均池化"版头部（fc1 输入维度 = 通道数 C），
    而 src/models.py 的 PaperHead 在特征图 ≥ 4×4 时用 AvgPool2d(4)（fc1 输入维度 = C×16）。
    直接用 PaperHead 载入会抛：
        RuntimeError: size mismatch for head.fc1.weight: copying a param with shape
        torch.Size([64, 512]) ... current model has torch.Size([64, 8192])
    所以这里根据 checkpoint 里 fc1.weight 的真实形状选择池化方式，
    两种存档都能载入，且不需要改 src/models.py（那是地基组的共享文件）。

用法：
    from analysis.checkpoint import load_model, list_checkpoints
    model, ckpt = load_model("results/runs/vgg16__D1__adam__s42", device="cpu")
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.models import TransferModel


class GlobalPoolHead(nn.Module):
    """与存档一致的分类头：全局平均池化 → Flatten → Linear → ReLU → Dropout → Linear。"""

    def __init__(self, in_ch: int, num_classes: int = 2, dense_units: int = 64,
                 dropout: float = 0.5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_ch, dense_units)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_units, num_classes)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(self.pool(feat), 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(self.drop(x))


def load_model(run_dir: str | Path, device: str | torch.device = "cpu",
               pretrained: bool = True) -> tuple[TransferModel, dict]:
    """复原模型。返回 (model, checkpoint 原始 dict)。

    pretrained=True 会从 timm 拉 ImageNet 权重（首次需要联网，之后有缓存）；
    demo 前请先跑一次 `python -m analysis.demo_app --warmup` 把权重下载好。
    """
    run_dir = Path(run_dir)
    ckpt_path = run_dir / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} 不存在（权重不进 Git 的 run 无法复原）")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TransferModel(
        ckpt["model"],
        num_classes=int(ckpt.get("num_classes", 2)),
        pretrained=pretrained,
        freeze=ckpt.get("freeze", "all_conv"),
        dense_units=int(ckpt.get("dense_units", 64)),
        dropout=float(ckpt.get("dropout", 0.5)),
        input_size=int(ckpt.get("input_size", 224)),
    )

    state = ckpt["head_state"]
    fc1_w = state["fc1.weight"]
    in_ch = int(ckpt.get("feat_shape", [fc1_w.shape[1]])[0])
    if int(fc1_w.shape[1]) == in_ch:
        head = GlobalPoolHead(in_ch, int(ckpt.get("num_classes", 2)),
                              int(ckpt.get("dense_units", 64)),
                              float(ckpt.get("dropout", 0.5)))
    elif int(fc1_w.shape[1]) == model.head.fc1.in_features:
        head = model.head
    else:
        raise RuntimeError(
            f"分类头形状无法匹配：存档 fc1={tuple(fc1_w.shape)}，"
            f"特征通道 C={in_ch}，当前 PaperHead 期望 {model.head.fc1.in_features}。"
            "请检查 checkpoint 是否来自另一个模型。")

    head.load_state_dict(state)
    model.head = head
    model.to(device).eval()
    return model, ckpt


def list_checkpoints(runs_dir: str | Path = "results/runs") -> list[Path]:
    """所有带 model.pt 的 run 目录，按数据集/模型名排序。"""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    return sorted(p for p in runs_dir.iterdir() if (p / "model.pt").exists())


def run_meta(run_dir: str | Path) -> dict:
    """从同目录 metrics.json 读展示用元信息（没有就返回空 dict）。"""
    import json

    p = Path(run_dir) / "metrics.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))
