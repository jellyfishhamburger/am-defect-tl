# Author: 组长 + agent | Task: 模型工厂（11 骨干 + 冻结 + 论文同款头） | PR: 待提
# Status: 本组原创实现。使用 timm 的 ImageNet 预训练权重（未修改，见 NOTICE.md）
"""11 个骨干 + 冻结卷积层 + 论文同款分类头。

论文设定（arXiv:2310.08645 §4.3 + 作者 notebook 核对）：
    骨干输出 → AveragePooling2D((4,4)) → Flatten → Dense(64, relu) → Dropout(0.5) → Dense(2)
    训练时冻结全部卷积层，只训练新分类头。

统一用 timm 加载（torchvision 缺 NASNet / Xception / InceptionResNetV2 / EfficientNetV2M）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# config 里的 model 名 → timm 模型名（列表 = 候选名，按顺序尝试，取第一个能加载预训练权重的）
# 说明：timm 里部分旧名字已不带默认预训练权重（如 efficientnet_b7），
#       所以这里给候选回退，避免单个模型卡住整批实验。
BACKBONES = {
    "vgg16": ["vgg16"],
    "vgg19": ["vgg19"],
    "resnet50": ["resnet50"],
    "resnet101": ["resnet101"],
    "inception_resnet_v2": ["inception_resnet_v2"],
    "xception": ["xception"],
    "mobilenet_v2": ["mobilenetv2_100"],
    "efficientnet_b0": ["efficientnet_b0", "tf_efficientnet_b0"],
    "efficientnet_b7": ["tf_efficientnet_b7", "efficientnet_b7.ra_in1k", "efficientnet_b7"],
    "efficientnet_v2m": ["tf_efficientnetv2_m", "efficientnetv2_m"],
    "nasnet_large": ["nasnetalarge"],
}


class PaperHead(nn.Module):
    """论文同款分类头：AvgPool(4,4) → Flatten → Dense(64,relu) → Dropout → Dense(2)。"""

    def __init__(self, in_ch: int, num_classes: int = 2, dense_units: int = 64,
                 dropout: float = 0.5, pool: int = 4, spatial: int = 7):
        super().__init__()
        # 论文用 pool_size=(4,4)；若特征图小于 4×4 则退化为全局平均（日志里会说明）
        self.pool = nn.AvgPool2d(pool) if spatial >= pool else nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_ch, dense_units)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_units, num_classes)

    def forward(self, x):
        x = torch.flatten(self.pool(x), 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


class TransferModel(nn.Module):
    """骨干冻结 + 可训练分类头。forward 时骨干包在 no_grad 里，省显存也更快。"""

    def __init__(self, name: str, num_classes: int = 2, pretrained: bool = True,
                 freeze: str = "all_conv", dense_units: int = 64,
                 dropout: float = 0.5, input_size: int = 224):
        super().__init__()
        import timm

        if name not in BACKBONES:
            raise KeyError(f"未知模型 {name}，可选：{list(BACKBONES)}")
        self.name = name
        self.freeze = freeze

        # 候选名依次尝试：某些 timm 旧名字没有默认预训练权重
        last_err = None
        self.net = None
        for candidate in BACKBONES[name]:
            try:
                self.net = timm.create_model(candidate, pretrained=pretrained, num_classes=0)
                self.timm_name = candidate
                break
            except Exception as exc:
                last_err = exc
                print(f"[models] {name} 候选 {candidate} 加载失败：{type(exc).__name__}: {exc}", flush=True)
        if self.net is None:
            raise RuntimeError(f"{name} 所有候选名都加载失败，最后一个错误：{last_err}")

        if freeze == "all_conv":
            for p in self.net.parameters():
                p.requires_grad = False

        # 探针：跑一张全零图，拿到通道数与特征图尺寸，避免手写每个模型的形状
        self.net.eval()
        with torch.no_grad():
            feat = self.net.forward_features(torch.zeros(1, 3, input_size, input_size))
        if feat.dim() != 4:
            raise RuntimeError(f"{name} 的 forward_features 返回 {tuple(feat.shape)}，期望 4 维特征图")
        self.feat_shape = tuple(int(v) for v in feat.shape[1:])  # (C, H, W)
        in_ch, h, w = self.feat_shape
        self.head = PaperHead(in_ch, num_classes, dense_units, dropout,
                              pool=4, spatial=min(h, w))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze == "all_conv":
            self.net.eval()  # 冻结时保持 BN 的统计量不变
        return self

    def forward(self, x):
        if self.freeze == "all_conv":
            with torch.no_grad():
                feat = self.net.forward_features(x)
        else:
            feat = self.net.forward_features(x)
        return self.head(feat)


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
