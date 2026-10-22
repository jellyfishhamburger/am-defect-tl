# Author: <姓名> | Task: 模型工厂（11 骨干 + 冻结 + 统一 head） | PR: #<编号>
# Status: 本组原创实现。使用 torchvision 的 ImageNet 预训练权重（未修改，见 NOTICE.md）
"""模型工厂：11 个骨干网络 + 冻结卷积层 + 论文同款分类头。

论文设定（arXiv:2310.08645 §4.3）：
  去掉原 FC 层 → 接 AveragePooling2D((4,4)) → Flatten → Dense → Dropout(0.5) → Dense(softmax)
  训练时冻结全部卷积层，只训练新分类头。

11 个骨干（对应 configs/*.yaml 的 model 字段）：
  vgg16, vgg19, resnet50, resnet101, inception_resnet_v2, xception,
  mobilenet_v2, efficientnet_b0, efficientnet_b7, efficientnet_v2m, nasnet_large

注意：torchvision 没有 NASNetLarge / EfficientNetV2M 的完全等价实现，
      需在 PR 描述里写清我们用了什么替代（如 timm），并登记到 NOTICE.md。
"""

from __future__ import annotations

MODEL_NAMES = [
    "vgg16", "vgg19", "resnet50", "resnet101", "inception_resnet_v2",
    "xception", "mobilenet_v2", "efficientnet_b0", "efficientnet_b7",
    "efficientnet_v2m", "nasnet_large",
]


def build_backbone(name: str, pretrained: bool = True):
    """返回 (backbone, feature_dim)。backbone 输出应为 (B, C, H, W)。"""
    raise NotImplementedError


def build_head(feature_dim: int, num_classes: int = 2, dense_units: int = 256,
               dropout: float = 0.5, pooling: tuple[int, int] = (4, 4)):
    """论文同款分类头。dense_units 论文未给出 → 我们的选择，需写进报告限制说明。"""
    raise NotImplementedError


def freeze_backbone(model, mode: str = "all_conv"):
    """冻结卷积层，只留分类头可训练。mode: all_conv | none | top_k"""
    raise NotImplementedError


def build_model(name: str, num_classes: int = 2, pretrained: bool = True,
                freeze: str = "all_conv", dense_units: int = 256, dropout: float = 0.5):
    """组装完整模型，并返回可训练参数量的统计（写进 metrics.json）。"""
    raise NotImplementedError


def count_params(model) -> tuple[int, int]:
    """返回 (总参数量, 可训练参数量)。"""
    raise NotImplementedError
