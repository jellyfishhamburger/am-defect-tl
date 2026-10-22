# Author: <姓名> | Task: 冻结主干的特征缓存（加速 66 run） | PR: #<编号>
# Status: 本组原创实现
"""特征缓存：主干全冻结时，同一 (backbone, dataset, input_size) 的特征只需算一次。

原理：卷积层参数不动 → 输出特征固定 → 把特征存成 .npz，之后每个 run 只训练小分类头，
      1–2 分钟一个 run，把 66 run 的 GPU 时间从数十小时压到 ~1 小时。

缓存文件：data/cache/<backbone>__<dataset>__s<size>.npz
  keys: features(float32, N×C), labels(int64, N), paths(str, N)

⚠️ 使用约束：只在"不做在线数据增强"时有效。若实验含 on-the-fly 增强，
   必须关掉缓存走完整前向（在 metrics.json 的 annotations 里注明走了哪条路径）。
"""

from __future__ import annotations

from pathlib import Path


def cache_path(backbone: str, dataset: str, size: int, cache_dir: Path = Path("data/cache")) -> Path:
    return cache_dir / f"{backbone}__{dataset}__s{size}.npz"


def build_cache(backbone: str, dataset: str, size: int = 224, batch_size: int = 32) -> Path:
    """遍历数据集一次，把主干输出（pool 之后、分类头之前的张量）存盘。"""
    raise NotImplementedError


def load_cache(path: Path):
    """返回 (features, labels, paths)。"""
    raise NotImplementedError
