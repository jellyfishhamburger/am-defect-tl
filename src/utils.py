# Author: 组长 + agent | Task: 通用工具（随机种子 / 目录 / JSON） | PR: 待提
# Status: 本组原创实现
"""通用工具函数。"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    """固定所有随机源，保证同一配置可复现。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_json(path: str | Path, obj) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def make_run_id(model: str, dataset: str, optimizer: str, seed: int) -> str:
    return f"{model}__{dataset}__{optimizer}__s{seed}"


def git_commit() -> str:
    """尽力读取当前 commit；读不到就返回 unknown（Kaggle 上代码是 zip 解压的，通常 unknown）。"""
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"
