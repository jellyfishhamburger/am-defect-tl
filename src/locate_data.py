# Author: 组长 + agent | Task: 自动定位数据集根目录 | PR: 待提
# Status: 本组原创实现
"""在 /kaggle/input 下自动找到数据集根目录（含 data/D1/defect 的那一层）。

为什么需要：
    Kaggle 的数据集挂载路径在不同环境/不同时间可能不同（例如
    /kaggle/input/<slug>/  或  /kaggle/input/datasets/<owner>/<slug>/），
    硬编码会在某些 kernel 上直接 FileNotFoundError。这里做一次搜索，兼容两种布局。

用法：
    python -m src.locate_data                     # 打印找到的路径，退出码 0/1
    python -m src.locate_data --diagnose          # 没找到时打印 /kaggle/input 的结构
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

MARKER = ("data", "D1", "defect")


def find_root(search_root: str = "/kaggle/input", max_depth: int = 5) -> str | None:
    base = Path(search_root)
    if not base.is_dir():
        return None
    base_depth = len(base.parts)
    for dirpath, dirnames, _ in os.walk(base):
        p = Path(dirpath)
        if (p / MARKER[0] / MARKER[1] / MARKER[2]).is_dir():
            return str(p)
        if len(p.parts) - base_depth >= max_depth:
            dirnames[:] = []
    return None


def diagnose(search_root: str = "/kaggle/input") -> None:
    base = Path(search_root)
    if not base.is_dir():
        print(f"[locate] {search_root} 不存在")
        return
    base_depth = len(base.parts)
    for dirpath, dirnames, filenames in os.walk(base):
        p = Path(dirpath)
        depth = len(p.parts) - base_depth
        if depth > 3:
            dirnames[:] = []
            continue
        print(f"[locate] {dirpath}  dirs={dirnames[:6]}  files={filenames[:3]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-root", default="/kaggle/input")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--wait", type=int, default=0,
                    help="最多等待多少秒（Kaggle 挂载可能延迟几秒到几十秒）")
    args = ap.parse_args()

    import time

    deadline = time.time() + max(args.wait, 0)
    attempt = 0
    while True:
        attempt += 1
        root = find_root(args.search_root)
        if root:
            print(root)  # 只输出路径，方便 shell 里 $(...) 取值
            return
        if time.time() >= deadline:
            break
        print(f"[locate] 第 {attempt} 次未找到，等待 10s 后重试 ...", file=sys.stderr)
        time.sleep(10)

    if args.diagnose:
        diagnose(args.search_root)
    print(f"[locate] 没找到包含 {'/'.join(MARKER)} 的目录", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
