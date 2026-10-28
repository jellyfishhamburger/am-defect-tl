# Author: 组长 + agent | Task: 图片预处理缓存 | PR: 待提
# Status: 本组原创实现
"""把原始图一次性缩到目标尺寸并缓存，避免每个 epoch 重复解码大图。

为什么需要：
    D1 原图 5184×3456（单张 3 MB），CPU 解码一张约 0.3 s；一个 epoch 要 271 张 → 80 s，
    30 个 epoch 就是 40 分钟/run，66 个 run 根本跑不完。
    预处理成 224×224 之后，一个 epoch 只要 1–2 秒。

规则（对齐作者实现：直接 resize，忽略长宽比）：
    D1: 5184×3456 → 224×224
    D2: 128×128   → 224×224
    全部保存为 JPEG（quality=92），文件名保留 stem。

用法：
    python -m src.data_prepare --data-root <原始数据根> --out /kaggle/working/cache --size 224
"""

from __future__ import annotations

import argparse
from pathlib import Path

DATASETS = ["D1", "D2"]
LABELS = ["defect", "nondefect"]


def prepare_one(src: Path, dst: Path, size: int, quality: int = 92) -> bool:
    """返回 True 表示真的写了文件，False 表示跳过（已存在）。"""
    if dst.exists():
        return False
    from PIL import Image

    img = Image.open(src)
    # draft：让 JPEG 解码器直接按 1/2、1/4、1/8 降采样解码，大图能快 5–8 倍
    img.draft("RGB", (size * 2, size * 2))
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, format="JPEG", quality=quality)
    return True


def prepare(data_root: Path, out_dir: Path, size: int = 224, quality: int = 92) -> dict:
    stats = {}
    for ds in DATASETS:
        written = skipped = 0
        for label in LABELS:
            src_dir = Path(data_root) / "data" / ds / label
            if not src_dir.is_dir():
                raise FileNotFoundError(f"找不到目录：{src_dir}")
            files = sorted(p for p in src_dir.iterdir() if p.is_file())
            for i, f in enumerate(files, 1):
                dst = Path(out_dir) / ds / label / f"{f.stem}.jpg"
                if prepare_one(f, dst, size, quality):
                    written += 1
                else:
                    skipped += 1
                if i % 500 == 0:
                    print(f"[prepare] {ds}/{label} {i}/{len(files)}", flush=True)
        stats[ds] = {"written": written, "skipped": skipped}
        print(f"[prepare] {ds}: 新写 {written}，跳过 {skipped}（缓存已存在）", flush=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args()

    stats = prepare(args.data_root, args.out, args.size, args.quality)
    total = sum(s["written"] + s["skipped"] for s in stats.values())
    print(f"[prepare] 完成，共 {total} 张图，缓存目录：{args.out}")


if __name__ == "__main__":
    main()
