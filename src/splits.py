# Author: 组长 + agent | Task: 数据扫描与划分（契约文件生成） | PR: 待提
# Status: 本组原创实现
"""扫描数据集 → 生成 splits/{D1,D2}.csv（全组唯一一份划分）。

数据目录约定（Kaggle 挂载或本地都适用）：
    <data-root>/data/D1/defect/*.JPG
    <data-root>/data/D1/nondefect/*.JPG
    <data-root>/data/D2/defect/*.png
    <data-root>/data/D2/nondefect/*.png

用法：
    python -m src.splits --data-root /kaggle/input/datasets/kennethdeng/am-defect-v1 --out splits
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

DATASETS = ["D1", "D2"]
LABELS = ["defect", "nondefect"]
RATIO = (0.7, 0.1, 0.2)  # train / val / test，对齐作者
SEED = 42


def scan(data_root: Path, dataset: str) -> list[dict]:
    """返回 [{'path': 相对路径, 'label': ..., 'source_id': 原图名去扩展名}]。"""
    rows: list[dict] = []
    base = Path(data_root) / "data" / dataset
    for label in LABELS:
        d = base / label
        if not d.is_dir():
            raise FileNotFoundError(f"找不到目录：{d}")
        for f in sorted(p for p in d.iterdir() if p.is_file()):
            rows.append({
                # 统一用 POSIX 分隔符，保证 Windows 上生成的 csv 在 Kaggle(Linux) 也能用
                "path": (Path("data") / dataset / label / f.name).as_posix(),
                "label": label,
                "source_id": f.stem,
            })
    if not rows:
        raise RuntimeError(f"{base} 下没有找到任何图片")
    return rows


def stratified_split(rows: list[dict], seed: int = SEED) -> list[dict]:
    """按类别分层抽 70/10/20；同一 seed 结果完全确定（按 label 固定偏移）。"""
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    out: list[dict] = []
    for offset, label in enumerate(LABELS):
        items = by_label[label][:]
        random.Random(seed + offset).shuffle(items)
        n = len(items)
        n_train = int(round(n * RATIO[0]))
        n_val = int(round(n * RATIO[1]))
        for i, r in enumerate(items):
            r = dict(r)
            r["split"] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
            out.append(r)
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "split", "source_id"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def generate_all(data_root: Path, out_dir: Path, seed: int = SEED) -> dict:
    """生成 D1/D2 两个 csv，返回每份的统计。"""
    stats = {}
    for ds in DATASETS:
        rows = stratified_split(scan(data_root, ds), seed)
        write_csv(rows, Path(out_dir) / f"{ds}.csv")
        counts = defaultdict(int)
        for r in rows:
            counts[r["split"]] += 1
        stats[ds] = {"total": len(rows), **counts}
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("splits"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    stats = generate_all(args.data_root, args.out, args.seed)
    for ds, s in stats.items():
        print(f"[splits] {ds}: total={s['total']}  train={s['train']}  val={s['val']}  test={s['test']}")


if __name__ == "__main__":
    main()
