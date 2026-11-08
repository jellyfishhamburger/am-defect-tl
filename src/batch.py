# Author: 组长 + agent | Task: 矩阵批量跑（66 runs） | PR: 待提
# Status: 本组原创实现
"""批量跑实验矩阵。

默认矩阵 = 11 模型 × 3 优化器 × 2 数据集 = 66 个 run。
已存在 metrics.json 的 run 自动跳过（可断点续跑）。

用法：
    # 冒烟：只跑 vgg16 / D1 / adam，20 张图 1 epoch
    python -m src.batch --data-root /kaggle/input/datasets/kennethdeng/am-defect-v1 \
        --out results --smoke --limit-runs 1

    # 正式：全矩阵
    python -m src.batch --data-root /kaggle/input/datasets/kennethdeng/am-defect-v1 \
        --out results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .train import load_config, run

MODELS = [
    "vgg16", "vgg19", "resnet50", "resnet101", "inception_resnet_v2",
    "xception", "mobilenet_v2", "efficientnet_b0", "efficientnet_b7",
    "efficientnet_v2m", "nasnet_large",
]
OPTIMIZERS = ["adam", "sgd", "rmsprop"]
DATASETS = ["D1", "D2"]

# 每个数据集的最优超参（论文 Study One / Study Two 的取值，见 docs/AUTHOR_CODE.md）
DATASET_HP = {
    "D1": {"batch_size": 8, "epochs": 30},
    "D2": {"batch_size": 32, "epochs": 50},
}


def build_matrix(models, optimizers, datasets) -> list[tuple[str, str, str]]:
    return [(m, d, o) for d in datasets for o in optimizers for m in models]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, default=Path("splits"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--template", type=Path, default=Path("configs/_template.yaml"))
    ap.add_argument("--models", default="all", help="逗号分隔，或 all")
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--optimizers", default="all")
    ap.add_argument("--limit-runs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cache-root", type=Path, default=None,
                    help="预处理缓存目录（强烈建议：否则每 epoch 都要解码 5184×3456 大图）")
    ap.add_argument("--save-weights", action="store_true", help="保存每个 run 的分类头权重")
    args = ap.parse_args()

    models = MODELS if args.models == "all" else [x.strip() for x in args.models.split(",")]
    datasets = DATASETS if args.datasets == "all" else [x.strip() for x in args.datasets.split(",")]
    optimizers = OPTIMIZERS if args.optimizers == "all" else [x.strip() for x in args.optimizers.split(",")]

    matrix = build_matrix(models, optimizers, datasets)
    if args.limit_runs:
        matrix = matrix[: args.limit_runs]
    print(f"[batch] 共 {len(matrix)} 个 run：{models} × {optimizers} × {datasets}", flush=True)

    base = load_config(args.template)
    done, failed = [], []
    for i, (model, dataset, optimizer) in enumerate(matrix, 1):
        cfg = dict(base)
        cfg.update({"model": model, "dataset": dataset, "optimizer": optimizer,
                    **DATASET_HP[dataset]})
        if args.save_weights:
            cfg["save_weights"] = True
        print(f"\n[batch] ({i}/{len(matrix)}) {model} / {dataset} / {optimizer}", flush=True)
        try:
            run(cfg, args.data_root, args.splits_dir, args.out,
                smoke=args.smoke, force=args.force, cache_root=args.cache_root)
            done.append(f"{model}__{dataset}__{optimizer}")
        except Exception as exc:  # 单个 run 失败不阻断整批
            print(f"[batch][ERROR] {model}/{dataset}/{optimizer}: {type(exc).__name__}: {exc}", flush=True)
            failed.append({"run": f"{model}__{dataset}__{optimizer}", "error": str(exc)})

    print(f"\n[batch] 完成 {len(done)} 个，失败 {len(failed)} 个", flush=True)
    if failed:
        summary = Path(args.out) / "batch_failures.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[batch] 失败清单：{summary}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
