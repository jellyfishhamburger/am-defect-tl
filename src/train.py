# Author: 组长 + agent | Task: 训练循环（单 run） | PR: 待提
# Status: 本组原创实现
"""单个 run 的训练入口。产物写入 <out>/runs/<run_id>/（字段见 docs/RESULTS.md）。

用法：
    python -m src.train --config configs/vgg16.yaml \
        --data-root /kaggle/input/datasets/kennethdeng/am-defect-v1 --splits-dir splits --out results
    # 冒烟（20 张图 / 1 epoch）
    python -m src.train --config configs/vgg16.yaml --data-root ... --smoke
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import evaluate as ev
from .dataset import AMDefectDataset, class_counts, read_split
from .models import TransferModel, count_params
from .utils import dump_json, ensure_dir, git_commit, make_run_id, set_seed

SMOKE_LIMITS = {"train": 40, "val": 20, "test": 20}


def load_config(path: Path) -> dict:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("model", "dataset", "optimizer", "lr", "batch_size", "epochs"):
        if key not in cfg:
            raise KeyError(f"config 缺少必填字段：{key}")
    cfg.setdefault("seed", 42)
    cfg.setdefault("input_size", 224)
    cfg.setdefault("num_classes", 2)
    cfg.setdefault("freeze", "all_conv")
    cfg.setdefault("pretrained", True)
    cfg.setdefault("augment", False)
    cfg.setdefault("num_workers", 2)
    cfg.setdefault("amp", True)
    cfg.setdefault("weight_decay", 0.0)
    if "augment" not in cfg:
        cfg["augment"] = bool((cfg.get("train_aug") or {}).get("enabled", False))
    head = cfg.get("head") or {}
    cfg["dense_units"] = int(head.get("dense_units", 64))
    cfg["dropout"] = float(head.get("dropout", 0.5))
    return cfg


def build_optimizer(model: nn.Module, cfg: dict):
    params = [p for p in model.parameters() if p.requires_grad]
    name = str(cfg["optimizer"]).lower()
    kw = {"lr": float(cfg["lr"]), "weight_decay": float(cfg["weight_decay"])}
    if name == "adam":
        return torch.optim.Adam(params, **kw)
    if name == "sgd":
        return torch.optim.SGD(params, momentum=0.9, **kw)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, **kw)
    raise ValueError(f"未知优化器：{cfg['optimizer']}（可选 adam / sgd / rmsprop）")


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type="cuda", enabled=(scaler is not None)):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        total_loss += float(loss.item()) * y.size(0)
        correct += int((logits.argmax(1) == y).sum().item())
        n += y.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


def predict(model, loader, device):
    import torch

    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device, non_blocking=True))
            probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
            ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(probs).numpy()


def run(cfg: dict, data_root: Path, splits_dir: Path, out_root: Path,
        smoke: bool = False, force: bool = False, cache_root: Path | None = None) -> Path:
    run_id = make_run_id(cfg["model"], cfg["dataset"], cfg["optimizer"], int(cfg["seed"]))
    run_dir = ensure_dir(Path(out_root) / "runs" / run_id)
    if (run_dir / "metrics.json").exists() and not force and not smoke:
        print(f"[train] 跳过已存在：{run_id}")
        return run_dir

    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] {run_id}  device={device}  smoke={smoke}", flush=True)

    epochs = 1 if smoke else int(cfg["epochs"])
    split_csv = Path(splits_dir) / f"{cfg['dataset']}.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"找不到划分文件 {split_csv}，先运行 python -m src.splits")

    loaders = {}
    for split in ("train", "val", "test"):
        rows = read_split(split_csv, split)
        if not rows:
            raise RuntimeError(f"{split_csv} 里 {split} 为空")
        ds = AMDefectDataset(rows, data_root, int(cfg["input_size"]),
                             train=(split == "train"), augment=bool(cfg["augment"]),
                             limit=SMOKE_LIMITS[split] if smoke else None,
                             cache_root=cache_root)
        loaders[split] = (DataLoader(
            ds, batch_size=int(cfg["batch_size"]), shuffle=(split == "train"),
            num_workers=int(cfg["num_workers"]), pin_memory=(device.type == "cuda"),
            drop_last=False,
        ), class_counts(ds.rows), len(ds))

    model = TransferModel(
        cfg["model"], num_classes=int(cfg["num_classes"]),
        pretrained=bool(cfg["pretrained"]), freeze=str(cfg["freeze"]),
        dense_units=cfg["dense_units"], dropout=cfg["dropout"],
        input_size=int(cfg["input_size"]),
    ).to(device)
    total_p, train_p = count_params(model)
    print(f"[train] 特征图形状={model.feat_shape}  总参数={total_p:,}  可训练={train_p:,}", flush=True)

    optimizer = build_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if (cfg["amp"] and device.type == "cuda") else None

    t0 = time.time()
    log = []
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, loaders["train"][0], criterion, device, optimizer, scaler)
        va_loss, va_acc = run_epoch(model, loaders["val"][0], criterion, device)
        log.append({"epoch": epoch, "train_loss": round(tr_loss, 6), "train_acc": round(tr_acc, 6),
                    "val_loss": round(va_loss, 6), "val_acc": round(va_acc, 6)})
        print(f"[train] epoch {epoch}/{epochs}  train_loss={tr_loss:.4f} acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f}", flush=True)
    elapsed = time.time() - t0

    metrics, confusion, sizes = {}, {}, {}
    for split in ("train", "val", "test"):
        loader = loaders[split][0]
        y_true, y_prob = predict(model, loader, device)
        metrics[split] = ev.compute_metrics(y_true, y_prob)
        confusion[split] = ev.confusion_matrix_2x2(y_true, y_prob)
        sizes[split] = len(loaders[split][0].dataset)
        print(f"[eval] {split}: " + "  ".join(f"{k}={metrics[split][k]:.3f}" for k in
                                             ("accuracy", "precision", "recall", "f1", "auc")), flush=True)

    with (run_dir / "train_log.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(log)

    dump_json(run_dir / "confusion_matrices.json", {
        "labels": ["nondefect", "defect"], **confusion,
        "note": "行=真实，列=预测，顺序同 labels",
    })
    dump_json(run_dir / "metrics.json", {
        "run_id": run_id,
        "git_commit": git_commit(),
        "author": "lead",
        "model": cfg["model"],
        "pretrained": "imagenet" if cfg["pretrained"] else None,
        "freeze": cfg["freeze"],
        "head": f"AvgPool2d(4)->Flatten->Dense({cfg['dense_units']})->Dropout({cfg['dropout']})->Dense(2)",
        "n_params_total": total_p,
        "n_params_trainable": train_p,
        "dataset": cfg["dataset"],
        "split_file": f"splits/{cfg['dataset']}.csv",
        "input_size": int(cfg["input_size"]),
        "n_train": sizes["train"], "n_val": sizes["val"], "n_test": sizes["test"],
        "optimizer": cfg["optimizer"], "lr": float(cfg["lr"]),
        "batch_size": int(cfg["batch_size"]), "epochs": epochs,
        "seed": int(cfg["seed"]), "device": str(device),
        "train_time_s": round(elapsed, 2),
        "metrics": metrics,
        "confusion": confusion,
        "smoke": smoke,
        "annotations": {
            "augment": bool(cfg["augment"]),
            "note": "冒烟结果，不可用于报告" if smoke else "",
        },
    })
    if cfg.get("save_weights"):
        # 只存「骨干名 + 分类头权重」：骨干是可复原的 ImageNet 预训练权重，
        # 分类头仅 3 万参数（约 130 KB），checkpoint 极小，便于入库与传输。
        torch.save({
            "model": cfg["model"],
            "num_classes": int(cfg["num_classes"]),
            "freeze": str(cfg["freeze"]),
            "dense_units": int(cfg["dense_units"]),
            "dropout": float(cfg["dropout"]),
            "input_size": int(cfg["input_size"]),
            "feat_shape": list(model.feat_shape),
            "head_state": model.head.state_dict(),
            "run_id": run_id,
        }, run_dir / "model.pt")
        print(f"[train] 已保存分类头权重：{run_dir / 'model.pt'}", flush=True)

    try:
        ev.plot_curves(log, run_dir / "curves.png")
        for split in ("train", "val", "test"):
            ev.plot_confusion(confusion[split], f"{run_id} / {split}", run_dir / f"confusion_{split}.png")
    except Exception as exc:  # 画图失败不影响结果落盘
        print(f"[warn] 画图失败：{exc}", flush=True)

    print(f"[train] 完成 {run_id}  用时 {elapsed:.1f}s  -> {run_dir}", flush=True)
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, default=Path("splits"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--optimizer", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--model", default=None, help="覆盖 config 里的 model（如 vgg16）")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cache-root", type=Path, default=None,
                    help="预处理缓存目录（建议先跑 src.data_prepare）")
    ap.add_argument("--save-weights", action="store_true", help="保存分类头权重 model.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.save_weights:
        cfg["save_weights"] = True
    for key, val in (("optimizer", args.optimizer), ("dataset", args.dataset),
                     ("model", args.model), ("seed", args.seed)):
        if val is not None:
            cfg[key] = val
    run(cfg, args.data_root, args.splits_dir, args.out, smoke=args.smoke, force=args.force,
        cache_root=args.cache_root)


if __name__ == "__main__":
    main()
