# Author: Leiman Pan + agent | Task: 生成结果看板（单文件 HTML） | PR: 待提
# Status: 本组原创实现。只读 results/ 下的交付物，不改任何实验结果。
"""把 results/ 里的 66 个 run 汇总成一个自包含的看板页面。

用法：
    python tools/build_dashboard.py                # → docs/index.html
    python tools/build_dashboard.py --out x.html

设计约束：
  1. 页面不依赖任何 CDN / 外链字体（答辩现场可能断网），所有数据与脚本内嵌；
  2. 数据只从 results/runs/*/metrics.json、confusion_matrices.json、train_log.csv
     与 results/leaderboard.csv 读取，不重算指标，避免"看板和交付物不一致"；
  3. 模板与数据分离：tools/dashboard/template.html 里的 __DATA_JSON__ 被替换，
     重复运行幂等（不会叠加注入）。
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
TEMPLATE = ROOT / "tools" / "dashboard" / "template.html"

METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "mcc", "kappa", "auc", "map",
               "sensitivity", "specificity"]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_curve(path: Path) -> dict:
    """train_log.csv → 4 条曲线（epoch 为 x 轴）。"""
    if not path.exists():
        return {"ep": [], "tr_acc": [], "va_acc": [], "tr_loss": [], "va_loss": []}
    rows = read_csv(path)
    curve = {"ep": [], "tr_acc": [], "va_acc": [], "tr_loss": [], "va_loss": []}
    for r in rows:
        if not r.get("epoch"):
            continue
        curve["ep"].append(int(float(r["epoch"])))
        curve["tr_acc"].append(fnum(r.get("train_acc"), 0.0))
        curve["va_acc"].append(fnum(r.get("val_acc"), 0.0))
        curve["tr_loss"].append(fnum(r.get("train_loss"), 0.0))
        curve["va_loss"].append(fnum(r.get("val_loss"), 0.0))
    return curve


def build_runs() -> list[dict]:
    board = {r["run_id"]: r for r in read_csv(ROOT / "results" / "leaderboard.csv")}
    runs = []
    for run_dir in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        rid = run_dir.name
        mpath = run_dir / "metrics.json"
        if not mpath.exists():
            continue
        meta = json.loads(mpath.read_text(encoding="utf-8"))
        row = board.get(rid, {})

        cm = {}
        cm_path = run_dir / "confusion_matrices.json"
        if cm_path.exists():
            raw = json.loads(cm_path.read_text(encoding="utf-8"))
            cm = {k: raw[k] for k in ("train", "val", "test") if k in raw}

        m = meta.get("metrics", {})
        entry = {
            "id": rid,
            "model": meta.get("model") or row.get("model"),
            "dataset": meta.get("dataset") or row.get("dataset"),
            "opt": meta.get("optimizer") or row.get("optimizer"),
            "seed": meta.get("seed"),
            "freeze": meta.get("freeze", ""),
            "input_size": meta.get("input_size"),
            "n_train": meta.get("n_train"),
            "n_val": meta.get("n_val"),
            "n_test": meta.get("n_test"),
            "params": meta.get("n_params_trainable") or fnum(row.get("n_params_trainable")),
            "params_total": meta.get("n_params_total"),
            "time": fnum(meta.get("train_time_s") or row.get("train_time_s")),
            "device": meta.get("device", ""),
            "acc": {s: fnum(m.get(s, {}).get("accuracy")) for s in ("train", "val", "test")},
            "test": {k: fnum(m.get("test", {}).get(k)) for k in METRIC_KEYS},
            "paper_acc": fnum(row.get("paper_acc")),
            "paper_f1": fnum(row.get("paper_f1")),
            "delta": fnum(row.get("delta_acc")),
            "cm_test": cm.get("test"),
            "note": (meta.get("annotations") or {}).get("notes", ""),
        }
        entry["curve"] = load_curve(run_dir / "train_log.csv")
        runs.append(entry)
    return runs


def summarise(runs: list[dict]) -> None:
    print(f"[dashboard] runs = {len(runs)}")
    for ds in sorted({r["dataset"] for r in runs}):
        sub = [r for r in runs if r["dataset"] == ds]
        best = max(sub, key=lambda r: r["acc"]["test"] or 0)
        print(f"  {ds}: {len(sub)} runs | best {best['model']}/{best['opt']} "
              f"acc={best['acc']['test']:.4f} (n_test={best['n_test']})")
    params = [r["params"] for r in runs if r["params"]]
    times = [r["time"] for r in runs if r["time"]]
    print(f"  trainable params {min(params):,}–{max(params):,} | "
          f"train time {min(times):.1f}–{max(times):.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "index.html"))
    args = ap.parse_args()

    runs = build_runs()
    if not runs:
        raise SystemExit("results/runs/ 下没有可用的 metrics.json")
    summarise(runs)

    payload = {
        "generated": date.today().isoformat(),
        "n_runs": len(runs),
        "models": sorted({r["model"] for r in runs}),
        "datasets": sorted({r["dataset"] for r in runs}),
        "optimizers": sorted({r["opt"] for r in runs}),
        "runs": runs,
    }
    data_js = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    html = TEMPLATE.read_text(encoding="utf-8")
    if "__DATA_JSON__" not in html:
        raise SystemExit("模板缺少 __DATA_JSON__ 占位符")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("__DATA_JSON__", data_js), encoding="utf-8")
    print(f"[dashboard] wrote {out} ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
