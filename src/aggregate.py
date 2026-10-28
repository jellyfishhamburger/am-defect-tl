# Author: 组长 + agent | Task: 结果汇总（leaderboard） | PR: 待提
# Status: 本组原创实现
"""汇总 results/runs/*/metrics.json → results/leaderboard.csv

同时读 docs/paper_results.csv（作者的 297 行报告值），算出 delta_acc / delta_f1，
方便直接看"我们和论文差多少"。

用法：
    python -m src.aggregate                          # 生成 leaderboard
    python -m src.aggregate --check                  # 只检查完整性
    python -m src.aggregate --out results/leaderboard.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

METRICS = ["accuracy", "precision", "recall", "f1", "mcc", "kappa",
           "auc", "map", "sensitivity", "specificity"]

FIELDS = ["run_id", "model", "dataset", "optimizer", "seed", *METRICS,
          "n_params_trainable", "train_time_s", "git_commit",
          "paper_acc", "paper_f1", "delta_acc"]

EXPECTED_RUNS = 11 * 3 * 2  # 66


def collect(runs_dir: Path) -> list[dict]:
    """读所有 metrics.json，返回按 (dataset, optimizer, model) 排序的记录。"""
    records = []
    for path in sorted(Path(runs_dir).glob("*/metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("smoke"):
            continue  # 冒烟结果不进正式榜
        test = data.get("metrics", {}).get("test", {})
        rec = {
            "run_id": data.get("run_id", path.parent.name),
            "model": data.get("model"),
            "dataset": data.get("dataset"),
            "optimizer": data.get("optimizer"),
            "seed": data.get("seed"),
            "n_params_trainable": data.get("n_params_trainable"),
            "train_time_s": data.get("train_time_s"),
            "git_commit": data.get("git_commit"),
        }
        for m in METRICS:
            rec[m] = test.get(m)
        records.append(rec)
    records.sort(key=lambda r: (str(r["dataset"]), str(r["optimizer"]), str(r["model"])))
    return records


def load_paper(path: Path) -> dict[tuple, dict]:
    """读作者报告值 → {(model, dataset, optimizer): {'accuracy':..,'f1':..}}（取 test 行）。"""
    out: dict[tuple, dict] = {}
    p = Path(path)
    if not p.exists():
        return out
    with p.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("split") != "test":
                continue
            key = (row["model"], row["dataset"], row["optimizer"])
            out[key] = {"accuracy": float(row["accuracy"]), "f1": float(row["f1"])}
    return out


def attach_paper(records: list[dict], paper: dict) -> list[dict]:
    for r in records:
        ref = paper.get((r["model"], r["dataset"], r["optimizer"]))
        r["paper_acc"] = ref["accuracy"] if ref else None
        r["paper_f1"] = ref["f1"] if ref else None
        r["delta_acc"] = (round(r["accuracy"] - ref["accuracy"], 4)
                          if ref and r["accuracy"] is not None else None)
    return records


def write_leaderboard(records: list[dict], out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(records)


def report(records: list[dict]) -> None:
    print(f"{'model':22s} {'set':4s} {'opt':8s} {'acc':>6s} {'f1':>6s} {'paper_acc':>9s} {'Δacc':>7s}")
    for r in records:
        pa = f"{r['paper_acc']:.3f}" if r["paper_acc"] is not None else "-"
        da = f"{r['delta_acc']:+.3f}" if r["delta_acc"] is not None else "-"
        print(f"{r['model']:22s} {r['dataset']:4s} {r['optimizer']:8s} "
              f"{r['accuracy']:6.3f} {r['f1']:6.3f} {pa:>9s} {da:>7s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=Path("results/runs"))
    ap.add_argument("--out", type=Path, default=Path("results/leaderboard.csv"))
    ap.add_argument("--paper", type=Path, default=Path("docs/paper_results.csv"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    records = collect(args.runs_dir)
    paper = load_paper(args.paper)
    attach_paper(records, paper)

    print(f"[aggregate] 正式 run 数：{len(records)} / {EXPECTED_RUNS}，"
          f"作者对标值 {len(paper)} 条")
    if args.check:
        missing = EXPECTED_RUNS - len(records)
        print("[aggregate] " + ("完整 ✅" if missing == 0 else f"还缺 {missing} 个 run"))
        raise SystemExit(0 if missing == 0 else 1)

    report(records)
    write_leaderboard(records, args.out)
    print(f"[aggregate] 已写出 {args.out}")


if __name__ == "__main__":
    main()
