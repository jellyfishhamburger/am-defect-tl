#!/usr/bin/env bash
# 一条命令跑完从数据到交付物的流水线（T6）
#
#   ./reproduce.sh --dry-run                 # 只打印将要执行的命令，不跑
#   DATA_ROOT=/kaggle/input/... ./reproduce.sh
#   ./reproduce.sh --skip-train --with-analysis
#
# 默认假设在 Kaggle Notebook 里跑（缓存写 /kaggle/temp）。本地跑请显式给 --cache。
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-}"
CACHE="${CACHE:-/kaggle/temp/cache}"
OUT="${OUT:-results}"
PY="${PYTHON:-python}"
DRY=0
SKIP_TRAIN=0
WITH_ANALYSIS=0
MODELS=""
DATASETS=""
LIMIT=""

usage() {
  cat <<'EOF'
用法: ./reproduce.sh [options]

  --data-root PATH   数据根目录（含 data/D1/...）；不给则用 src.locate_data 自动找
  --cache PATH       224 缓存目录（默认 /kaggle/temp/cache；本地请改成可写目录）
  --out PATH         结果目录（默认 results）
  --models LIST      只跑指定模型，逗号分隔（默认 all）
  --datasets LIST    只跑指定数据集（默认 all）
  --limit-runs N     矩阵只跑前 N 个（调试用）
  --skip-train       跳过缓存/划分/训练，只重算汇总、图、表
  --with-analysis    额外跑 efficiency + lime（CPU，较慢）
  --dry-run          只打印命令
  -h, --help         显示本帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --cache) CACHE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --limit-runs) LIMIT="$2"; shift 2 ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --with-analysis) WITH_ANALYSIS=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage; exit 2 ;;
  esac
done

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
do_run() {
  if [[ $DRY -eq 1 ]]; then
    printf '  [dry-run] %s\n' "$*"
  else
    printf '  $ %s\n' "$*"
    "$@"
  fi
}

if [[ -z "$DATA_ROOT" ]]; then
  if [[ $DRY -eq 1 ]]; then
    DATA_ROOT="<auto: python -m src.locate_data>"
    printf '  [dry-run] %s -m src.locate_data --search-root /kaggle/input --wait 300\n' "$PY"
  else
    step "0/7 定位数据"
    DATA_ROOT="$("$PY" -m src.locate_data --search-root /kaggle/input --wait 300)"
    echo "  DATA_ROOT=$DATA_ROOT"
  fi
fi

if [[ $SKIP_TRAIN -eq 0 ]]; then
  step "1/7 建 224 缓存"
  do_run "$PY" -m src.data_prepare --data-root "$DATA_ROOT" --out "$CACHE" --size 224

  step "2/7 生成划分（已有则跳过；splits/ 冻结后不要重跑）"
  if [[ -f splits/D1.csv && -f splits/D2.csv ]]; then
    echo "  splits/D1.csv 与 D2.csv 已存在，跳过（要重建请先备份并确认全组同意）"
  else
    do_run "$PY" -m src.splits --data-root "$DATA_ROOT" --out splits --seed 42
  fi

  step "3/7 训练矩阵（66 run，GPU）"
  BATCH_ARGS=(--data-root "$DATA_ROOT" --cache-root "$CACHE" --out "$OUT" --save-weights)
  [[ -n "$MODELS" ]] && BATCH_ARGS+=(--models "$MODELS")
  [[ -n "$DATASETS" ]] && BATCH_ARGS+=(--datasets "$DATASETS")
  [[ -n "$LIMIT" ]] && BATCH_ARGS+=(--limit-runs "$LIMIT")
  do_run "$PY" -m src.batch "${BATCH_ARGS[@]}"
else
  step "1–3/7 已跳过训练（--skip-train）"
fi

step "4/7 汇总 leaderboard.csv"
do_run "$PY" -m src.aggregate

step "5/7 报告图表（T1）"
do_run "$PY" -m analysis.make_figures

step "6/7 报告表格（T2，含逐格校验）"
do_run "$PY" -m analysis.export_tables --check

if [[ $WITH_ANALYSIS -eq 1 ]]; then
  step "7/7 效率（T3）与 LIME（T4）"
  do_run "$PY" -m analysis.efficiency
  do_run "$PY" -m analysis.lime_explain
else
  step "7/7 跳过 efficiency / lime（加 --with-analysis 才会跑）"
fi

printf '\n完成。产物：\n'
printf '  %s/leaderboard.csv\n  %s/figures/（fig_*.png + INDEX.md）\n  reports/tables/（*.md + *.tex）\n' "$OUT" "$OUT"
[[ $DRY -eq 1 ]] && printf '\n（以上为 dry-run，未执行任何命令）\n'
