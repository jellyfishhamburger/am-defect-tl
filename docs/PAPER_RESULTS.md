# 作者报告的数值（不是我们的结果）

`docs/paper_results.csv` 由 `tools/fetch_paper_results.py` 自动生成，**内容全部来自论文作者**。

| 项 | 说明 |
|---|---|
| 来源 | 作者仓库 <https://github.com/mahsan2/AM3d> 里每个 `StudyX/<优化器>/<模型>model_results.xlsx` 的 `Metrics` sheet |
| 规模 | 3 研究 × 3 优化器 × 11 模型 × 3 split = **297 行** |
| 生成 | `python tools/fetch_paper_results.py`（约 1 分钟，需联网 + openpyxl） |
| 数据集映射 | `Studyone → D1`、`Studytwo → D2`、`Studythree → D3` |

## ⚠️ 三条使用纪律

1. **这不是我们的实验结果。** 报告里引用时必须写成 "reported by Ahsan et al."，表格中用 † 或 * 标注；
   我们的结果只来自 `results/runs/*/metrics.json`。
2. **D3 的行只能引用、不能复现** —— 作者用的第三个数据集（1,557 张）未公开，我们的 `leaderboard.csv`
   里 D3 的 `delta_acc` 留空。
3. 这张表是**对标基线**，用来算 `delta_acc = 我们的 accuracy − 作者的 accuracy`，并写进报告的
   "reproduction gap" 讨论（差异主要来自：不同随机划分、PyTorch vs Keras、作者未公开的 head 细节）。

## 已发现的一处内部不一致（可写进报告）

`Study Three` 的 `resnet50 / adam / test`：accuracy 0.7961、precision 0.7541、recall 0.8846 ——
与论文正文（ESWA 版）表格的呈现方式不同，说明作者仓库 xlsx 与论文排版表格并非逐字一致。
引用时以 **xlsx 为准**（因为它是逐指标的原始记录），并在报告中注明这一点。
