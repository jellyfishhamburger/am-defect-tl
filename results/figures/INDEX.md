# Figure index（报告第 3 节用图）

> 由 `python -m analysis.make_figures` 生成，**不要手改**。数字全部读自 `results/leaderboard.csv` 与
> `results/runs/*/{metrics,confusion_matrices}.json`；论文值来自 `docs/paper_results.csv`。

| 图号 | 文件 | Caption (EN) | 说明 |
|---|---|---|---|
| Figure 1 | [`fig_accuracy_by_model.png`](fig_accuracy_by_model.png) | Test accuracy of 11 ImageNet backbones on D1 and D2 (Adam, seed 42). Bars are our runs; short horizontal ticks are the values reported by Ahsan et al. (2024). The y-axis starts at 0.40. | ① 11 模型 × 2 数据集准确率柱状图，短横线为论文值；纵轴从 0.40 起，读图时注意 |
| Figure 2 | [`fig_optimizer_heatmap.png`](fig_optimizer_heatmap.png) | Optimizer ablation: test accuracy of each backbone under Adam / SGD / RMSprop with a frozen backbone and head-only training (lr = 1e-3, 30 epochs). Colour scale spans the observed range. | ② 3 优化器消融热力图（D1 / D2 两栏）；色阶按实际最小值拉伸 |
| Figure 3 | [`fig_confusion_grid.png`](fig_confusion_grid.png) | Confusion matrices of the best and worst run per dataset (test split). Rows = true label, columns = predicted; each panel is shaded to its own maximum. | ③ 4 个代表 run 的混淆矩阵（每数据集最好 / 最差） |
| Figure 4 | [`fig_delta_acc.png`](fig_delta_acc.png) | Accuracy gap between our reproduction and the reported values (Adam runs). Pink labels mark the authors' EfficientNet-B0 / B7 / V2-M rows, whose reported F1 is 0.000 (majority-class collapse, D1 accuracy 0.4909) — those gaps are not evidence that our implementation is better. | ④ 我们 vs 论文的 Δaccuracy；浅红标注是作者侧塌成多数类的三行 |
| Figure 5 | [`fig_ranking.png`](fig_ranking.png) | Model ranking by test accuracy (best of the three optimizers). The line joins the best D1 and D2 run of each backbone; the right-hand column names the optimizer that produced the D1 best run. D1 test n=54, so ties within ~0.02 are not statistically meaningful. | ⑤ 模型准确率排名；右侧注明 D1 最优的优化器 |

## 生成命令

```bash
python -m analysis.make_figures            # 全部
python -m analysis.make_figures --only 1,3 # 只画 1、3
```

## 口径与已知问题

- 图 1 / 4 用 **Adam** run 与论文的 Adam 行对比；每个模型只比同一优化器，避免拿我们最好的和论文最差的比。
- D1 测试集只有 54 张：错 1 张 = 1.8 个百分点，所以 0.98 与 1.00 的差别不要过度解读。
- 论文的 EfficientNet-B0 / B7 / V2-M 报告 F1 = 0（全部预测为多数类），图 4 用浅红底标出，不能读作"我们更强"。
- 图 3 选的是每个数据集准确率最高 / 最低的 run，属于**示意**，不是随机抽样。
