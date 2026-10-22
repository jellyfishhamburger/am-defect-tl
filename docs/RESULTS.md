# 结果格式契约（RESULTS）

> 目的：让 8 个人跑出来的 66 个 run 能**自动汇总、自动画图、自动和论文逐格比对**。
> 没有 `metrics.json` 的结果不算结果；字段名不许改。

## 1. run_id 命名

```text
<model>__<dataset>__<optimizer>__s<seed>
例：vgg16__D1__adam__s42
    efficientnet_b7__D2__sgd__s42
```

## 2. 目录

```text
results/runs/<run_id>/
├── metrics.json            # 必须
├── confusion_matrices.json # 必须
├── train_log.csv           # 必须（epoch, train_loss, train_acc, val_loss, val_acc, lr）
├── curves.png              # acc/loss 曲线
└── model.pt                # ❌ 不进 Git（.gitignore 已排除），需要时放 Kaggle Models
```

## 3. `metrics.json` 完整结构

字段直接对齐作者 `AM3d` 仓库 xlsx 的 4 个 sheet（Metrics / Train-Val-Test 混淆矩阵），便于逐格对比。

```json
{
  "run_id": "vgg16__D1__adam__s42",
  "git_commit": "a1b2c3d",
  "created_at": "2026-11-03T10:22:31+08:00",
  "author": "<成员姓名>",

  "model": "vgg16",
  "pretrained": "imagenet",
  "freeze": "all_conv",
  "head": "AvgPool2d(4)->Flatten->Dense(256)->Dropout(0.5)->Dense(2)",
  "n_params_total": 138357544,
  "n_params_trainable": 262656,

  "dataset": "D1",
  "split_file": "splits/D1.csv",
  "input_size": 224,
  "n_train": 190,
  "n_val": 27,
  "n_test": 55,

  "optimizer": "adam",
  "lr": 0.001,
  "batch_size": 8,
  "epochs": 30,
  "weight_decay": 0.0,
  "seed": 42,
  "device": "NVIDIA T4 x2",
  "train_time_s": 412.5,

  "metrics": {
    "train": {"accuracy": 0.968, "precision": 0.939, "recall": 1.0, "f1": 0.968,
              "mcc": 0.939, "kappa": 0.937, "auc": 1.0, "map": 1.0,
              "sensitivity": 1.0, "specificity": 0.939},
    "val":   {"accuracy": 1.0,   "precision": 1.0,   "recall": 1.0, "f1": 1.0,
              "mcc": 1.0, "kappa": 1.0, "auc": 1.0, "map": 1.0,
              "sensitivity": 1.0, "specificity": 1.0},
    "test":  {"accuracy": 1.0,   "precision": 1.0,   "recall": 1.0, "f1": 1.0,
              "mcc": 1.0, "kappa": 1.0, "auc": 1.0, "map": 1.0,
              "sensitivity": 1.0, "specificity": 1.0}
  },

  "annotations": {
    "split": "随机分层划分（seed 42）",
    "notes": "batch 50 显存不足，改为 32 + 梯度累积（作者未说明其做法）"
  }
}
```

## 4. 指标口径（10 个，全部为 0–1 小数）

| 字段 | 口径 |
|---|---|
| `accuracy` | (TP+TN) / N |
| `precision` | TP / (TP+FP) |
| `recall` = `sensitivity` | TP / (TP+FN) |
| `specificity` | TN / (TN+FP) |
| `f1` | 2PR/(P+R) |
| `mcc` | Matthews 相关系数（`sklearn.metrics.matthews_corrcoef`） |
| `kappa` | Cohen's kappa（`sklearn.metrics.cohen_kappa_score`） |
| `auc` | ROC 曲线下面积（正类 = defect） |
| `map` | average precision（`sklearn.metrics.average_precision_score`） |

> 类别定义：**positive = defect（缺陷）**，与论文一致（TP = 缺陷被判为缺陷）。
> 正类定义一旦写错，全部指标会翻转 —— 这是最容易犯的错，PR 里必须自查。

## 5. `confusion_matrices.json`

```json
{
  "labels": ["nondefect", "defect"],
  "train": [[92, 6], [0, 92]],
  "val":   [[11, 0], [0, 16]],
  "test":  [[27, 0], [0, 28]],
  "note": "行 = 真实，列 = 预测，顺序同 labels"
}
```

## 6. `results/leaderboard.csv`

由 `python -m src.aggregate` 生成，**禁止手改**。列：

```text
run_id,model,dataset,optimizer,seed,accuracy,precision,recall,f1,mcc,kappa,auc,map,
sensitivity,specificity,n_params_trainable,train_time_s,git_commit,paper_acc,paper_f1,delta_acc
```

- `paper_acc` / `paper_f1` / `delta_acc`：从作者 xlsx 读到的对应数值与差值（Study Three 的行留空）；
- 这张表就是报告 Table 的主干，也是 PPT 的核心图表数据源。

## 7. 图表规范（报告用）

- 全部由脚本从 `results/` 生成，**不许截图或手工拼图**；
- 输出到 `results/figures/`，命名 `fig_<内容>.pdf`（矢量）+ `fig_<内容>.png`（预览）；
- 字体统一 10pt，图注英文，正文用 `Figure 1` / `Table 1` 交叉引用；
- 必备图：① 11 模型 × 2 数据集准确率柱状图；② 3 优化器消融热力图；③ 混淆矩阵网格；
④ 效率散点图（参数量 vs 准确率）；⑤ 泄漏实验对照图；⑥ Grad-CAM / LIME 示例。
