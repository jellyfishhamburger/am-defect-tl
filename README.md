# am-defect-tl

**3D 打印件缺陷检测 · 迁移学习复现**

用 11 个 ImageNet 预训练 CNN 对 3D 打印件做「有缺陷 / 无缺陷」二分类，复现 Ahsan et al. (2024) 的对比实验：
11 个骨干 × 3 种优化器 × 2 个数据集 = **66 个完整实验**，附 Grad-CAM 可解释性分析。

课程：IDAT7215 Computer Programming for Product Development and Applications（HKU, 2026/27 Sem 1, Group Project 25%）

> 复现对象：M. M. Ahsan, S. Raman, Y. Liu, and Z. Siddique, "Defect analysis of 3D printed object using
> transfer learning approaches," *Expert Systems with Applications*, vol. 253, p. 124293, Nov. 2024,
> doi: [10.1016/j.eswa.2024.124293](https://doi.org/10.1016/j.eswa.2024.124293)

---

## 1. 这是什么

| 项 | 内容 |
|---|---|
| 任务 | 输入一张 3D 打印件照片 → 输出 `defect` / `nondefect`（图像二分类） |
| 方法 | **冻结** ImageNet 预训练骨干的全部卷积层，只训练新分类头 |
| 分类头 | `AvgPool2d(4,4) → Flatten → Linear(64, ReLU) → Dropout(0.5) → Linear(2)` |
| 骨干（11 个） | VGG16/19、ResNet50/101、InceptionResNetV2、Xception、MobileNetV2、EfficientNetB0/B7/V2M、NASNetLarge |
| 优化器（3 种） | Adam、SGD、RMSprop |
| 数据集（2 个） | D1（271 张真实照片）、D2（3,017 张增强图） |
| 实现 | PyTorch + timm（全部代码在本仓库，未复制作者代码） |
| 训练环境 | Kaggle Notebook（免费 GPU T4），无需本地显卡 |

**为什么冻结骨干**：数据只有几百到三千张，从零训练必然过拟合；借用 ImageNet 学到的通用视觉特征，只学习本任务的
分类边界。冻结后单次实验在 T4 上约 **30 秒–3 分钟**。

---

## 2. 数据

| 数据集 | 是什么 | 缺陷 | 无缺陷 | 分辨率 |
|---|---|---|---|---|
| **D1** | 真实拍摄的圆柱试样（延时摄影连续帧） | 136 | 135 | 5184×3456 |
| **D2** | **D1 的数据增强版**（旋转/翻转/调色） | 947 | 2,070 | 128×128 |

- 数据来源：<https://github.com/mahsan2/AMdataset>（University of Oklahoma, SMIS Laboratory）
- 仓库内**不含数据**；使用 Kaggle 私有数据集 [am-defect-v1](https://www.kaggle.com/datasets/kennethdeng/am-defect-v1)（854.6 MB）
- 挂载路径（新版 Kaggle）：`/kaggle/input/datasets/kennethdeng/am-defect-v1/data/{D1,D2}/{defect,nondefect}/`
- 代码用 `python -m src.locate_data` 自动定位，**不硬编码路径**
- 详见 [docs/DATA.md](docs/DATA.md)

**数据划分**（冻结，`splits/*.csv`，seed=42，分层 70/10/20）：

| 数据集 | train | val | test |
|---|---|---|---|
| D1 | 189 | 28 | 54 |
| D2 | 2,112 | 302 | 603 |

---

## 3. 实验结果（66 / 66 已完成）

完整结果见 **[results/leaderboard.csv](results/leaderboard.csv)**（含 10 个指标 + 作者报告值 + `Δacc`）。
每个 run 的产物在 `results/runs/<run_id>/`。

准确率（3 个优化器平均，我们 vs 论文）：

| 模型 | D1 本仓库 | D1 论文 | D2 本仓库 | D2 论文 |
|---|---|---|---|---|
| VGG16 | 0.994 | 0.867 | 0.991 | 0.911 |
| VGG19 | 1.000 | 0.867 | 0.989 | 0.930 |
| ResNet50 | 0.975 | 0.758 | 0.988 | 0.674 |
| ResNet101 | 0.975 | 0.745 | 0.988 | 0.792 |
| InceptionResNetV2 | 0.981 | 1.000 | 0.994 | 0.996 |
| Xception | 1.000 | 0.976 | 0.997 | 0.996 |
| MobileNetV2 | 1.000 | 0.994 | 0.995 | 0.995 |
| EfficientNetB0 | 1.000 | 0.491 | 0.996 | 0.674 |
| EfficientNetB7 | 0.975 | 0.491 | 0.996 | 0.674 |
| EfficientNetV2M | 0.981 | 0.491 | 0.920 | 0.674 |
| NASNetLarge | 0.926 | 0.988 | 0.980 | 0.996 |

三条主要发现：

1. **模型排序基本复现**：Xception / MobileNetV2 / InceptionResNetV2 与论文差 ≤0.02。
2. **EfficientNet 家族在论文中完全失效**（D1 上 F1 = 0.00，等于全猜一类），本仓库正常收敛到 0.98–1.00。
   最可能的原因：原作者对所有骨干统一用 `/255` 归一化，而 EfficientNet 需要自己的 `preprocess_input`。
3. **整体分数偏高（≈1.00）提示数据泄漏**：D1 是延时摄影连续帧，随机划分会让同一物体的相邻帧
   横跨训练/测试集。这是后续「按原图分组划分」对照实验的动机。

---

## 4. 仓库结构

```text
src/                 # 训练管线（10 个模块，全部已实现）
  locate_data.py     #   自动定位 Kaggle 数据挂载路径
  data_prepare.py    #   图片预处理缓存（大图 → 224，提速约 90 倍）
  splits.py          #   生成并冻结 train/val/test 划分
  dataset.py         #   按划分读图 + ImageNet 归一化
  models.py          #   11 个骨干 + 冻结逻辑 + 论文同款分类头
  train.py           #   单 run 训练 + 10 指标 + 混淆矩阵 + checkpoint
  batch.py           #   实验矩阵批跑（66 run，断点续跑）
  aggregate.py       #   leaderboard + 作者报告值对标
  evaluate.py        #   指标与绘图
  utils.py           #   种子、路径、JSON
analysis/            # 分析与可解释性
  gradcam.py         #   Grad-CAM 热力图 + 预测示例图（已实现）
  leakage.py         #   泄漏对照实验（未实现）
  lime_explain.py    #   LIME（未实现）
  efficiency.py      #   参数量/FLOPs/时延（未实现）
  demo_app.py        #   Gradio 演示页面（未实现）
configs/             # 实验配置模板
splits/              # 冻结的数据划分 csv（D1.csv / D2.csv）
results/
  runs/<run_id>/     #   每个 run：metrics.json + 混淆矩阵 + 训练曲线 + model.pt
  leaderboard.csv    #   66 行结果总表（脚本生成，禁止手改）
  figures/           #   Grad-CAM / 预测示例图
docs/                # 数据说明、结果格式、作者代码分析、论文数值对照
tools/               # 自包含 Kaggle kernel 构建器、作者结果表抽取脚本
tests/               # 契约测试（划分不泄漏 / metrics schema）
```

---

## 5. 如何复现

Kaggle Notebook（推荐，GPU T4）或任意有 `torch` + `timm` 的环境：

```bash
# 0) 依赖（Kaggle 镜像已预装 torch/timm，通常无需安装）
pip install -r requirements.txt

# 1) 自动定位数据目录（兼容 /kaggle/input/<slug> 与 /kaggle/input/datasets/<owner>/<slug>）
DATA=$(python -m src.locate_data --wait 300)

# 2) 预处理缓存：把 5184×3456 大图一次性缩到 224（必须做，否则每 epoch 解码约 80 秒）
python -m src.data_prepare --data-root "$DATA" --out /kaggle/temp/cache --size 224

# 3) 生成划分（固定 seed=42；仓库已附带，可跳过）
python -m src.splits --data-root "$DATA" --out splits --seed 42

# 4) 跑单个 run
python -m src.train --config configs/_template.yaml --model vgg16 \
    --data-root "$DATA" --cache-root /kaggle/temp/cache \
    --dataset D1 --optimizer adam --save-weights

# 5) 跑完整矩阵（66 个 run；已存在 metrics.json 的自动跳过）
python -m src.batch --data-root "$DATA" --out results --cache-root /kaggle/temp/cache

# 6) 汇总 leaderboard + 与论文对比
python -m src.aggregate

# 7) Grad-CAM / 预测示例图（依赖第 4 步的 model.pt）
python -m analysis.gradcam --data-root "$DATA" --cache-root /kaggle/temp/cache \
    --runs-dir results/runs --out results/figures --dataset D1
```

**推到 Kaggle 上跑**（无需 token / clone，代码 base64 内嵌）：

```bash
python tools/build_kernel.py --slug my-run --cmd "<上面的任意命令>"
```

---

## 6. 产物契约

- `run_id`：`<model>__<dataset>__<optimizer>__s<seed>`，例如 `vgg16__D1__adam__s42`
- 每个 run 目录固定包含：
  - `metrics.json`：train/val/test 各 10 个指标（accuracy, precision, recall, f1, mcc, kappa, auc, map, sensitivity, specificity）+ 混淆矩阵 + 超参 + `git_commit`
  - `confusion_matrices.json`、`confusion_{train,val,test}.png`、`curves.png`、`train_log.csv`
  - `model.pt`：**只含分类头权重**（131–515 KB），骨干由 timm 重建
- 字段定义与取值口径见 [docs/RESULTS.md](docs/RESULTS.md)

---

## 7. 引用与合规

- 论文与数据引用见上；数据来自作者公开仓库，**仓库未声明许可证**，仅用于课程复现，不对外分发
- 第三方组件与开源使用登记：[NOTICE.md](NOTICE.md)
- 作者仓库 `AM3d` 提供 3 个 Keras notebook 与 99 份结果表，但**未发布任何训练权重**；本仓库全部代码为
  PyTorch 重写，仅用其结果表做数值对照，详见 [docs/AUTHOR_CODE.md](docs/AUTHOR_CODE.md)

---

## 8. 已知限制

1. 论文使用 3 个数据集，**第三个（1,557 张）未公开** → 本仓库覆盖 66/99 个实验格，其余以作者报告值标注。
2. **D2 是 D1 的增强版**，并非独立采集 → 跨数据集结论需谨慎表述。
3. 作者未公开划分文件，本仓库用自己的冻结划分 → 绝对数值与论文不可逐格对齐。
4. 主矩阵未使用在线数据增强（作者 notebook 用了 `rotation_range=15`），差异已记录在 `docs/AUTHOR_CODE.md`。

---

## 9. 给 AI agent 的说明

- 仓库的机器可读约定、规范命令、不变量与已知坑：[**AGENTS.md**](AGENTS.md)
- jellyfishhamburger（负责非 GPU 训练类工作）的任务卡与协同规则：[**docs/TASKS.md**](docs/TASKS.md)
- 她交接给 Kenneth 的文档：[**docs/HANDOFF.md**](docs/HANDOFF.md)

维护分工：**Kenneth**（GitHub: `Zheyi-D`）负责 Kaggle 训练类实验与最终合并；**jellyfishhamburger** 负责图表、表格、推理类分析与 demo。
