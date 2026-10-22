# am-defect-tl

**IDAT7215 Computer Programming for Product Development and Applications** · 2026/27 Sem 1 · Group Project (25%)
**Topic 4** · Python-Based Transfer Learning for 3D Printed Object Defect Detection

复现 Ahsan et al. (2024) 的结论：11 个 ImageNet 预训练 CNN 在 3D 打印件缺陷二分类任务上的迁移学习对比，
含 3 种优化器消融、LIME / Grad-CAM 可解释性与数据泄漏分析。

> 复现对象：M. M. Ahsan, S. Raman, Y. Liu, and Z. Siddique, "Defect analysis of 3D printed object using
> transfer learning approaches," *Expert Systems with Applications*, vol. 253, p. 124293, Nov. 2024,
> doi: [10.1016/j.eswa.2024.124293](https://doi.org/10.1016/j.eswa.2024.124293)

---

## 1. 项目速览

| 项 | 内容 |
|---|---|
| 组号 / 班级 | `<待填>` |
| 成员（8 人） | `<待填>` |
| 组长（兼统稿） | `<待填>` |
| 数据 | `AMdataset`（作者公开部分）D1 = 271 张原图，D2 = 3017 张增强图 → 详见 [docs/DATA.md](docs/DATA.md) |
| 实验矩阵 | 11 模型 × 3 优化器 × 2 数据集 = **66 runs**（论文原始矩阵 99 = 3 数据集，第三个数据集作者未公开） |
| 结果格式 | 每个 run 一份 `results/runs/<run_id>/metrics.json` → 详见 [docs/RESULTS.md](docs/RESULTS.md) |
| 口头报告 | 2026-11-19 或 11-26（周四） |
| 提交 | 2026-11-30（周一）17:00 前，Moodle + Turnitin |

---

## 2. 怎么跑（复现步骤）

> ⚠️ 当前是**仓库骨架**：`src/` 下是接口占位（`NotImplementedError`），由对应 owner 实现后逐条替换下面的命令。
> 目标状态：以下四条命令能在 Kaggle Notebook 或本地跑通。

```bash
# 0) 环境
pip install -r requirements.txt

# 1) 数据：按 docs/DATA.md 把 AMdataset 放到 data/raw/，再生成预处理与划分
python -m src.data_prepare --raw data/raw --out data/processed --size 224
python -m src.splits --seed 42

# 2) 训练单个 run（结果写入 results/runs/<run_id>/）
python -m src.train --config configs/vgg16.yaml

# 3) 汇总所有 run → leaderboard 与论文图表
python -m src.aggregate
```

单次 run 的产物固定为：`metrics.json` + `confusion_matrices.json` + `curves.png` + `train_log.csv`。

---

## 3. 目录结构与 owner

**一个文件只有一个 owner。** 改别人的文件先在群里说，由 owner 合入。

| 路径 | 内容 | Owner | 是否进 Git |
|---|---|---|---|
| `data/` | 原始图、预处理图、manifest | 地基组 | ❌（只有 `data/README.md` 进） |
| `splits/` | 训练/验证/测试划分 csv（**全组唯一一份**） | 地基组 | ✅ |
| `src/data_prepare.py`, `src/splits.py` | 数据管线 | 地基组 | ✅ |
| `src/models.py` | 11 个骨干 + 冻结逻辑 + 统一 head | 地基组 | ✅ |
| `src/train.py`, `src/evaluate.py`, `src/features.py` | 训练 / 评估 / 特征缓存 | 地基组 | ✅ |
| `src/aggregate.py` | leaderboard + 图表汇总 | 地基组 | ✅ |
| `configs/` | 每个模型一个 yaml（模型组自己维护自己那几个） | 各模型组 | ✅ |
| `analysis/` | 消融、泄漏实验、Grad-CAM、LIME、效率测量、demo | 模型组 C + 分析 | ✅ |
| `results/runs/` | 每个 run 的 metrics.json + 图 | 跑该 run 的人 | ✅（权重除外） |
| `results/leaderboard.csv` | 全组结果总表（脚本生成，禁止手改） | 地基组 | ✅ |
| `docs/` | 数据说明、结果格式、作者代码清单、工作区总览 | 组长 | ✅ |
| `.github/` | PR / Issue 模板、CI | 地基组 | ✅ |
| `tests/` | 契约测试（划分不泄漏、metrics schema 合法） | 地基组 | ✅ |

**不进 Git 的东西**：`data/`、模型权重（`*.pt`/`*.pth`）、`__pycache__`、虚拟环境、原始压缩包。
数据集只存在 **Kaggle 私有 Dataset** `am-defect-v1`，8 人挂载同一个路径。

---

## 4. 三份契约（开工前必须遵守）

1. **划分契约**：[docs/DATA.md](docs/DATA.md) — 只有地基组产出 `splits/*.csv`，固定 seed，冻结后任何人不得改。
2. **结果契约**：[docs/RESULTS.md](docs/RESULTS.md) — 字段照抄作者仓库 xlsx 的结构（10 个指标 + 3 个混淆矩阵），便于和论文逐格对比。
3. **所有权契约**：见上表。共享文件（`src/train.py`、`requirements.txt`、`metrics.json` 格式）只有地基组能改。

---

## 5. 协作流程（零基础版）

**一次性准备（15 分钟）**

1. 装 [GitHub Desktop](https://desktop.github.com/)（不用记命令）；
2. 接受仓库邀请 → Clone 到本地；
3. `git config --global user.name "你的名字"` / `user.email "你的邮箱"`。

**每天固定动作**

```text
Fetch origin → Pull
新建分支 feat/<名字>-<任务>   (例: feat/zhangsan-vgg16)
只改自己 owner 的文件
Commit（写清做了什么）→ Push → 网页开 Pull Request → 指派 1 人 Review → 合并
```

**红线**

- 不直接 push `main`；不 `git push -f`；不删别人的分支；
- 一个 PR 只做一件事，标题 `feat(modelA): add vgg19 D1 runs`；
- **没有 `metrics.json` 的结果不算结果**；不加说明的魔改超参不算实验。

---

## 6. 开源声明（评分 40% 直接相关）

本项目**核心实现由本组自己编写**。允许使用第三方库与预训练权重，但必须在此表登记。
详见 [NOTICE.md](NOTICE.md)。

| 组件 | 类型 | 用途 | 是否修改 |
|---|---|---|---|
| PyTorch / torchvision | 框架 + ImageNet 预训练权重 | 骨干网络与训练 | 否（仅调用） |
| scikit-learn | 指标库 | 10 个评估指标 | 否 |
| AM3d（作者仓库） | 99 份结果 xlsx + 3 个 Keras notebook | 数值对照 + 方法对齐（**未复制代码**，见 [docs/AUTHOR_CODE.md](docs/AUTHOR_CODE.md)） | 否 |
| `lime` / `grad-cam` | 可解释性库 | 若使用需登记 | 待定 |

> 作者仓库 `AM3d` 公开了 3 个 Keras notebook（约 4,781 行），但**没有发布任何训练好的权重**，代码也不能直接复现
> （路径写死、模型靠注释切换、无划分文件、无 seed 治理）。我们仅用它做方法对齐，全部代码在 PyTorch 中重写，
> 详见 [docs/AUTHOR_CODE.md](docs/AUTHOR_CODE.md)。

---

## 7. 时间线

| 时间 | 里程碑 |
|---|---|
| 10/25 | 8 人可 clone + 可挂载 Kaggle 数据 |
| 11/1 | `leaderboard.csv` 出现第一条结果（VGG16 / D1 / Adam） |
| 11/8 | 66 runs 齐全 |
| 11/15 | 扩展实验 + 报告初稿 ≥3000 词 + PPT 初稿 |
| 11/16–11/18 | 彩排 2 次（≤15:00） |
| 11/19 或 11/26 | 口头报告 |
| 11/29 | 定稿、格式检查、源码打包 |
| 11/30 15:00 | 提交（留 2 小时缓冲） |

---

## 8. 成员与分工

| 组 | 成员 | 模型 / 任务 | 报告章节 | 口头报告 |
|---|---|---|---|---|
| 地基组 | `<待填>` `<待填>` | 数据管线、框架、汇总、CI | 2 Project Description | 1:00–3:30 |
| 模型组 A | `<待填>` `<待填>` | VGG16, VGG19, ResNet50, ResNet101（24 runs） | 3.1 | 3:30–5:30 |
| 模型组 B | `<待填>` `<待填>` | InceptionResNetV2, Xception, MobileNetV2, EfficientNetB0（24 runs） | 3.2 + 效率对比 | 5:30–7:30 |
| 模型组 C + 分析 | `<待填>` `<待填>` | EfficientNetB7, EfficientNetV2M, NASNetLarge（18 runs）+ 泄漏/可解释/图表/PPT | 3.3 | 7:30–10:00 |
| 组长（兼） | `<待填>` | 建仓、统稿、提交、对外沟通 | 1 Introduction + 4 Conclusion | 0:00–1:00, 12:30–14:00 |

每人贡献由 [CONTRIBUTIONS.md](CONTRIBUTIONS.md) 自动沉淀（模块 → 作者 → PR），个人贡献报告从这里取材。
