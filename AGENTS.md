# AGENTS.md — 给 AI coding agent 的仓库说明

> 本文件面向自动化 agent。人类读者请看 [README.md](README.md)。
> 约定：只写可执行的事实与约束，不写愿景。

```yaml
# agent-context v1
repo: am-defect-tl
kind: research-reproduction
task: binary image classification (defect vs nondefect, positive = defect)
paper_doi: 10.1016/j.eswa.2024.124293
language: python
frameworks: [pytorch, timm, scikit-learn, matplotlib]
entrypoint: "python -m src.batch"
compute: "Kaggle Notebook, CUDA T4; no local GPU needed"
data_source: "https://github.com/mahsan2/AMdataset"
kaggle_dataset: "kennethdeng/am-defect-v1"
data_in_repo: false
primary_outputs:
  - "results/runs/<run_id>/metrics.json"
  - "results/leaderboard.csv"
  - "results/figures/*.png"
status:
  runs_completed: 66
  runs_expected: 66
  checkpoints: 5        # results/runs/*/model.pt (head only)
  figures: 11
  unimplemented:
    - analysis/leakage.py
    - analysis/lime_explain.py
    - analysis/efficiency.py
    - analysis/demo_app.py
```

---

## 1. 任务边界

复现论文的对比矩阵：**11 个冻结骨干 × 3 优化器 × 2 数据集 = 66 个 run**。
每个 run = 训练一次 + 在 test 上算 10 个指标。不做检测/分割，只做整图二分类。
论文原始矩阵是 99 格（3 个数据集），第三个数据集未公开；本仓库缺的 33 格在 `results/leaderboard.csv`
里以作者报告值出现，`delta_acc` 为空。

## 2. 环境事实

| 项 | 值 |
|---|---|
| 训练环境 | Kaggle Notebook，GPU `Tesla T4`，Python 3.13，torch 2.11+cu128，timm（预装） |
| 数据挂载 | `/kaggle/input/datasets/<owner>/<slug>`（**也出现过** `/kaggle/input/<slug>`，用 `src.locate_data` 自动识别） |
| 可写目录 | `/kaggle/working`（产物，会打包进 kernel output）、`/kaggle/temp`（缓存，**不进 output**） |
| 并发限制 | 免费账号最多 **2 个 GPU 会话**，第 3 个 push 会报 `Maximum batch GPU session count of 2 reached` |
| 单会话上限 | 9 小时；超时结果可能丢失，所以结果直接写 `/kaggle/working/results` |
| 本地（开发机） | 无 GPU、无 torch；只用于写代码 / 生成 splits / 跑 `aggregate`（纯 stdlib） |

## 3. 规范命令

| 目的 | 命令 |
|---|---|
| 定位数据 | `DATA=$(python -m src.locate_data --wait 300 --diagnose)` |
| 建图片缓存（**必做**） | `python -m src.data_prepare --data-root "$DATA" --out /kaggle/temp/cache --size 224` |
| 生成划分 | `python -m src.splits --data-root "$DATA" --out splits --seed 42` |
| 单 run | `python -m src.train --config configs/_template.yaml --model vgg16 --data-root "$DATA" --cache-root /kaggle/temp/cache --dataset D1 --optimizer adam --save-weights` |
| 矩阵批跑 | `python -m src.batch --data-root "$DATA" --out results --cache-root /kaggle/temp/cache` |
| 汇总 | `python -m src.aggregate`（`--check` 只校验完整性） |
| 解释图 | `python -m analysis.gradcam --data-root "$DATA" --cache-root /kaggle/temp/cache --dataset D1` |
| 推 Kaggle | `python tools/build_kernel.py --slug <slug> --cmd "<command>"` |
| 契约测试 | `python -m unittest discover -s tests -v` |

`tools/build_kernel.py` 会把 `src/*.py analysis/*.py configs/*.yaml splits/*.csv requirements.txt`
打包成 zip → base64 内嵌进 `main.py` → 推送执行。**不需要 token、不需要 git clone**。

## 4. 代码地图

| 文件 | 职责 | 备注 |
|---|---|---|
| `src/locate_data.py` | 在 `/kaggle/input` 下搜索含 `data/D1/defect` 的目录 | 支持挂载延迟重试 |
| `src/data_prepare.py` | 大图 → 224 缓存（`Image.draft` + resize，JPEG 重编码） | 幂等，已存在则跳过 |
| `src/splits.py` | 扫描目录 → 分层 70/10/20 → `splits/{D1,D2}.csv` | 路径必须 `as_posix()` |
| `src/dataset.py` | 按 csv 读图；`cache_root` 存在时读缓存 | 归一化用 ImageNet mean/std |
| `src/models.py` | `BACKBONES`（值**是候选名列表**）+ `PaperHead` + `TransferModel` | 骨干冻结时 forward 包在 `no_grad` |
| `src/train.py` | 单 run：训练/评估/落盘/画图/存头权重 | `--smoke` 抽样；`--force` 覆盖 |
| `src/batch.py` | 66 矩阵；失败不中断；`DATASET_HP` 定义每数据集超参 | 结束有失败则 `SystemExit(1)` |
| `src/aggregate.py` | 合并 `metrics.json` → `leaderboard.csv`，与 `docs/paper_results.csv` 对标 | 跳过 `smoke=True` |
| `src/evaluate.py` | 10 指标 + 2×2 混淆矩阵 + 绘图 | **positive = defect** |
| `analysis/gradcam.py` | Grad-CAM + 预测示例图；从 `model.pt` 复原模型 | 需要 `model.pt` |

## 5. 产物契约（不要改字段名）

- `run_id` = `<model>__<dataset>__<optimizer>__s<seed>`，目录名必须与 `metrics.json.run_id` 一致
- `metrics.json` 必含：`run_id, git_commit, model, dataset, optimizer, seed, input_size, n_train, n_val, n_test,
  metrics{train,val,test}{10 个指标}, confusion{train,val,test}, train_time_s`
- 10 个指标固定为：`accuracy, precision, recall, f1, mcc, kappa, auc, map, sensitivity, specificity`（0–1）
- `model.pt` **只存分类头**：`{model, num_classes, freeze, dense_units, dropout, input_size, feat_shape, head_state, run_id}`

## 6. 不变量（违反会导致结果不可比）

1. **`splits/*.csv` 冻结**：改了就必须重跑全部 66 个 run。生成后不要重新划分。
2. **positive = defect**：混淆矩阵 [[TN,FP],[FN,TP]]，行=真实、列=预测。定义反了 4 个指标会互换。
3. **数据不进 Git**：`data/`、原始图、除 `model.pt` 外的权重都在 `.gitignore` 里。
4. **缓存放 `/kaggle/temp`**，不要放 `/kaggle/working`（否则 3,000+ 张缓存图会被打进 kernel output，下载极慢）。
5. **不要手改 `results/leaderboard.csv`**，它是 `src.aggregate` 的产物。
6. `metrics.json` 存在即视为已完成，`batch` 会跳过；要重跑必须 `--force`。

## 7. 变更流程

- 分支命名 `feat/<topic>` / `fix/<topic>`；一个 PR 只做一件事
- 提交前：`python -m compileall -q src analysis tools && python -m unittest discover -s tests -v`
- CI（`.github/workflows/ci.yml`）只做「编译 + 契约测试」，**不装 torch**，所以不要在新代码里把 torch 变成 import 时强依赖
- 注意：免费版私有仓**无法开启分支保护**，`main` 可被直接 push —— 请自觉走 PR

## 8. 当前状态快照（2026-11-09）

| 项 | 状态 |
|---|---|
| 66 个 run | ✅ 全部完成，指标在 `results/leaderboard.csv` |
| checkpoint | ✅ 5 个（vgg16 / mobilenet_v2 / xception / resnet50 / inception_resnet_v2，均 D1+Adam） |
| Grad-CAM / 预测图 | ✅ 10 张，在 `results/figures/` |
| 论文对标表 | ✅ `docs/paper_results.csv`（297 行，作者报告值） |
| 未实现 | `analysis/{leakage,lime_explain,efficiency,demo_app}.py`（均为 `NotImplementedError`） |
| 报告 / PPT | ❌ 不在本仓库（由非代码成员在外部文档完成） |

## 9. 已知坑（踩过的）

| 现象 | 原因 | 对策 |
|---|---|---|
| `No pretrained weights exist for efficientnet_b7` | timm 移除了该旧名字的默认权重 | `BACKBONES` 的值是候选名列表，按序回退（如 `tf_efficientnet_b7`） |
| `FileNotFoundError: .../data/D1/defect` | 数据集挂载延迟，或挂载路径布局不同 | 用 `src/locate_data --wait`，不要硬编码路径 |
| 每 epoch 约 80 秒 | 每个 epoch 重新解码 5184×3456 的 JPEG | 必须用 `data_prepare` 缓存；`Image.draft` 也加速解码 |
| kernel 输出下载卡住（数千小文件） | 缓存写进了 `/kaggle/working` | 缓存路径改 `/kaggle/temp` |
| 第 3 个 kernel push 失败 | 免费账号 GPU 会话上限 2 | 排队等槽位，或串行 |
| kernel 报 `SyntaxError` | 把带引号的命令直接拼进 runner 的 Python 字符串 | 已改为 `json.dumps(cmd)` 注入 + 本地 `compile()` 自检 |
| Windows 上生成的 csv 在 Linux 读不到 | 路径用了 `\` | `splits.py` 统一 `as_posix()`；`dataset.py` 再做一次防御替换 |
| `batch` 结尾返回 1、kernel 显示 ERROR | 有 run 失败时 `SystemExit(1)`（但成功的 run 已落盘） | 看 `results/batch_failures.json`，补跑失败项 |

## 10. 扩展指南

**加一个骨干**：在 `src/models.py` 的 `BACKBONES` 加 `"new_name": ["timm_name", "备用候选"]`；
在 `src/batch.py` 的 `MODELS` 追加；跑 `--models new_name` 验证一个 run。

**加一个数据集**：把图放到 `<data-root>/data/D3/{defect,nondefect}/`；在 `src/splits.py` 的 `DATASETS` 加 `"D3"`；
在 `src/batch.py` 的 `DATASETS` 与 `DATASET_HP` 加对应超参；重跑 splits 与 matrix。

**改分类头**：改 `PaperHead` 与 `configs/_template.yaml` 的 `head.*`；注意 `metrics.json.head` 字符串要同步，
且**不要**在对比实验中悄悄改（会导致与已有 66 个 run 不可比）。

**加一个分析实验**：在 `analysis/` 新建模块，输出到 `results/`，并把产物格式登记到 `docs/RESULTS.md`。

## 11. 贡献边界

- 代码与实验产物（`src/`、`analysis/`、`results/`）由 2 名核心成员维护；其他人以文档、演示、报告形式贡献
- 文档类改动（`docs/`、README）可以直接提 PR，不需要跑实验
- 提交信息用 `feat(scope): ...` / `fix(scope): ...` / `docs: ...`

## 12. jellyfishhamburger 的任务入口

如果你的使用者是 **jellyfishhamburger**（GitHub: `jellyfishhamburger`，负责**不需要 GPU 训练**的分析与前端工作）：

> **先读 [docs/TASKS.md](docs/TASKS.md)** —— 她的任务卡（T1–T6）、验收标准、与 Kenneth 的协同规则。
> **交接文档是 [docs/HANDOFF.md](docs/HANDOFF.md)** —— 每个任务完成后由她更新，Kenneth 按其中的「待 Kenneth 在 Kaggle 执行」清单跑 GPU 验证。

三条最容易踩的边界（详见 TASKS.md §0）：

1. 不要改 `splits/*.csv`、`src/train.py`、`src/models.py`、`src/batch.py`、已有的 66 个 `metrics.json`
2. 不要训练模型、不要跑完整实验矩阵
3. 需要 GPU 才能验证的东西 → 写进 `docs/HANDOFF.md` 的待办清单，标 `[需 Kenneth 在 Kaggle 验证]`，不要绕过

维护者：**Kenneth**（GitHub: `Zheyi-D`）负责 Kaggle 训练类实验、最终合并与数字核对。
