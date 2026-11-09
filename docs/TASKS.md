# jellyfishhamburger 的任务书（给她的 AI agent 读）

> 面向：**jellyfishhamburger** 使用的 AI agent。
> 项目背景 → [README.md](../README.md)；仓库约定与已知坑 → [AGENTS.md](../AGENTS.md)；
> 本文件只回答两件事：**jellyfishhamburger 要完成什么**、**如何与 Kenneth 协同**。

---

## 0. 两个人怎么分（先看，违反会被拒 PR）

| 人 | GitHub | 负责 |
|---|---|---|
| **Kenneth** | `Zheyi-D` | Kaggle 上的**训练类**实验（补 checkpoint、泄漏实验）、最终合并、数字核对、提交 |
| **jellyfishhamburger** | `jellyfishhamburger` | 所有**不需要 GPU 训练**的代码与分析：图表、表格、推理类分析、demo、工程收尾 |

**硬约束（指导 jellyfishhamburger 的 agent 遵守）：**

1. ❌ 不要修改 `splits/*.csv`、`src/train.py`、`src/models.py`、`src/batch.py`
2. ❌ 不要修改 `results/runs/*/metrics.json`（Kenneth 已跑完的 66 个正式结果）与 `docs/paper_results.csv`
3. ❌ 不要训练模型、不要跑完整实验矩阵（jellyfishhamburger 没有 GPU 配额，也不需要）
4. ❌ 不要提交数据、`__pycache__`、除 `model.pt` 外的权重
5. ✅ 所有数字必须来自仓库文件（`results/leaderboard.csv`、`docs/paper_results.csv`），**不许编造**
6. ✅ 分支 `feat/<topic>` → 提 PR → **不要直接 push `main`**
7. ⚠️ 若认为必须改第 1、2 条里的文件：**停下来**，在 PR 描述里写清理由，由 Kenneth 决定

**提交前自检：**

```bash
python -m compileall -q src analysis tools && python -m unittest discover -s tests -v
```

---

## 1. 上手顺序

1. `README.md` — 这个仓库是什么、66 个实验的结论
2. `AGENTS.md` — 产物契约、不变量、**已知坑 8 条**（先看，能省几小时）
3. 本文件 — jellyfishhamburger 的任务
4. `results/leaderboard.csv` — 全部实验数字（10 个指标 + `paper_acc` + `delta_acc`）

---

## 2. jellyfishhamburger 的本机环境（一次性，约 20 分钟）

```bash
# CPU 版 PyTorch（不需要显卡）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install timm scikit-learn gradio pandas matplotlib pillow pyyaml
```

验证：

```bash
python -c "import torch, timm, gradio; print('ok', torch.__version__)"
```

> 骨干权重首次会自动下载（100–500 MB/模型），之后有缓存。

---

## 3. 与 Kenneth 的协同方式

| 环节 | 规则 |
|---|---|
| 任务来源 | 本文件 §4 的任务卡 |
| 交付方式 | jellyfishhamburger 每个任务一个分支 + 一个 PR |
| PR 必须包含 | ① 目标 ② 产出文件路径 ③ 自验命令与输出 ④ **需要 Kenneth 在 Kaggle 验证的部分** |
| 需要 GPU 验证时 | **不要自己想办法跑**。在 PR 里写清命令与预期，标 `[需 Kenneth 在 Kaggle 验证]` |
| Kenneth 的职责 | review PR → 在 Kaggle 跑需要 GPU 的验证 → 回帖结果 → merge |
| 共享文件冲突 | 需要改 `src/train.py` 等受限文件时，先在 PR 描述里说明理由，由 Kenneth 决定 |
| 状态同步 | 只更新 §5 状态表 + PR 评论；群里只做提醒，**PR 是唯一权威记录** |
| **交接文档** | **每个任务完成后必须更新 [docs/HANDOFF.md](HANDOFF.md)**：写清产物路径、自验输出、以及「待 Kenneth 在 Kaggle 执行」的命令与预期。Kenneth 跑完后在 `HANDOFF.md` 回填结果 |

**PR 标题格式：** `feat(figures): add report figure generator (T1)`

---

## 4. 任务卡（按优先级）

### T1 · 报告图表生成器【最高优先】

| 项 | 内容 |
|---|---|
| 目标 | 生成报告第 3 节所需的汇总图（目前只有 per-run 图和 Grad-CAM，缺全局对比图） |
| 输入 | `results/leaderboard.csv`、`results/runs/*/metrics.json`、`results/runs/*/confusion_matrices.json` |
| 输出 | `analysis/make_figures.py` + `results/figures/fig_*.png` + `results/figures/INDEX.md`（图号、文件、一句话说明） |
| 需要的图 | ① 11 模型 × 2 数据集 准确率柱状图（含论文值对比）② 3 优化器消融热力图 ③ 混淆矩阵网格（选 4 个代表 run）④ 我们 vs 论文 Δacc 图 ⑤ 模型准确率排名 |
| 验收 | `python -m analysis.make_figures` 本地跑通；产出 ≥5 张 PNG；`INDEX.md` 每张图有一句话说明；**不修改任何 `metrics.json`** |
| 注意 | 中文字体缺失时用英文标签；图注英文，字号 ≥9pt，输出 200 dpi |

### T2 · 报告表格导出

| 项 | 内容 |
|---|---|
| 目标 | 把结果导出成报告可直接粘贴的表格（Markdown + LaTeX） |
| 输出 | `analysis/export_tables.py`、`reports/tables/*.md`、`reports/tables/*.tex` |
| 表格 | Table 1 数据集体量；Table 2 D1 全部 33 行；Table 3 D2 全部 33 行；Table 4 Δacc 汇总；Table 5 效率（等 T3 完成后填） |
| 验收 | 命令可重跑；表格数字与 `leaderboard.csv` 完全一致（可写个小校验） |

### T3 · 效率测量（CPU）

| 项 | 内容 |
|---|---|
| 目标 | 补上"轻量可部署"的论证数据 |
| 输出 | `analysis/efficiency.py`、`results/efficiency.csv`、`results/figures/fig_efficiency.png` |
| 列 | `model, n_params, n_params_trainable, gflops, cpu_ms_per_image, gpu_ms_per_image` |
| 做法 | 用 `src/models.py` 的 `TransferModel` 构建 11 个骨干，输入 `torch.zeros(1,3,224,224)`；CPU 时延取 warmup 5 + 20 次均值 |
| 验收 | CSV 11 行；`gpu_ms_per_image` 可留空（标 `N/A`），PR 里标 `[需 Kenneth 在 Kaggle 补 GPU 时延]` |

### T4 · LIME 可解释性（CPU）

| 项 | 内容 |
|---|---|
| 目标 | 对齐论文使用的 LIME，补一组解释图 |
| 输入 | `results/runs/*/model.pt`（5 个模型的分类头）、仓库内 `samples/`（20 张 224×224 样图） |
| 输出 | `analysis/lime_explain.py`、`results/figures/lime_<run_id>.png` |
| 验收 | 至少 3 个模型各出 1 张图；命令行可复现；CPU 上单图 ≤60 秒 |
| 注意 | 用 `lime.lime_image` 时把 `predict_fn` 包成接受 `(N,224,224,3)` 的 numpy 函数；在 `NOTICE.md` 登记 `lime` 库 |

### T5 · Gradio demo（CPU，本地跑）

| 项 | 内容 |
|---|---|
| 目标 | pre 现场演示用的网页：上传图 → 预测 + 置信度 + Grad-CAM |
| 输出 | `analysis/demo_app.py`；`python -m analysis.demo_app --model vgg16 --port 7860` 可本地启动 |
| 功能 | ① 上传/示例图选择 ② 显示 `defect/nondefect` + 置信度 ③ Grad-CAM 叠加图 ④ 下拉切换 2–3 个模型 |
| 依赖 | 复用 `analysis/gradcam.py` 里已有的 `load_trained` / `compute_cam` / `heat_overlay`（**不要重写**） |
| 验收 | 本地 `localhost:7860` 能打开并成功预测；用 `samples/` 自测；录 90 秒演示视频的步骤说明写进 PR |

### T6 · 工程收尾（低优先，可最后做）

| 项 | 内容 |
|---|---|
| 目标 | 仓库整洁 + 可复现 |
| 输出 | ① `reproduce.sh`（缓存→划分→矩阵→汇总→图）② 删除死代码 `src/features.py` ③ 补 2–3 个测试到 `tests/` ④ 清理旧分工表述：`docs/WORKSPACES.md`、`docs/LEADER-SETUP.md`（移入 `docs/archive/`）、`docs/DATA.md`、`data/README.md`、`splits/README.md` |
| 验收 | `bash reproduce.sh --dry-run` 能打印计划；测试全绿；文档里不再出现"地基组/模型组/8 人" |

---

## 5. 状态看板（jellyfishhamburger 维护）

| 任务 | 状态 | 分支 / PR | 产物 | 备注 |
|---|---|---|---|---|
| T1 图表生成器 | 👀 待 Kenneth review | feat/leiman-tasks | `analysis/make_figures.py`、5 张 `fig_*.png`、`INDEX.md` | 详见 HANDOFF T1 |
| T2 表格导出 | 👀 待 Kenneth review | feat/leiman-tasks | `analysis/export_tables.py`、`reports/tables/*.{md,tex}` | 448 个数值已逐格核对 |
| T3 效率测量 | 👀 待 Kenneth review | feat/leiman-tasks | `analysis/efficiency.py`、`results/efficiency.csv`、`fig_efficiency.png` | GPU 时延待补（HANDOFF 有可直接跑的脚本） |
| T4 LIME | 👀 待 Kenneth review | feat/leiman-tasks | `analysis/lime_explain.py`、3 张 `lime_*.png` | 单图 5–9 秒 |
| T5 Gradio demo | 👀 待 Kenneth review | feat/leiman-tasks | `analysis/demo_app.py`、`analysis/checkpoint.py`、`docs/DEMO.md` | 额外附了离线看板 `docs/index.html` |
| T6 工程收尾 | 👀 待 Kenneth review | feat/leiman-tasks | `reproduce.sh`、`tests/test_analysis_outputs.py`、`docs/archive/` | 13 tests OK |

状态取值：⬜ 未开始 / 🔄 进行中 / 👀 待 Kenneth review / ✅ 已合并

---

## 6. 当前仓库状态（避免重复劳动）

| 项 | 状态 |
|---|---|
| 66 个正式 run | ✅ Kenneth 已完成，数字在 `results/leaderboard.csv` |
| 5 个分类头 checkpoint | ✅ `results/runs/*/model.pt` |
| Grad-CAM / 预测示例图 10 张 | ✅ `results/figures/` |
| 论文对标表 297 行 | ✅ `docs/paper_results.csv` |
| 报告图表生成器 | ❌ 不存在（= T1） |
| 表格导出 | ❌ 不存在（= T2） |
| 效率 / LIME / demo | ❌ 空壳（= T3 / T4 / T5） |

---

## 7. 卡住时怎么办

1. 先查 [AGENTS.md §9 已知坑](../AGENTS.md)（timm 权重名、Kaggle 挂载、缓存路径等）
2. 仍不通 → 在 PR 里写：**现象 / 已尝试 / 判断 / 需要 Kenneth 做什么**
3. 不要为了让代码跑起来去改共享文件（`src/train.py`、`splits/`），那会破坏 66 个结果的可比性
