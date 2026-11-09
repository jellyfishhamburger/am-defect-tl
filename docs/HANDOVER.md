# am-defect-tl · 交付说明（给 Kenneth）

> 写于 2026-11-09 ｜ 对应分支 `feat/leiman-tasks` ｜ 对应 PR #8  
> 本文只回答六件事：我们做了什么、东西在哪、看板能看什么、demo 是干什么的、原始资料在哪、你还需要做什么。

---

## 0. 一句话

在 `Zheyi-D/am-defect-tl` 上完成了**不需要 GPU 的 6 项任务**（T1–T6）：报告图表、报告表格、效率测量、LIME 可解释性、现场 demo、工程收尾。
全部产物已提交，分支 `feat/leiman-tasks` → PR #8，CI 绿，可合并。

---

## 1. 东西在哪（先看这张表）

| 交付物 | 位置 | 说明 |
|---|---|---|
| **离线看板（网页）** | <https://jellyfishhamburger.github.io/am-defect-tl/> | 66 个 run 的可交互结果页，单文件、零外链，断网也能开 |
| 看板源文件 | `docs/index.html`（183 KB） | 生成物，**不要手改**；改模板后重跑脚本 |
| 看板生成脚本 | `tools/build_dashboard.py` + `tools/dashboard/template.html` | `python tools/build_dashboard.py` |
| **现场 demo（本机）** | `analysis/demo_app.py` | 双击 `start-demo.bat`，浏览器打开 127.0.0.1:7860 |
| demo 的模型复原 | `analysis/checkpoint.py` | 从 `model.pt` 重建分类头（兼容全局池化存档） |
| 报告图（5 张） | `results/figures/fig_*.png` + `INDEX.md` | 索引里有英文 caption，可直接抄进报告 |
| LIME 图（3 张） | `results/figures/lime_<run_id>.png` | VGG16 / ResNet50 / MobileNetV2 |
| 效率数据 | `results/efficiency.csv`（11 行）+ `fig_efficiency.png` | 参数量 / FLOPs / CPU 时延；GPU 列待你补 |
| 报告表（5 张 × 2 格式） | `reports/tables/table{1..5}_*.md` / `.tex` | Markdown + LaTeX（booktabs） |
| 一键复现 | `reproduce.sh` | `bash reproduce.sh --dry-run` 只打印计划 |
| 复核脚本 | `tools/eval_test_split.py` | 用 checkpoint 重跑 D1 测试集 54 张，核对 metrics |
| 任务交接 | `docs/HANDOFF.md` | 每个任务的产物 / 自验输出 / 待办 / 遗留问题 |
| 现场手册 | `docs/DEMO.md` | 2 分 30 秒演示脚本、断网降级、8 个 Q&A、录屏步骤 |

---

## 2. 离线看板能看什么

打开就是 6 个区块，全部鼠标可交互（点表格行 → 展开该 run 的详情）：

| 区块 | 内容 | 能回答的问题 |
|---|---|---|
| 01 Leaderboard | 66 行排名表，可按数据集/优化器筛选、点表头排序，含 ΔAcc（我们 − 论文） | 谁最好？和论文差多少？ |
| 02 模型 × 数据集 | 11 模型 × 2 数据集的准确率热力图（D1/D2 或三个优化器切换） | 哪个骨干在哪个数据集强？ |
| 03 优化器消融 | 每个模型三种优化器的准确率**最大差值**（横条 + 最差优化器） | 换优化器影响多大？ |
| 04 效率 | 训练时长（对数轴）× 测试准确率，点大小 = 可训练参数 | 哪个"又准又快"？ |
| 05 复现差距 | 我们 vs 论文的 Δaccuracy 哑铃图 | 差距集中在哪？ |
| 06 单 run 详情 | 测试集混淆矩阵 + 10 个指标 + 训练/验证曲线 | 这个 run 具体错在哪？ |

**看板里已经标出来的两个坑**（防止误读）：

- 作者报告的 EfficientNet-B0 / B7 / V2-M 三行 **F1 = 0**（全部预测成多数类，D1 accuracy 0.4909），所以 Δacc ≈ +0.49 **不是**"我们更强"；
- D1 测试集只有 **54 张**，错 1 张 = 1.8 个百分点，1.000 与 0.981 的差距没有统计意义。

---

## 3. 现场 demo 是看什么、做什么

**它演示的不是"产品准确率"，而是三件事：**

1. **权重真的能用** —— 不是只有表格数字，分类头能载入、能出概率；
2. **模型在看哪里** —— Grad-CAM 热力图指出做判断时最依赖的区域，这是报告 Discussion 里"它学的是缺陷还是背景"的证据；
3. **模型之间会分歧** —— 同一张图 5 个 D1 checkpoint 的判定可以不同，这正好呼应为什么要跑 66 个 run 的对比。

**操作流程**：上传照片（或点示例）→ 出 `defect / nondefect` 概率 → 右侧 Grad-CAM → 勾"同时对比全部模型"看 10 行对比表 → 下拉切换模型。

**必须知道的边界**（写报告时直接用）：

- 只有 **10 个 checkpoint**（5 个 D1 + 5 个 D2，均 Adam）能推理，其余 56 个 run 只有指标；
- **D2 的 checkpoint 在 128×128 增强图上训练，对 224 原图明显更差**（10 张缺陷样例只对 2–6 张），界面上已用 ⚠ 标注，演示时用 D1；
- 模型只在 D1 的实验条件（Ender 3 Pro + 固定光照/背景）下训练过，**换场景就是分布外**，手机随手拍大概率判 nondefect，这是局限不是 bug；
- 单张判断不代表整体（测试集 n=54）。

---

## 4. 原始资料在哪

| 资料 | 地址 |
|---|---|
| 复现对象论文 | Ahsan et al., *Expert Systems with Applications* 253:124293 (2024), doi:10.1016/j.eswa.2024.124293（arXiv:2310.08645） |
| 论文作者代码 | <https://github.com/mahsan2/AM3d>（3 个 Keras notebook + 99 份结果 xlsx，**未复制代码**） |
| 数据集 | <https://github.com/mahsan2/AMdataset>（D1 271 张、D2 3017 张） |
| 数据集 Kaggle 副本 | <https://www.kaggle.com/datasets/kennethdeng/am-defect-v1> |
| 复现仓库（你的） | <https://github.com/Zheyi-D/am-defect-tl>，分支 `feat/leiman-tasks` |
| 镜像（我的） | <https://github.com/jellyfishhamburger/am-defect-tl>（与上面同步，含 Pages） |
| 任务书 / 交接 | `docs/TASKS.md` · `docs/HANDOFF.md` |

---

## 5. 我们做了什么（T1–T6）

| 任务 | 产物 | 自验结果 |
|---|---|---|
| T1 报告图表 | `analysis/make_figures.py`；5 张 `fig_*.png` + `INDEX.md`（英文 caption） | 5/5 生成；数据全部读自 `leaderboard.csv`，无手写常数 |
| T2 表格导出 | `analysis/export_tables.py`；`reports/tables/*.{md,tex}` | `--check` 逐格核对 **448 个数值，0 处不一致** |
| T3 效率测量 | `analysis/efficiency.py`；`results/efficiency.csv`（11 行）+ `fig_efficiency.png` | 11/11 模型；CPU 时延 23–390 ms |
| T4 LIME | `analysis/lime_explain.py`；3 张 `lime_*.png`；`NOTICE.md` 已登记 | 单图 5–9 秒（要求 ≤60 s） |
| T5 Gradio demo | `analysis/demo_app.py`、`analysis/checkpoint.py`、`docs/DEMO.md`、`start-demo.bat` | 本地 HTTP 200；10 个 checkpoint 可切换；示例来自 test split |
| T6 工程收尾 | `reproduce.sh`、`tests/test_analysis_outputs.py`、`docs/archive/`、删除 `src/features.py` | 13 个测试全绿；`--dry-run` 正常 |

**关键结果（和论文对照）**

- D1 / D2 各 33 个 run，全部完成；准确率大多在 0.92–1.00；
- Adam 是 11 个骨干里对 D1 最稳的优化器；SGD 对 ResNet 系明显更差（差约 7 个百分点）；
- 作者侧三个 EfficientNet 全部塌成多数类（F1 = 0），我们的复现正常收敛 —— 报告里要写成"作者该组未收敛"，不是"我们领先 49 个点"。

---

## 6. 你需要做什么（三件必做 + 一件建议）

### ① Review + merge PR #8

- PR：<https://github.com/Zheyi-D/am-defect-tl/pull/8>
- 分支 `feat/leiman-tasks`，按任务拆了 commit，可以逐个看；
- CI 已通过，与 `main` 无冲突。

### ② 补 GPU 时延（唯一需要 Kaggle 的活，约 10 分钟）

`docs/HANDOFF.md` 的 **T3** 小节里有可直接复制的脚本：构建 11 个 `TransferModel`，warmup 5 次 + 计时 50 次，把 `gpu_ms_per_image` 写回 `results/efficiency.csv`。
跑完再执行一次：

```bash
python -m analysis.export_tables     # Table 5 会自动带上 GPU 列
```

### ③ 合并后更新 `AGENTS.md` §8 的状态快照

现在还写着 `analysis/{lime_explain,efficiency,demo_app}.py` 未实现、`figures: 14`。
合并后应为：四个模块全部可用、`figures: 22`。（`docs/TASKS.md` §6 我已经同步改好了。）

### ④（建议）核对 VGG16 的 checkpoint

我用保存的 `model.pt` 重跑了 D1 测试集 54 张：

| 模型 | 复跑结果 | `metrics.json` |
|---|---|---|
| MobileNetV2 · D1 | **54/54 = 1.000** | 1.000 ✅ |
| VGG16 · D1 | **52/54 = 0.963** | 1.000 ⚠️ |

VGG16 错的是 `IMG_9081`、`IMG_9082`（都判成 nondefect）。差 2 张不算大，但如果报告里写"VGG16 复现 1.000"，答辩被问到会尴尬 —— 建议确认保存的 checkpoint 是不是最终 epoch。

复核命令：

```bash
python tools/eval_test_split.py --models vgg16,mobilenet_v2 --dataset D1
```

---

## 7. 已知边界（写报告时别踩）

1. **D1 测试集只有 54 张** —— 1.000 与 0.981 的差距没有统计意义，别过度解读。
2. **作者第三个数据集未公开** —— 99 个实验单元里我们复现 66 个，其余 33 个只能引用作者报告值（表中标 † / 留空）。
3. **CPU 时延只能读相对排序** —— 数字与机器绑定，不要当绝对性能写进论文。
4. **图 1 纵轴从 0.40 起** —— 否则 0.9–1.0 的差别看不见；caption 已注明。
5. **demo 的示例图是挑选样本，不是随机抽样** —— 界面上和 `samples/D1_test/README.md` 都写明了。
6. **Grad-CAM 是相关性，不是因果** —— 只能说"这块区域对当前判定贡献大"，不能证明模型理解了缺陷。

---

## 8. 想自己跑一遍（可选）

```bash
git clone https://github.com/Zheyi-D/am-defect-tl.git && cd am-defect-tl
git checkout feat/leiman-tasks

python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

python -m analysis.make_figures                     # 5 张报告图
python -m analysis.export_tables --check            # 表格 + 逐格校验
python -m analysis.efficiency --replot              # 效率图（复用已有 CSV）
python -m analysis.lime_explain --models vgg16,resnet50,mobilenet_v2   # LIME
python -m analysis.demo_app --offline               # 现场 demo → 127.0.0.1:7860
python -m unittest discover -s tests                # 13 个测试
```

---

*有问题直接在 PR #8 里评论，或按 `docs/HANDOFF.md` 里对应的小节回填。*
