# 交接文档：jellyfishhamburger → Kenneth

> **这份文件就是交接单。** 每个任务完成后由 jellyfishhamburger（她的 agent）填写对应小节，
> Kenneth 按「待 Kenneth 在 Kaggle 执行」跑 GPU 验证并回填结果，然后把状态改为 ✅。

## 维护规则

| 谁 | 改哪里 |
|---|---|
| **jellyfishhamburger** | 填写任务小节（完成内容 / 产物 / 自验结果 / 待执行命令）+ 更新 §总表 |
| **Kenneth** | 执行待办命令 → 回填「Kenneth 验证结果」→ 更新状态为 ✅ / ❌（附原因） |

PR 描述里**直接引用本文件对应小节**（例：`见 docs/HANDOFF.md#t1`）。

---

## 总表（一眼看进度）

| 任务 | 产物 | 本地自验 | 待 Kenneth 在 Kaggle 执行 | 状态 |
|---|---|---|---|---|
| T1 报告图表生成器 | `analysis/make_figures.py` + 5 张 `fig_*.png` + `INDEX.md` | ✅ 5/5 生成 | 无（纯本地） | 👀 待 review |
| T2 表格导出 | `analysis/export_tables.py` + `reports/tables/*.{md,tex}` | ✅ 448 个数值 0 处不一致 | 无（纯本地） | 👀 待 review |
| T3 效率测量 | `analysis/efficiency.py` + `results/efficiency.csv` + `fig_efficiency.png` | ✅ 11 行 | **补 GPU 时延列**（见下） | 👀 待 review |
| T4 LIME | `analysis/lime_explain.py` + 3 张 `lime_*.png` | ✅ 3 模型各 1 张，单图 5–9 秒 | 无（纯本地） | 👀 待 review |
| T5 Gradio demo | `analysis/demo_app.py` + `analysis/checkpoint.py` | ✅ 本地 200，10 个 checkpoint 可切换 | 无（本地跑最稳） | 👀 待 review |
| T6 工程收尾 | `reproduce.sh` + 新测试 + `docs/archive/` | ✅ 13 tests OK，dry-run 通过 | 无 | 👀 待 review |

状态取值：⬜ 未开始 / 🔄 进行中 / 🟡 待 Kenneth 验证 / ✅ 已完成 / ❌ 被退回

**分支**：全部任务在 `feat/leiman-tasks`（6 个 commit，一个任务一个 commit，方便按 commit review）。

---

## T1 · 报告图表生成器

- **完成时间**：2026-11-09
- **产物文件**：
  - `analysis/make_figures.py`
  - `results/figures/fig_accuracy_by_model.png`（11 模型 × 2 数据集，叠论文值）
  - `results/figures/fig_optimizer_heatmap.png`（D1 / D2 两栏，共用色阶）
  - `results/figures/fig_confusion_grid.png`（每数据集最好/最差的 run）
  - `results/figures/fig_delta_acc.png`（我们 − 论文，标出作者侧 F1=0 的三行）
  - `results/figures/fig_ranking.png`（按 D1 排序，右侧注明最优优化器）
  - `results/figures/INDEX.md`（图号 / 文件 / 英文 caption / 说明 / 生成命令）
- **本地自验**：

  ```bash
  $ python -m analysis.make_figures
  [figures] 1 -> fig_accuracy_by_model.png
  [figures] 2 -> fig_optimizer_heatmap.png
  [figures] 3 -> fig_confusion_grid.png
  [figures] 4 -> fig_delta_acc.png
  [figures] 5 -> fig_ranking.png
  [figures] 5 张图 + INDEX.md → results/figures

  $ ls -lh results/figures/fig_*.png
  95K  fig_accuracy_by_model.png
  97K  fig_confusion_grid.png
  143K fig_delta_acc.png
  159K fig_optimizer_heatmap.png
  88K  fig_ranking.png
  ```

- **待 Kenneth 在 Kaggle 执行**：无（纯本地任务）
- **Kenneth 验证结果**：
- **遗留问题**：
  - 图 1 纵轴从 0.40 起（否则 0.9–1.0 的差别看不见），caption 里已注明；如果报告要求从 0 起，改 `fig_accuracy` 的 `set_ylim` 即可。
  - 图 3 每个面板按各自最大值着色（D1 的计数 3 vs D2 的 19），跨面板不要直接比颜色深浅。
  - 只用了 Adam 跑与论文对齐；如果要用"三个优化器里最好的一次"对比，改 `pick()` 的调用即可。

---

## T2 · 报告表格导出

- **完成时间**：2026-11-09
- **产物文件**：`analysis/export_tables.py`、`reports/tables/table{1..5}_*.md`、`reports/tables/table{1..5}_*.tex`
  - Table 1 数据集体量（计数直接数 `splits/*.csv`）
  - Table 2 D1 全部 33 行、Table 3 D2 全部 33 行（9 个指标 + 参数量 + 时长）
  - Table 4 Δaccuracy（Adam 对齐，标出作者侧 F1=0 的行）
  - Table 5 效率（T3 完成后自动填；GPU 列写 N/A）
- **本地自验**：

  ```bash
  $ python -m analysis.export_tables --check
  [tables] wrote 5 markdown + 5 latex files → reports/tables
  [check] 核对 448 个数值，0 处不一致
  ```

  `--check` 会把生成的 Markdown 表格解析回来，逐格和 `results/leaderboard.csv` 比较；448 = 2 个数据集 × 33 行 × 7 个指标。
- **待 Kenneth 在 Kaggle 执行**：无
- **Kenneth 验证结果**：
- **遗留问题**：LaTeX 用 booktabs（需要 `\usepackage{booktabs}`）；表格宽 11 列，正文里建议 `\resizebox{\textwidth}{!}{...}` 或 `\small`。

---

## T3 · 效率测量

- **完成时间**：2026-11-09
- **产物文件**：`analysis/efficiency.py`、`results/efficiency.csv`（11 行）、`results/figures/fig_efficiency.png`
- **本地自验**：

  ```bash
  $ python -m analysis.efficiency
  [efficiency] 11 行 → results/efficiency.csv
  [efficiency] 图 → results/figures/fig_efficiency.png
  ```

  CSV 前三行（**空载重测**，见下方遗留问题）：

  ```text
  model,n_params,n_params_trainable,gflops,cpu_ms_per_image,gpu_ms_per_image
  mobilenet_v2,2305986,82114,0.599,22.9,N/A
  efficientnet_b0,4089662,82114,0.769,34.6,N/A
  resnet50,23639298,131266,8.175,82.8,N/A
  ```

  实测单模型耗时（权重已缓存、机器空载）：MobileNetV2 5 s，EfficientNet-B0 1 s，ResNet50 3 s，
  Xception 3 s，InceptionResNetV2 6 s，EfficientNetV2-M 6 s，EfficientNet-B7 7 s，ResNet101 5 s，
  VGG16 7 s，VGG19 8 s，NASNet-Large 13 s。

  ⚠️ 第一次测量时机器上同时在下模型/跑 LIME，均值被拖高（例如 EfficientNet-B0 51.3 → 34.6 ms，
  NASNet-Large 580.1 → 389.5 ms，EfficientNetV2-M 344.9 → 161.8 ms，后者从"比 VGG16 慢一倍"变成"和 VGG16 持平"）。
  CSV 与图已用空载数据覆盖。**结论只读相对排序，且必须在空载机器上测。**
- **待 Kenneth 在 Kaggle 执行**：

  ```bash
  # 目标：补 gpu_ms_per_image 列（本地无 GPU）
  # 预计 10 分钟，T4 GPU 会话
  python - <<'PY'
  import csv, time, torch
  from src.models import TransferModel
  rows = list(csv.DictReader(open("results/efficiency.csv")))
  dev = "cuda"
  x = torch.zeros(1, 3, 224, 224, device=dev)
  for r in rows:
      m = TransferModel(r["model"], pretrained=True, freeze="all_conv").to(dev).eval()
      with torch.no_grad():
          for _ in range(5): m(x)          # warmup
          torch.cuda.synchronize(); t0 = time.perf_counter()
          for _ in range(50): m(x)
          torch.cuda.synchronize()
      r["gpu_ms_per_image"] = round((time.perf_counter()-t0)/50*1000, 1)
      print(r["model"], r["gpu_ms_per_image"])
  with open("results/efficiency.csv", "w", newline="") as fh:
      w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
  PY
  # 预期：11 行全部有 GPU 时延；MobileNetV2 应在 3–6 ms，NASNet-Large 30–60 ms 量级
  ```

  跑完后请把结果贴回本节，并改 `analysis/export_tables.py` 的 Table 5 注释（目前写的是 "GPU pending"）。
- **Kenneth 验证结果**：
- **遗留问题**：CPU 数字与本机绑定（作者笔记本，无 GPU、未锁线程），**只能读相对排序**，不要写进论文当绝对性能。
  另外本次踩到一个坑：**测量时必须让机器空载**（并行下载/跑 LIME 会让均值虚高 40%–100%），我已按空载重测覆盖。

---

## T4 · LIME

- **完成时间**：2026-11-09
- **产物文件**：`analysis/lime_explain.py`、`results/figures/lime_{vgg16,resnet50,mobilenet_v2}__D1__adam__s42.png`
- **本地自验**：

  ```bash
  $ python -m analysis.lime_explain --models vgg16,resnet50,mobilenet_v2 --num-samples 100
  [lime] VGG16 · vgg16__D1__adam__s42 …
  [lime]   -> lime_vgg16__D1__adam__s42.png
  [lime] ResNet50 · resnet50__D1__adam__s42 …
  [lime] MobileNetV2 · mobilenet_v2__D1__adam__s42 …
  ```

  单图耗时 5–9 秒（`num_samples=100`，CPU），满足 T4「≤60 秒」的要求。图里上排原图、下排解释，
  红 = 支持当前判断，蓝 = 反对。
- **待 Kenneth 在 Kaggle 执行**：无（CPU 可完成；如需在 GPU 上复跑更快可自行决定）
- **遗留问题**：
  - 用的是 `samples/D1` 的 224×224 缩放图（仓库自带 20 张），不是原始 5184×3456；对解释结论无影响，但不要在报告里说这是原始分辨率。
  - 从图里能看出：VGG16 有时把背景/桌面当成证据（红色落在试样之外），这是可以写进 Discussion 的真实观察，不要删。

---

## T5 · Gradio demo

- **完成时间**：2026-11-09（demo 本体与看板在前一个 commit，本分支内为 `feat(demo)`）
- **产物文件**：
  - `analysis/demo_app.py`（上传/示例图 → 概率 + Grad-CAM + 10 个 checkpoint 对比表）
  - `analysis/checkpoint.py`（从 `model.pt` 复原模型；兼容"全局平均池化"存档头）
  - `tools/build_dashboard.py` + `tools/dashboard/template.html` + `docs/index.html`（附加交付：66 个 run 的离线看板）
  - `docs/DEMO.md`（现场脚本、断网降级、8 个 Q&A、90 秒录屏步骤）
- **本地自验**：

  ```bash
  $ python -m analysis.demo_app --warmup          # 5/5 ok（D1）；D2 的 5 个 checkpoint 同样可载入
  $ python -m analysis.demo_app --offline --no-browser --port 7860
  * Running on local URL:  http://127.0.0.1:7860   # HTTP 200
  ```

  用 `samples/D1/defect/IMG_9079.jpg` 自测：MobileNetV2 判 `defect`（p=0.937），
  勾"同时对比全部模型"返回 10 行（含 D2 checkpoint，判定与 D1 不同——正好可以讲"不同划分训练出的模型分歧"）。
- **待 Kenneth 在 Kaggle 执行**：无（按约定用本地 CPU 运行，pre 当天不依赖网络）
- **遗留问题**：
  - Gradio 6 把 `theme` 从 `Blocks()` 挪到 `launch()`，代码里已做兼容（`try/except TypeError`）。
  - Windows 上若开着系统代理（v2rayN 等），`127.0.0.1` 会被代理拦截导致启动 503；`demo_app.py` 已在 import 时写入 `NO_PROXY`。
  - 只有 10 个 checkpoint 能推理（5 个 D1 + 5 个 D2，均 Adam），其余 56 个 run 只有指标，界面上已写明。

---

## T6 · 工程收尾

- **完成时间**：2026-11-09
- **产物文件**：
  - `reproduce.sh`（缓存 → 划分 → 矩阵 → 汇总 → 图 → 表；支持 `--dry-run`、`--skip-train`、`--with-analysis`）
  - `tests/test_analysis_outputs.py`（3 个用例：表格逐格校验、看板内嵌数据校验、INDEX 图文件存在）
  - 删除死代码 `src/features.py`（全仓库无引用）
  - `docs/WORKSPACES.md`、`docs/LEADER-SETUP.md` → `docs/archive/`（附 `docs/archive/README.md` 说明为什么过时）
  - 清理旧分工表述：`docs/DATA.md`、`data/README.md`、`splits/README.md`
- **本地自验**：

  ```bash
  $ bash reproduce.sh --dry-run
  == 1/7 建 224 缓存
  [dry-run] python -m src.data_prepare --data-root <...> --out /kaggle/temp/cache --size 224
  == 2/7 生成划分（已有则跳过…）
  splits/D1.csv 与 D2.csv 已存在，跳过
  == 3/7 训练矩阵（66 run，GPU）
  == 4/7 汇总 leaderboard.csv
  == 5/7 报告图表（T1）
  == 6/7 报告表格（T2，含逐格校验）
  == 7/7 跳过 efficiency / lime（加 --with-analysis 才会跑）

  $ python -m compileall -q src analysis tools && echo COMPILE_OK
  COMPILE_OK

  $ python -m unittest discover -s tests
  Ran 13 tests in 1.206s
  OK
  [progress] 已完成 run: 66/66
  ```

- **待 Kenneth 在 Kaggle 执行**：无
- **遗留问题**：`reproduce.sh` 在 Kaggle 上跑要 `bash reproduce.sh`；Windows 本地跑请显式给 `--cache`（默认 `/kaggle/temp/cache` 不存在）。

---

## 提交 PR 前的检查清单（jellyfishhamburger 自查）

- [x] 产物文件已提交到仓库（`results/` 或 `reports/`），不是只留在本地
- [x] 自验命令**是本机真实跑过的**，输出是真贴的（不是编的）
- [x] 需要 GPU 的部分写成了**可直接复制的命令 + 预期输出 + 预计耗时**（见 T3）
- [x] 没有修改受限文件（`splits/*.csv`、`src/train.py`、`src/models.py`、`src/batch.py`、已有 `metrics.json`）
- [x] 新增的第三方库已登记到 `NOTICE.md`（`lime`、`timm`、`gradio`、`scikit-image`）
- [x] 已更新本文件的**总表**和对应任务小节
