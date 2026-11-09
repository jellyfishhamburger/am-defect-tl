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
| T1 报告图表生成器 | — | — | 无（纯本地） | ⬜ 未开始 |
| T2 表格导出 | — | — | 无（纯本地） | ⬜ 未开始 |
| T3 效率测量 | — | — | 补 GPU 时延列 | ⬜ 未开始 |
| T4 LIME | — | — | 无（纯本地） | ⬜ 未开始 |
| T5 Gradio demo | — | — | 无（本地跑最稳） | ⬜ 未开始 |
| T6 工程收尾 | — | — | 无 | ⬜ 未开始 |

状态取值：⬜ 未开始 / 🔄 进行中 / 🟡 待 Kenneth 验证 / ✅ 已完成 / ❌ 被退回

---

## T1 · 报告图表生成器

- **完成时间**：
- **产物文件**：
- **本地自验**（贴命令 + 真实输出）：

  ```bash
  # 命令
  ```
  ```text
  # 输出
  ```

- **待 Kenneth 在 Kaggle 执行**：无（纯本地任务）
- **Kenneth 验证结果**：
- **遗留问题**：

---

## T2 · 报告表格导出

- **完成时间**：
- **产物文件**：
- **本地自验**：
  ```bash
  ```
- **待 Kenneth 在 Kaggle 执行**：无
- **Kenneth 验证结果**：
- **遗留问题**：

---

## T3 · 效率测量

- **完成时间**：
- **产物文件**：
- **本地自验**（应包含 CSV 的 11 行前 3 行）：
  ```bash
  ```
- **待 Kenneth 在 Kaggle 执行**：
  ```bash
  # 目标：补 gpu_ms_per_image 列（本地无法测）
  # 预期：11 个模型的 GPU 单张推理时延，写回 results/efficiency.csv 的 gpu_ms_per_image 列
  # 预计耗时：约 10 分钟
  ```
- **Kenneth 验证结果**：
- **遗留问题**：

---

## T4 · LIME

- **完成时间**：
- **产物文件**：
- **本地自验**（单图耗时也要记）：
  ```bash
  ```
- **待 Kenneth 在 Kaggle 执行**：无（CPU 可完成；如需在 GPU 上复跑更快可自行决定）
- **Kenneth 验证结果**：
- **遗留问题**：

---

## T5 · Gradio demo

- **完成时间**：
- **产物文件**：
- **本地自验**（启动命令 + 访问地址 + 用了哪几张样图）：
  ```bash
  ```
- **待 Kenneth 在 Kaggle 执行**：无（按约定用本地 CPU 运行，pre 当天不依赖网络）
- **Kenneth 验证结果**：
- **遗留问题**：

---

## T6 · 工程收尾

- **完成时间**：
- **产物文件**：
- **本地自验**：
  ```bash
  bash reproduce.sh --dry-run
  python -m unittest discover -s tests -v
  ```
- **待 Kenneth 在 Kaggle 执行**：无
- **Kenneth 验证结果**：
- **遗留问题**：

---

## 提交 PR 前的检查清单（jellyfishhamburger 自查）

- [ ] 产物文件已提交到仓库（`results/` 或 `reports/`），不是只留在本地
- [ ] 自验命令**是本机真实跑过的**，输出是真贴的（不是编的）
- [ ] 需要 GPU 的部分写成了**可直接复制的命令 + 预期输出 + 预计耗时**
- [ ] 没有修改受限文件（`splits/*.csv`、`src/train.py`、`src/models.py`、`src/batch.py`、已有 `metrics.json`）
- [ ] 新增的第三方库已登记到 `NOTICE.md`
- [ ] 已更新本文件的**总表**和对应任务小节
