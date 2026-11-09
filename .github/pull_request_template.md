# Pull Request

## 这个 PR 做了什么
<!-- 一句话。例：新增 VGG19 在 D1 上的 3 个优化器结果 -->

- 

## 类型
- [ ] `feat` 新功能 / 新模型结果
- [ ] `fix` 修 bug
- [ ] `data` 数据 / 划分变更（**需全组重跑，谨慎**）
- [ ] `docs` 文档
- [ ] `refactor` 重构（不改变结果）

## 自查清单（勾不完不许提交）

- [ ] 我只改了本 PR 声明范围内的文件（目录/接口约定见 [AGENTS.md](../AGENTS.md)）
- [ ] **没有**提交 `data/`、模型权重（除 `model.pt` 外的 `*.pt`）、`__pycache__`
- [ ] 新跑的结果有 `metrics.json` + `confusion_matrices.json` + `train_log.csv`，`run_id` 与目录名一致
- [ ] 我确认了 **positive = defect**，指标没有翻转
- [ ] 若引入了新的第三方库/代码片段，已在 [NOTICE.md](../NOTICE.md) 登记
- [ ] `python -m unittest discover -s tests` 通过（或说明为何 skip）

## 影响面
<!-- 是否会影响别人的实验？是否需要谁重跑？ -->

- 

## 截图 / 结果（有图请贴）
<!-- 曲线、混淆矩阵、leaderboard 片段 -->
