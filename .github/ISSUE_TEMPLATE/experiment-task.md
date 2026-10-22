---
name: 实验任务
about: 一个 run、一个模块或一个报告章节的认领与跟踪
title: "[modelA] vgg19 D1 三个优化器"
labels: ""
assignees: ""
---

## 任务
<!-- 说清楚交付物是什么。例：跑通 vgg19 在 D1 上的 adam/sgd/rmsprop 三个 run，共 3 份 metrics.json -->

## 对应契约
- 划分文件：`splits/D1.csv`
- 超参：`configs/vgg19.yaml`
- 结果格式：[docs/RESULTS.md](../docs/RESULTS.md)

## 验收标准
- [ ] `results/runs/vgg19__D1__{adam,sgd,rmsprop}__s42/metrics.json` 各一份
- [ ] `results/leaderboard.csv` 由 `python -m src.aggregate` 自动收录
- [ ] 与作者 xlsx 数值的差距写进 PR 描述（预期会有差异，说明原因即可）

## 截止
<!-- 例：11/5 -->
