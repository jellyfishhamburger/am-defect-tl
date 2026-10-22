# 贡献记录（个人贡献报告的唯一素材来源）

> 每完成一个 PR，**本人**在这里加一行。写个人贡献报告（300–600 词）时直接按这张表展开：
> 负责什么 → 具体做了什么 → 遇到什么问题怎么解决 → 学到什么。
> 期末这份表也是"谁真的干了活"的客观证据。

## 一、模块 → 作者 → PR

| # | 模块 / 文件 | 内容 | 作者 | PR | 约行数 | 完成 |
|---|---|---|---|---|---|---|
| 1 | `src/data_prepare.py` | 下载校验 + 预处理 + manifest | | | 60 | ☐ |
| 2 | `src/splits.py` | 分层划分 + 分组划分 | | | 50 | ☐ |
| 3 | `src/models.py` | 11 骨干 + 冻结 + 统一 head | | | 120 | ☐ |
| 4 | `src/train.py` | 训练循环 / early stop / checkpoint | | | 120 | ☐ |
| 5 | `src/evaluate.py` | 10 指标 + 混淆矩阵 + 曲线 | | | 100 | ☐ |
| 6 | `src/features.py` | 冻结主干特征缓存 | | | 50 | ☐ |
| 7 | `src/aggregate.py` | leaderboard + 论文图 | | | 80 | ☐ |
| 8 | `configs/vgg16|vgg19|resnet50|resnet101` | 模型 A 的 4 个配置 + 24 run | | | — | ☐ |
| 9 | `configs/inception_resnet_v2|xception|mobilenet_v2|efficientnet_b0` | 模型 B 的 4 个配置 + 24 run | | | — | ☐ |
| 10 | `configs/efficientnet_b7|efficientnet_v2m|nasnet_large` | 模型 C 的 3 个配置 + 18 run | | | — | ☐ |
| 11 | `analysis/leakage.py` | D2′ 自建增强 + 泄漏对照实验（**自主贡献**） | | | 90 | ☐ |
| 12 | `analysis/gradcam.py` | Grad-CAM | | | 40 | ☐ |
| 13 | `analysis/lime_explain.py` | LIME（对齐论文） | | | 60 | ☐ |
| 14 | `analysis/efficiency.py` | 参数量 / FLOPs / 时延 | | | 50 | ☐ |
| 15 | `analysis/demo_app.py` | Gradio 现场 demo | | | 60 | ☐ |
| 16 | `tools/fetch_paper_results.py` | 从作者 99 份 xlsx 抽取对标数值 | | | 80 | ☐ |
| 17 | `tests/test_contracts.py` | 契约测试（防泄漏 / schema） | | | 120 | ☐ |
| 18 | 报告章节 | 见 README 第 8 节 | | — | — | ☐ |
| 19 | PPT / 彩排 | | | — | — | ☐ |

## 二、贡献日志（按日期追加，写贡献报告时直接搬）

格式：`日期 | 姓名 | 做了什么 | 产出（PR / 文件 / run_id）| 大约耗时`

```text
2026-10-25 | 张三 | 搭好仓库骨架、.gitignore、CI | PR #1 | 2h
2026-10-28 | 李四 | 拉取 AMdataset 并核对 271/3017 | PR #3 | 1.5h
```

## 三、个人贡献报告写作模板（每人 300–600 词，英文）

1. **My role**（60–80 词）：我在哪个组、负责哪个模块/模型、报告哪一节。
2. **What I did**（150–250 词）：具体到文件与 run_id，用数字（跑了几 run、多少 epoch、多少行代码）。
3. **Problems and how I solved them**（100–150 词）：至少一个真实技术问题（显存、类别不平衡、指标翻转、Git 冲突都算）。
4. **What I learned**（60–100 词）：与课程内容（迁移学习 / CNN / Python / 调参）挂钩。
