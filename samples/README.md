# samples/ — 本地调试用样图（20 张）

| 项 | 说明 |
|---|---|
| 内容 | `D1/defect/*.jpg` 10 张、`D1/nondefect/*.jpg` 10 张 |
| 来源 | 从 D1 原始图（5184×3456）缩放到 **224×224**，JPEG q=85 |
| 大小 | 共约 73 KB |
| 用途 | **仅供本地跑通** LIME / Grad-CAM / Gradio demo 等 CPU 任务，以及给报告/PPT 做示例 |

## 使用规则

1. ❌ **不要**用这些图训练或评估（数量太少，且已从原图缩放，不能代表数据集）
2. ❌ **不要**把它们当作正式结果的数据来源；正式指标只认 `results/runs/*/metrics.json`
3. ✅ 所有报告/PPT 里的正式结果图，必须来自 `results/figures/`，不是这里的示例

## 代码里怎么用

```bash
# Grad-CAM / LIME / demo 支持 --data-root 指向任意目录；
# 本目录结构与正式数据集一致（data/D1/{defect,nondefect}），但没有 data/ 这一层，
# 所以用 --data-root samples/.. 不行，需写成如下形式（见 analysis 模块的 --samples-root 参数）
python -m analysis.lime_explain --samples-root samples --model vgg16
```

> 若分析模块尚未支持 `--samples-root`，可用 `--data-root` 指向完整数据集（需要数据访问权）。
