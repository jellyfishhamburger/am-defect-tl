# samples/D1_test —— 演示用测试集样例（4 张）

| 项 | 说明 |
|---|---|
| 来源 | AMdataset 原始仓库 `dataset/{badcylinder,goodcylinder}/`，按 `splits/D1.csv` 里 **split = test** 的行抽取 |
| 标签 | `defect/`：IMG_9153、IMG_9213；`nondefect/`：IMG_8930、IMG_9033 |
| 预处理 | 与训练完全一致：`Image.draft → resize 224×224（忽略长宽比）→ JPEG q=92`（见 `src/data_prepare.py`） |
| 大小 | 每张 4–9 KB，共 4 张 |
| 为什么放这里 | 演示界面的"示例"按钮用这 4 张：**测试集图，训练时从未见过** |

## 使用与说明的边界

1. ✅ 可以说："这 4 张来自 D1 的 test split，训练时没见过。"
2. ✅ 可以说："这是从测试集里挑的代表样本。"
3. ❌ **不要**说"随机抽的"——它们是按模型表现和直观程度**挑**出来的；如果要声称随机，就用固定 seed 重新抽并接受可能判错。
4. ❌ 不要用这 4 张反推准确率。正式指标只看 `results/runs/*/metrics.json`（n=54）。

## 复核方式

```bash
# 用保存的 checkpoint 在完整测试集（54 张）上复跑
python tools/eval_test_split.py --models vgg16,mobilenet_v2 --dataset D1
```

实测（2026-11-09）：MobileNetV2 54/54（= metrics 的 1.000）；VGG16 52/54（metrics 写 1.000，差 2 张：IMG_9081、IMG_9082）。
