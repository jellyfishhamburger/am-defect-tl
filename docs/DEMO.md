# 现场演示手册（DEMO）

> 面向口头报告 10:00–12:30 那一段的 live demo。两份东西：
> ① `docs/index.html` 结果看板（静态、离线、可投屏）；② `analysis/demo_app.py` 推理界面（本机 Gradio）。

---

## 0. 一句话说清我们做了什么

11 个 ImageNet 预训练骨干，冻结卷积、只训练论文同款分类头，在 AMdataset 的 D1 / D2 上重跑 66 个实验单元，
并和作者报告的 99 个单元逐格对比。看板回答"谁好、差多少"，推理界面回答"它到底学到了什么"。

---

## 1. 文件清单（谁生成、能不能手改）

| 文件 | 作用 | 生成方式 | 手改？ |
|---|---|---|---|
| `docs/index.html` | 66 个 run 的可交互看板（单文件、零外链） | `python tools/build_dashboard.py` | ❌ 改了会被下次生成覆盖 |
| `tools/dashboard/template.html` | 看板模板（CSS/JS/结构） | 手写 | ✅ 改这里再重新生成 |
| `tools/build_dashboard.py` | 读 `results/` → 注入数据 → 生成页面 | — | ✅ |
| `analysis/demo_app.py` | Gradio 推理界面 + `--warmup` / `--selftest` | — | ✅ |
| `analysis/checkpoint.py` | 从 `model.pt` 复原模型（解决头部形状不一致） | — | ✅ |
| `demo_examples/` | 4 张 D1 测试集样例（2 defect / 2 nondefect，已压到 1400 px） | `splits/D1.csv` 里 `split=test` 的行 | ✅ 可替换 |

> `demo_examples/` 的图取自 AMdataset 公开仓库（University of Oklahoma SMIS Lab），仅用于课程演示，
> 按作者要求注明引用；**不要**把它们二次分发到其他仓库或数据集。

改完模板后必须重新跑一次生成命令，否则页面还是旧的：

```bash
python tools/build_dashboard.py
```

---

## 2. 演讲前 24 小时：准备（有网时做一次）

```bash
# 1) 建环境（只需一次）
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2) 预热 5 个骨干（会下载 ImageNet 权重，VGG16 一个就 528 MB，务必提前）
python -m analysis.demo_app --warmup

# 3) 自检：不开界面跑一次完整推理
python -m analysis.demo_app --selftest
```

`--warmup` 成功时会打印 5 行 `ok`。**看到 5 个 ok 才算准备好。**

---

## 3. 现场启动（两条命令）

```bash
# 看板：双击 docs/index.html 即可，或
start docs/index.html

# 推理界面
set HF_HUB_OFFLINE=1
python -m analysis.demo_app --offline --no-browser
# 浏览器打开 http://127.0.0.1:7860
```

要让学生用手机看：`python -m analysis.demo_app --host 0.0.0.0`，然后在同一 Wi-Fi 下访问
`http://<你的电脑 IP>:7860`（Windows 防火墙第一次会弹窗，选"允许专用网络"）。

---

## 4. 演示脚本（2 分 30 秒，别超时）

| 时间 | 动作 | 话术要点 |
|---|---|---|
| 0:00–0:30 | 打开看板，停在"模型 × 数据集" | "11 个骨干在两个数据集上，10 个都挤在 0.98 以上——这说明任务本身不难，难的是别把简单当成结论。" |
| 0:30–1:00 | 切到 03 优化器消融 | "SGD 对 ResNet 系明显更差，差 7 个点；其余模型几乎不受优化器影响。这是论文没做的消融。" |
| 1:00–1:30 | 切到 05 复现差距 | "作者报告里三个 EfficientNet 的 F1 是 0，全预测成多数类；我们复现的这一组是正常收敛的。所以那三行 +0.49 不是我们更强，是作者那组没跑起来。" |
| 1:30–2:10 | 打开推理界面，传一张 test 图 | 先说"这是 D1 测试集里的一张，不是训练图"，再点"判定"，指出 Grad-CAM 红区落在缺陷处还是背景。 |
| 2:10–2:30 | 勾"对比全部模型" | "同一张图，5 个骨干的判定一致/不一致——不一致说明这个样本本身处在决策边界上。" |

**手上要有的备用图**：从 Kaggle 数据集里挑 4 张（2 defect / 2 nondefect）存到本地文件夹，
其中 1 张故意选很模糊的，用来演示"模型什么时候会错"。

---

## 5. 断网预案（按顺序降级）

1. **权重已预热 + `HF_HUB_OFFLINE=1`** → 完全不需要网，正常演示。
2. 模型加载失败 → 界面自动回退到 `results/figures/predict_<run_id>.png`（预生成示例图），
   并在下方用红字说明原因；讲解改成"这是刚才那张图在这个模型下的输出"。
3. 连 Gradio 都起不来 → 直接放 `results/figures/` 里的 10 张成品图（5 模型 × 预测 / Grad-CAM）。
4. 全崩 → 播放提前录好的 60 秒屏幕录像（**演讲前一天一定录一份**，放在桌面，命名 `demo_fallback.mp4`）。

---

## 6. 可能被问的 8 个问题（照着答）

1. **为什么只有 5 个模型能演示？**
   只有这 5 个 run 保存了分类头权重（130–516 KB），其余 61 个 run 只留指标。课程算力下这是刻意的取舍：
   完整矩阵保证可对比，权重只给 demo 留样本。

2. **为什么用 timm 而不是作者用的 Keras？**
   torchvision 缺 NASNet / Xception / InceptionResNetV2 / EfficientNetV2M，timm 一个接口全覆盖且权重可复现。
   代价是 PyTorch 与 Keras 的预处理/初始化差异会成为复现差距的一部分，这在 05 里单独讨论了。

3. **作者报告的 accuracy 0.49、F1 = 0 是怎么回事？**
   说明该组实验全部预测成 nondefect（多数类）。D1 里 nondefect 占 135/271 ≈ 0.498，0.4909 正好是这个量级。
   这是"没收敛"，不是"模型差 49 个点"。

4. **为什么 D1 上准确率能到 1.000？**
   测试集只有 54 张，错 1 张就掉 1.8 个点。1.000 与 0.981 的差别在这个样本量下没有统计意义。
   我们没有做置信区间是因为 n 太小，区间会宽到没有信息量（这点可以在 Discussion 里写）。

5. **D2 的 0.99 可信吗？**
   不完全可信。D2 是作者对 D1 做的增强，文件名把溯源抹掉了，作者按随机 70/10/20 划分，
   同一张原图的增强副本可能同时进 train 和 test —— 这是数据泄漏。我们用 D2′ 分组划分做了对照实验来量化它。

6. **Grad-CAM 的红区能证明模型看的是缺陷吗？**
   只能说明"这块区域对当前 logit 贡献大"，不构成因果。我们的判据是：红区稳定落在试样边缘/层纹上，
   而不是背景或反光板。5 个模型的 Grad-CAM 图都放在 `results/figures/gradcam_*.png`。

7. **为什么冻结全部卷积层？**
   复现论文设定：只训练新分类头（约 3–26 万参数），这也是 04 效率图里"x 轴是分类头参数量"的原因。
   解冻微调的对照组没有做，是明确的范围外（Scope）声明。

8. **这套东西能上线吗？**
   不能，也不打算。D1 是实验室条件下的圆柱试样、单一打印机（Ender 3 Pro）、单一相机（EOS 60D），
   换设备/换材料后分布就变了。报告里把它定位为"方法复现"，不是产品。

---

## 7. 发布看板（可选，5 分钟）

`docs/index.html` 推上去后，在 GitHub 仓库 **Settings → Pages → Source: Deploy from a branch → main / docs**，
得到 `https://zheyi-d.github.io/am-defect-tl/`。页面无外链依赖，手机打开也正常。

---

## 8. 排错清单

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: timm` | 环境没装 timm（原 requirements 漏了，已补） | `pip install timm` |
| `size mismatch for head.fc1.weight` | 用旧代码直接 `load_state_dict` | 用 `analysis.checkpoint.load_model`（已内置兼容） |
| 卡在 "Loading weights" | 现场没网，timm 在下载 | `--offline` + 提前 `--warmup` |
| 下载权重超时 / 连不上 huggingface.co | 国内网络 | `python -m analysis.demo_app --warmup --mirror`（走 hf-mirror.com） |
| 端口被占用 | 7860 被别的进程占 | `--port 7861` |
| 手机打不开 | 只监听了 127.0.0.1 / 防火墙拦截 | `--host 0.0.0.0` + 放行 7860 |
| 看板打开是空白 | 用了旧缓存或文件被截断 | 重新 `python tools/build_dashboard.py`，确认约 180 KB |
| 看板数字和表格对不上 | 有人手改了 `leaderboard.csv` | 结果只认 `results/runs/*/metrics.json`，重跑 `src.aggregate` |
