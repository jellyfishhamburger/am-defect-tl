# 作者公开的代码里到底有什么（以及我们能用它做什么 / 不能做什么）

> 核实时间：2026-10-22，直接拉取 <https://github.com/mahsan2/AM3d> 的 notebook 并解析代码单元。

## 1. 结论先说

| 问题 | 答案 |
|---|---|
| 作者发布**训练好的权重**了吗？ | ❌ **没有**。全库没有任何 `.h5` / `.pt` / `.ckpt`。notebook 里 `model.save("vgg16xray.h5")` 是被注释掉的 |
| 作者发布**源代码**了吗？ | ✅ **有，3 个 Jupyter notebook，合计约 4,781 行代码** |
| 这些代码能直接跑吗？ | ❌ 不能。路径写死、模型靠注释切换、没有划分文件、没有 seed 控制、没有结果落盘格式 |
| 我们要用的"预训练权重"从哪来？ | ✅ ImageNet 公开权重，`torchvision.models.*(weights="IMAGENET1K_V1")` 一行下载。**这才是迁移学习的起点** |

**注意区分两种"权重"**：
- **起点权重**（ImageNet 预训练）= 公开、随便用、不构成学术问题；
- **产出权重**（在 AMdataset 上训好的 66 个分类头）= 作者没给，**必须我们自己训练出来**。

## 2. 三个 notebook 的内容清单

| 文件 | 规模 | 内容 |
|---|---|---|
| `fault_3d_printing_experiment.ipynb` | 47 cells / **886 行** | **主实验**：迁移学习训练 + 评估 + LIME + Grad-CAM |
| `basic_algorithm.ipynb` | 139 cells / **3,736 行** | 大型草稿本：11 个骨干逐个试、按优化器分目录跑（路径 `/data/ahsan/additive3d/result1/adam`）、大量中间实验与图 |
| `base_algorithm.ipynb` | 4 cells / **159 行** | 3 个 **baseline**（MLP / 简单 CNN / 双流 CNN）→ 对应 Studythree 里的 `baseline_one/two/three_*.xlsx` |

`fault_3d_printing_experiment.ipynb` 的结构：

| cell | 内容 |
|---|---|
| 0–1 | import 11 个骨干；`INIT_LR=1e-3, EPOCHS=50, BS=5` |
| 2–3 | 读图：`cv2.imread` → BGR2RGB → **`cv2.resize(img, (224,224))`** → `/255.0` |
| 4 | `LabelBinarizer` + `to_categorical`；**`train_test_split(test_size=0.2, random_state=42)`**；`ImageDataGenerator(rotation_range=15, fill_mode="nearest")` |
| 5 | `VGG16(weights="imagenet", include_top=False)` → `AveragePooling2D(4,4)` → `Flatten` → **`Dense(64, relu)`** → `Dropout(0.5)` → `Dense(2, softmax)` → 全部卷积层 `trainable=False` |
| 6 | `Adam()`（默认 lr=0.001）+ `binary_crossentropy`；`model.fit(trainAug.flow(...), validation_data=(testX, testY))` |
| 7–17 | classification report、混淆矩阵、sensitivity/specificity、ROC/AUC 图 |
| 18–26 | LIME（`skimage` 超像素 + `lime.lime_image`） |
| 27–34 | 手写 Grad-CAM（ResNet50 的 `conv5_block3_out` × 分类层权重） |
| 35–46 | **粘贴进来的开源 `pytorch-grad-cam` 库代码**（226 + 138 行）+ 出图 |

## 3. 从这份代码里，我们能拿到 5 条"高保真"细节（之前只能猜）

1. **预处理是直接拉伸，不是裁剪**：`cv2.resize(img, (224,224))`，**忽略长宽比**把 5184×3456 压成正方形。→ 我们 `docs/DATA.md` 原写的"短边缩放+中心裁剪"已按此更正。
2. **分类头 Dense 是 64 维**（不是 256）。这是论文正文没写、只能从代码里拿到的关键超参。
3. **训练时还有在线增强**：`rotation_range=15, fill_mode="nearest"`。也就是说 D2 已经是"离线增强过的数据集"，训练时又叠了一层在线增强。
4. **划分用 80/20，`random_state=42`**，并且 `validation_data=(testX, testY)` —— **这个版本里验证集就是测试集**（用测试集选模型，是方法学上的硬伤）。
5. **优化器就是 Keras 默认 `Adam()`**（lr 0.001），和论文说的网格搜索结论一致。

## 4. 但这 3 个 notebook 不能当工程代码用（5 个理由）

1. 路径写死在 `/content/drive/MyDrive/3d printing/Experiment2`（Colab + 个人网盘）；
2. 11 个模型靠**注释掉一行、打开另一行**来切换，没有配置化；
3. **没有划分文件**：每次都现场 `train_test_split`，别人无法复现同一份 train/test；
4. **没有 seed 治理**、没有结果落盘格式（指标靠人工看输出抄进 xlsx）；
5. 混入第三方库源码（gradcam）、无关实验和调试代码，无法审计。

> 反过来说：**这正是我们能做出"比作者更规范"的地方**——配置驱动、固定划分、统一 `metrics.json`、一键复现。
> 这既是 40% 实现分的来源，也是报告 Discussion 里"reproducibility gap"的实证。

## 5. 合规边界（评分 40% 直接相关，务必遵守）

| 做法 | 判定 |
|---|---|
| 读这份 Keras 代码，理解它的预处理/头结构/超参，然后在 **PyTorch** 里自己重写 | ✅ 允许，且**必须在 NOTICE.md 记录"参考了作者 notebook"** |
| 把 notebook 逐行翻译成 PyTorch，变量名和结构照搬 | ⚠️ 高风险：属于"改写他人代码"，必须显著标注且不应作为主体实现 |
| 把 notebook 直接跑一遍，拿它的输出当成我们的结果 | ❌ 不允许（那是作者的结果，不是我们的实验） |
| 使用 ImageNet 预训练权重、`lime`、`gradcam` 库 | ✅ 允许，登记到 NOTICE.md 即可（作者自己也用了 gradcam 库） |

## 6. 引用方式（写进报告 References 与 README）

> M. M. Ahsan, S. Raman, Y. Liu, and Z. Siddique, "Defect analysis of 3D printed object using transfer
> learning approaches," *Expert Systems with Applications*, vol. 253, p. 124293, 2024.
> Reference implementation: <https://github.com/mahsan2/AM3d> (Keras notebooks; used for method alignment only).
