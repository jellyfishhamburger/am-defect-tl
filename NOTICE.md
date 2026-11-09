# 开源使用声明（Open-source attribution）

> 课程评分要求：*"If you need to use some open-source code, please clearly specify which parts of the
> code are open-source and which are developed by you."* (40%)
>
> **登记规则**：引入任何第三方代码（库、预训练权重、代码片段）时，在下面加一行。
> 只是 `pip install` 并使用公开 API 的库也要登记；复制/改写了别人的代码必须在"是否修改"列注明。

## 一、第三方组件（本组只调用，未修改）

| 组件 | 版本 | 用途 | 许可证 | 是否修改 | 使用位置 |
|---|---|---|---|---|---|
| PyTorch | ≥2.2（本地实测 2.15.0） | 深度学习框架 | BSD-3 | 否 | 全部训练/评估代码 |
| torchvision | ≥0.17 | **ImageNet 预训练权重** + transforms | BSD-3 | 否 | `src/models.py` |
| timm | ≥0.9（本地实测 1.0.31） | 11 个骨干 + ImageNet 预训练权重（torchvision 缺 NASNet / Xception / InceptionResNetV2） | Apache-2.0 | 否 | `src/models.py`, `analysis/{demo_app,checkpoint,lime_explain,efficiency}.py` |
| scikit-learn | ≥1.4 | 指标（MCC / Kappa / AUC / AP）+ 划分 | BSD-3 | 否 | `src/evaluate.py`, `src/splits.py` |
| NumPy / pandas | — | 数据处理 | BSD-3 | 否 | 全项目 |
| Pillow / OpenCV | — | 读图与预处理 | HPND / Apache-2.0 | 否 | `src/data_prepare.py` |
| Matplotlib / seaborn | — | 绘图 | PSF / BSD-3 | 否 | `src/evaluate.py`, `src/aggregate.py` |
| PyYAML | — | 配置读取 | MIT | 否 | `src/train.py` |
| `lime` | 0.2.0.1 | LIME 超像素解释（T4） | BSD-2 | 否（只调用 `LimeImageExplainer`） | `analysis/lime_explain.py` |
| scikit-image | `lime` 的间接依赖 | 图像分割（超像素） | BSD-3 | 否 | 仅经 `lime` 调用 |
| Gradio | ≥4（本地实测 6.31.0） | demo 网页界面（T5） | Apache-2.0 | 否 | `analysis/demo_app.py` |
| Grad-CAM | — | **未使用第三方库**：CAM 算法由本组在 `analysis/gradcam.py` 自行实现，只用 matplotlib 上色 | — | — | `analysis/gradcam.py` |

## 二、参考但不含代码的外部资源

| 资源 | 我们用它做了什么 | 是否含可复用代码 |
|---|---|---|
| [AM3d](https://github.com/mahsan2/AM3d)（论文作者仓库） | ① 解析其 **99 份 xlsx 结果表**做数值对照；② 阅读其 **3 个 Keras notebook（约 4,781 行）**做方法对齐 | ⚠️ **有代码**：`fault_3d_printing_experiment.ipynb`（主实验）、`basic_algorithm.ipynb`（草稿）、`base_algorithm.ipynb`（3 个 baseline）。我们**未复制**，仅参考其预处理/头结构/超参，在 PyTorch 中重写。详见 [docs/AUTHOR_CODE.md](docs/AUTHOR_CODE.md) |
| [AMdataset](https://github.com/mahsan2/AMdataset) | 数据来源 | ❌ 只有图片 |
| 论文 arXiv:2310.08645 | 方法细节（冻结主干 + AvgPool→Flatten→Dense→Dropout→Dense） | ❌ 只有文字与图 |
| `tools/fetch_paper_results.py` | 我们**自己写**的脚本，用于把作者 xlsx 抽成对照表 | ✅ 本组原创 |

## 三、本组原创模块（其余全部）

`src/` 与 `analysis/` 下所有 `.py`、`configs/*.yaml`、`tests/`、`tools/`、`docs/`、报告正文与图表脚本
均为本组自己编写，作者归属见 [CONTRIBUTIONS.md](CONTRIBUTIONS.md)。

## 四、每个源码文件的署名约定

```python
# Author: <姓名> | Task: <模块名> | PR: #<编号>
# Status: 本组原创实现。参考：<外部文档链接，若无则写 "无">
```

## 五、数据使用声明

AMdataset 仓库**未声明任何开源许可证**（仅要求引用论文）。本项目仅在课程作业范围内使用该数据，
不二次分发、不公开上传，报告与 README 均给出原作者论文引用。
