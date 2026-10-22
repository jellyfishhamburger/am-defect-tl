# 开源使用声明（Open-source attribution）

> 课程评分要求：*"If you need to use some open-source code, please clearly specify which parts of the
> code are open-source and which are developed by you."* (40%)
>
> **登记规则**：引入任何第三方代码（库、预训练权重、代码片段）时，在下面加一行。
> 只是 `pip install` 并使用公开 API 的库也要登记；复制/改写了别人的代码必须在"是否修改"列注明。

## 一、第三方组件（本组只调用，未修改）

| 组件 | 版本 | 用途 | 许可证 | 是否修改 | 使用位置 |
|---|---|---|---|---|---|
| PyTorch | ≥2.2 | 深度学习框架 | BSD-3 | 否 | 全部训练/评估代码 |
| torchvision | ≥0.17 | **ImageNet 预训练权重** + transforms | BSD-3 | 否 | `src/models.py` |
| scikit-learn | ≥1.4 | 指标（MCC / Kappa / AUC / AP）+ 划分 | BSD-3 | 否 | `src/evaluate.py`, `src/splits.py` |
| NumPy / pandas | — | 数据处理 | BSD-3 | 否 | 全项目 |
| Pillow / OpenCV | — | 读图与预处理 | HPND / Apache-2.0 | 否 | `src/data_prepare.py` |
| Matplotlib / seaborn | — | 绘图 | PSF / BSD-3 | 否 | `src/evaluate.py`, `src/aggregate.py` |
| PyYAML | — | 配置读取 | MIT | 否 | `src/train.py` |
| `lime` | 待定 | LIME 可解释性 | BSD-2 | 待定 | `analysis/lime_explain.py` |
| Grad-CAM | 待定 | 若不自写则用库 | MIT | 待定 | `analysis/gradcam.py` |
| Gradio | 待定 | demo 界面 | Apache-2.0 | 待定 | `analysis/demo_app.py` |

## 二、参考但不含代码的外部资源

| 资源 | 我们用它做了什么 | 是否含可复用代码 |
|---|---|---|
| [AM3d](https://github.com/mahsan2/AM3d)（论文作者仓库） | 只读取其 **99 份 xlsx 结果表 / png 曲线** 作为数值对照 | ❌ **仓库内没有任何训练代码**（413 个文件全为 xlsx、png、notebook 输出） |
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
