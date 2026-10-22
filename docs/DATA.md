# 数据集说明（DATA）

> 本文所有数字由本组于 2026-10-22 通过 GitHub Tree API 逐目录统计、并抽样读图核实，
> 与作者 README 有 1 处不一致（已标注）。**不要照抄作者 README 的数字写报告。**

## 1. 必需数据：AMdataset（作者公开部分）

- 仓库：<https://github.com/mahsan2/AMdataset>（全库 896 MB，3,295 个文件）
- 采集：Creality Ender 3 Pro 打印的圆柱试样 + Canon EOS 60D 延时摄影，University of Oklahoma SMIS Lab
- 许可：**仓库未声明任何许可证**，仅要求引用论文。→ 仅用于课程复现，不二次分发，报告注明出处。

| | **D1（Dataset One）** | **D2（Dataset Two）** |
|---|---|---|
| 目录 | `dataset/badcylinder/`、`dataset/goodcylinder/` | `Augmented_data/Badaug/`、`Augmented_data/Goodaug/` |
| 类别计数 | 缺陷 **136** / 无缺陷 **135** = **271** | 缺陷 **947** / 无缺陷 **2070** = **3017** |
| 文件名 | `IMG_8915.JPG` … `IMG_9213.JPG`（原始编号连续） | `aug_<0–9>_<随机数>.png`（**无原图索引**） |
| 分辨率 | **5184 × 3456** | **128 × 128** |
| 单张体积 | ≈ 3 MB | ≈ 14–18 KB |
| 目录体积 | **848.8 MB** | **47.3 MB** |
| 作者划分 | train 190 / val 27 / test 55（70/10/20） | train 2111 / val 308 / test 598（70/10/20） |

> ⚠️ **与作者 README 的差异**：README 写 D1 为 136/136（272 张），仓库实际只有 135 张 goodcylinder，
> 共 271 张。作者论文的混淆矩阵求和为 272 → 上传时漏了 1 张。报告里需要说明这一点。

## 2. 第三个数据集：确认存在，但未公开

论文摘要写 "across **three** datasets"，公开仓库只有两个。本组用作者 `AM3d` 仓库里 99 份结果表的
混淆矩阵求和，反推出三个研究的样本总量：

| 研究 | 由混淆矩阵求和 | 公开仓库 | 说明 |
|---|---|---|---|
| Study One | 190 + 27 + 55 = **272** | ✅ = D1 | — |
| Study Two | 2111 + 308 + 598 = **3017** | ✅ = D2 | D2 是 D1 的增强版，非独立采集 |
| Study Three | 1089 + 159 + 309 = **1557** | ❌ 无 | 任何仓库都没有这批图 |

**处置**：① 组长发邮件向通讯作者 `ahsan@ou.edu` 索取；② 报告写明复现 66/99 个单元，剩余 33 个在表中打
† 标注 "reported by authors, not reproduced"；③ 用自建 D2′（见 §4）与公开的 FDM 缺陷数据集补位做扩展实验。

## 3. 下载与预处理（地基组负责，只做一次）

```bash
# ① 下载（二选一）
git clone --depth 1 https://github.com/mahsan2/AMdataset.git data/raw/AMdataset
# 或
curl -L -o data/raw/amdataset.zip https://codeload.github.com/mahsan2/AMdataset/zip/refs/heads/main

# ② 预处理（resize 参数必须记录，报告要写）
python -m src.data_prepare --raw data/raw/AMdataset --out data/processed --size 224
```

预处理规则（**写进报告，作者没有说明他们的做法，这是我们的选择**）：

| 数据集 | 规则 | 理由 |
|---|---|---|
| D1 | **直接 `cv2.resize` 到 224×224，忽略长宽比**（把 5184×3456 拉成正方形），像素 /255 | 与作者 notebook 的实现一致（见 [AUTHOR_CODE.md](AUTHOR_CODE.md) §3） |
| D2 | 同上：`cv2.resize` 128→224 | 已是正方形，无变形 |

> **保真 vs 改进**：作者原始实现是"直接拉伸、不裁剪"。我们的主实验沿用这一做法以保证可比性；
> 另设一组"保持长宽比 + 中心裁剪"的对照实验，用来验证预处理对结果的影响（写进报告的调参讨论）。

产物：`data/processed/D1/<label>/*.jpg`、`data/processed/D2/<label>/*.jpg`、`data/manifest.csv`
（字段：`path,dataset,label,orig_w,orig_h,sha256,source_id`）。

## 4. 划分契约（全组唯一一份，冻结后不得修改）

| 文件 | 用途 | 规则 |
|---|---|---|
| `splits/D1.csv` | 主实验 | 分层抽样 70/10/20，seed = 42 |
| `splits/D2.csv` | 主实验 | 同上；`source_id` 留空（作者已丢失溯源） |
| `splits/D2p_group.csv` | **泄漏对照实验** | 由我们自建 D2′：从 D1 重做增强，`source_id` = 原图文件名；按 `source_id` **分组**划分，保证同一原图的增强副本不跨 train/test |

列定义：`path,label,split,source_id`，其中 `label ∈ {defect, nondefect}`，`split ∈ {train, val, test}`。

**校验**：`pytest tests/test_contracts.py` 会检查 ① 三个 split 的类别分布；② 同一 `source_id` 不跨 split；
③ 所有 path 真实存在（数据未就位时自动 skip）。

## 5. 已知数据问题（直接作为报告 Discussion 素材）

1. **D2 是 D1 的增强版，且溯源被抹掉** —— `aug_<k>_<随机数>` 的随机数范围 99–9894，不是原图编号，
   无法还原"哪张增强图来自哪张原图"。作者按 70/10/20 **随机**划分 → 同一原图的增强副本可能同时落在
   train 和 test → **数据泄漏**，可解释作者报告的 0.99–1.00 准确率。
   → 我们的 D2′ 分组划分实验就是对这一点的定量验证。
2. **两个数据集分辨率差 40 倍**（5184→224 下采样 vs 128→224 上采样），是 ResNet 系在 D1 上掉到 0.85 的
   合理解释之一。→ 补 128 / 224 / 320 输入尺寸对照实验。
3. **D2 类别不平衡**（947 : 2070 ≈ 1 : 2.2），只看 Accuracy 会掩盖少数类表现。
   → 除论文的 10 个指标外，额外报 balanced accuracy 与每类 recall。

## 6. 可选补充数据集

| 数据集 | 体量 | 许可 | 用途 |
|---|---|---|---|
| [FDM-3D-Printing-Defect-Dataset](https://github.com/wengmhu/FDM-3D-Printing-Defect-Dataset) | 1,912 张，5 类缺陷 | MIT | 5 分类迁移实验，证明方法不只适用于二分类 |
| [Quality Control in AM (Zenodo 14712897)](https://zenodo.org/records/14712897) | 32.3 GB | CC BY 4.0 | 体积过大，**不建议** |

## 7. 引用

> M. M. Ahsan, S. Raman, Y. Liu, and Z. Siddique, "Defect analysis of 3D printed object using transfer
> learning approaches," *Expert Systems with Applications*, vol. 253, p. 124293, Nov. 2024,
> doi: 10.1016/j.eswa.2024.124293.
