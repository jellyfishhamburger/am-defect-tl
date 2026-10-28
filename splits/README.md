# splits/ —— 全组唯一的数据划分（契约文件）

**只有地基组能改这里。** 冻结（2026-10-29）之后任何人不得修改；必须改则全组重跑。

## 文件

| 文件 | 用途 | 划分方式 |
|---|---|---|
| `D1.csv` | 主实验 | 分层随机，train/val/test = 70/10/20，seed = 42 |
| `D2.csv` | 主实验 | 同上（`source_id` 为空，作者已丢失增强溯源） |
| `D2p_group.csv` | 泄漏对照实验 | 按 `source_id` 分组划分，同一原图的增强副本不跨 split |

## 列定义

```csv
path,label,split,source_id
```

- `path`：**相对数据根目录**的路径，例 `data/D1/defect/IMG_9078.JPG`
  （代码里用 `--data-root` 拼接，本地传 `_kaggle-build/am-defect-v1/`，
   Kaggle 传 `/kaggle/input/datasets/kennethdeng/am-defect-v1/`，一份 splits 两边都能用）
- `label`：`defect` | `nondefect`
- `split`：`train` | `val` | `test`
- `source_id`：D1 = 原图文件名（`IMG_9078`）；D2 = 空；D2p = 原图文件名

## 校验

```bash
pytest tests/test_contracts.py -v
```

检查项：① 三个 split 的类别比例；② 同一 `source_id` 不跨 split（D2p 必须通过）；
③ 每行 path 真实存在（数据未就位时该用例自动 skip）。

## 生成命令（由 `src/splits.py` 实现）

```bash
python -m src.splits --dim D1 --seed 42
python -m src.splits --dim D2 --seed 42
python -m src.splits --dim D2p --group-by source_id --seed 42
```
