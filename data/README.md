# data/ —— 本目录不进 Git

真实数据只存在两个地方：

1. **Kaggle 私有 Dataset `am-defect-v1`**（唯一真源，8 人挂载同一个）；
2. 数据管理员本机的一份原始备份。

本目录结构（本地）：

```text
data/
├── raw/AMdataset/            # git clone --depth 1, ≈896 MB
│   ├── dataset/badcylinder/      # D1 defect   136 张
│   ├── dataset/goodcylinder/     # D1 nondefect 135 张
│   └── Augmented_data/{Badaug,Goodaug}/  # D2 947 + 2070 张
├── processed/                # 预处理产物（224×224）
│   ├── D1/{defect,nondefect}/
│   ├── D2/{defect,nondefect}/
│   └── D2p/{defect,nondefect}/   # 我们自建、保留 source_id 的增强版
└── manifest.csv              # path,dataset,label,orig_w,orig_h,sha256,source_id
```

下载、预处理、划分的详细规则见 [`../docs/DATA.md`](../docs/DATA.md)。

> 第一次拿到数据后，请核对：D1 = 136 + 135 = 271；D2 = 947 + 2070 = 3017。
> 对不上说明下载不完整，不要继续跑。
