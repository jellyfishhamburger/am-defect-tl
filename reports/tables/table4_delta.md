### Table 4 — Reproduction gap versus reported values (Adam runs)

| Model | D1 ours | D1 reported | Δ D1 | D2 ours | D2 reported | Δ D2 | Note |
|---|---|---|---|---|---|---|---|
| VGG16 | 1.0000 | 1.0000 | +0.0000 | 0.9900 | 0.9933 | -0.0033 |  |
| VGG19 | 1.0000 | 1.0000 | +0.0000 | 0.9900 | 0.9916 | -0.0016 |  |
| ResNet50 | 1.0000 | 0.9273 | +0.0727 | 0.9884 | 0.6739 | +0.3145 |  |
| ResNet101 | 1.0000 | 0.9091 | +0.0909 | 0.9917 | 0.8344 | +0.1573 |  |
| InceptionResNetV2 | 0.9815 | 1.0000 | -0.0185 | 0.9900 | 0.9967 | -0.0067 |  |
| Xception | 1.0000 | 1.0000 | +0.0000 | 0.9967 | 1.0000 | -0.0033 |  |
| MobileNetV2 | 1.0000 | 1.0000 | +0.0000 | 0.9950 | 0.9933 | +0.0017 |  |
| EfficientNet-B0 | 1.0000 | 0.4909 | +0.5091 | 0.9967 | 0.6739 | +0.3228 | collapsed$^{*}$ |
| EfficientNet-B7 | 0.9815 | 0.4909 | +0.4906 | 0.9983 | 0.6739 | +0.3244 | collapsed$^{*}$ |
| EfficientNetV2-M | 0.9815 | 0.4909 | +0.4906 | 0.9221 | 0.6739 | +0.2482 | collapsed$^{*}$ |
| NASNet-Large | 0.9444 | 1.0000 | -0.0556 | 0.9867 | 0.9967 | -0.0100 |  |

> $^{*}$ Authors' reported F1 = 0.000 (all samples predicted as the majority class), so the large Δ is a collapsed run on their side, not evidence of a better implementation.
