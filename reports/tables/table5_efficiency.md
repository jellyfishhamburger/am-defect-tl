### Table 5 — Inference cost per image (batch size 1, 224×224)

| Model | Total params | GFLOPs | CPU ms/img | GPU ms/img |
|---|---|---|---|---|
| MobileNetV2 | 2,305,986 | 0.599 | 26.1 | N/A |
| EfficientNet-B0 | 4,089,662 | 0.769 | 51.3 | N/A |
| ResNet50 | 23,639,298 | 8.175 | 71.0 | N/A |
| Xception | 20,938,218 | 9.102 | 114.9 | N/A |
| InceptionResNetV2 | 54,404,962 | 12.935 | 239.3 | N/A |
| EfficientNetV2-M | 52,940,470 | 10.722 | 344.9 | N/A |
| EfficientNet-B7 | 63,950,994 | 10.335 | 340.5 | N/A |
| ResNet101 | 42,631,426 | 15.599 | 156.2 | N/A |
| VGG16 | 134,293,506 | 30.693 | 169.9 | N/A |
| VGG19 | 139,603,202 | 39.017 | 251.2 | N/A |
| NASNet-Large | 84,978,392 | 20.560 | 580.1 | N/A |

> CPU timings: mean of 20 forward passes after 5 warm-up iterations, batch size 1. GPU timings pending (needs a Kaggle T4 session).
