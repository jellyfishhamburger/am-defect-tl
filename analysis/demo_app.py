# Author: <姓名> | Task: 预测 demo（Gradio） | PR: #<编号>
# Status: 基于 Gradio 框架（开源，见 NOTICE.md），界面逻辑为本组编写
"""现场演示：上传一张 3D 打印件照片 → 输出 defect / nondefect 概率 + Grad-CAM 热力图。

口头报告 10:00–12:30 的 live demo 用，代码量小、视觉效果好。
用法：
    python -m analysis.demo_app --model vgg16 --ckpt <path>   # 本地
    # Kaggle 上需开启 share=True 才会生成公网链接
"""

from __future__ import annotations


def predict(image, model, transform, device) -> dict:
    """返回 {"label": ..., "p_defect": ..., "p_nondefect": ...}"""
    raise NotImplementedError


def build_ui(model, transform, device):
    raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError
