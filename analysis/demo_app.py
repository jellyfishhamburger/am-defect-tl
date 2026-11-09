# Author: Leiman Pan + agent | Task: 现场演示界面（Gradio） | PR: 待提
# Status: 基于 Gradio 框架（开源，见 NOTICE.md），界面逻辑与推理封装为本组编写
"""现场演示：上传一张 3D 打印件照片 → defect / nondefect 概率 + Grad-CAM 热力图。

用法：
    python -m analysis.demo_app                      # 打开浏览器界面
    python -m analysis.demo_app --warmup             # 只预热权重（演讲前必做一次）
    python -m analysis.demo_app --selftest           # 不开界面，跑通一次推理就退出
    python -m analysis.demo_app --model vgg16 --share
    python -m analysis.demo_app --host 0.0.0.0       # 让同一 Wi-Fi 的手机也能打开

两个环境变量（演讲现场没网时用）：
    HF_HUB_OFFLINE=1        禁止 timm 去 HuggingFace 下载骨干（必须提前 --warmup）
    GRADIO_ANALYTICS_ENABLED=False

只有 D1 + Adam 的 5 个 run 保存了 model.pt，所以界面只提供这 5 个模型；
其余 61 个 run 只有 metrics.json，不能推理——这一点写在界面说明里，避免被问倒。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
# Windows 上常见的系统代理（v2rayN 等）会把 127.0.0.1 也走代理，导致 Gradio 自查 503
for _k in ("NO_PROXY", "no_proxy"):
    _v = os.environ.get(_k, "")
    if "127.0.0.1" not in _v:
        os.environ[_k] = (_v + ",127.0.0.1,localhost").lstrip(",")

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "results" / "runs"
FIG_DIR = ROOT / "results" / "figures"

PRETTY = {
    "vgg16": "VGG16", "vgg19": "VGG19", "resnet50": "ResNet50", "resnet101": "ResNet101",
    "inception_resnet_v2": "InceptionResNetV2", "xception": "Xception",
    "mobilenet_v2": "MobileNetV2", "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_b7": "EfficientNet-B7", "efficientnet_v2m": "EfficientNetV2-M",
    "nasnet_large": "NASNet-Large",
}

# 模型缓存：首次使用时载入，之后常驻（CPU 上一次前进约 0.2–1.5 s）
_CACHE: dict[str, object] = {}


def available_runs() -> list[Path]:
    from analysis.checkpoint import list_checkpoints

    return list_checkpoints(RUNS_DIR)


def label_of(run_dir: Path) -> str:
    """'vgg16__D1__adam__s42' → 'VGG16 · D1 · Adam'"""
    parts = run_dir.name.split("__")
    model = PRETTY.get(parts[0], parts[0])
    ds = parts[1] if len(parts) > 1 else "?"
    opt = (parts[2] if len(parts) > 2 else "?").capitalize()
    return f"{model} · {ds} · {opt}"


def get_model(run_dir: Path):
    key = str(run_dir)
    if key not in _CACHE:
        from analysis.checkpoint import load_model

        model, ckpt = load_model(run_dir, device="cpu", pretrained=True)
        _CACHE[key] = (model, ckpt)
    return _CACHE[key]


def predict_one(run_dir: Path, image_path: str) -> dict:
    """单模型推理，返回 label 概率、叠加图、Grad-CAM 数组。"""
    import numpy as np
    from PIL import Image

    from analysis.gradcam import compute_cam, heat_overlay, load_image

    model, ckpt = get_model(run_dir)
    size = int(ckpt.get("input_size", 224))
    x, im = load_image(Path(image_path), size)
    cam, probs = compute_cam(model, x, class_idx=1)

    overlay = heat_overlay(im, cam, alpha=0.5)
    overlay_img = Image.fromarray((np.clip(overlay, 0, 1) * 255).astype("uint8"))
    return {
        "p_defect": float(probs[1]),
        "p_nondefect": float(probs[0]),
        "overlay": overlay_img,
        "cam": cam,
    }


def run_table(image_path: str) -> list[list]:
    """同一个输入跑遍所有可用 checkpoint，做模型间对比。"""
    import pandas as pd

    rows = []
    for run_dir in available_runs():
        meta = {}
        try:
            meta = __import__("json").loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        except Exception:
            pass
        try:
            r = predict_one(run_dir, image_path)
            rows.append([label_of(run_dir), round(r["p_defect"], 3),
                         "defect" if r["p_defect"] >= 0.5 else "nondefect",
                         round(meta.get("metrics", {}).get("test", {}).get("accuracy", float("nan")), 3)])
        except Exception as exc:  # 单个模型失败不影响整个对比表
            rows.append([label_of(run_dir), None, f"加载失败: {type(exc).__name__}", None])
    return rows


def ui_predict(image_path, run_label, compare_all):
    """Gradio 回调：返回 (Label, 叠加图, Markdown 说明, 对比表)。"""
    import gradio as gr

    if not image_path:
        return None, None, "先上传一张图片。", None

    runs = available_runs()
    by_label = {label_of(p): p for p in runs}
    if not runs:
        return None, None, "results/runs 下没有 model.pt，无法推理。", None

    run_dir = by_label.get(run_label, runs[0])
    try:
        r = predict_one(run_dir, image_path)
    except Exception as exc:
        # 降级：显示提前渲染好的示例图，保证现场不断流
        fig = FIG_DIR / f"predict_{run_dir.name}.png"
        msg = (f"**模型加载失败**：`{type(exc).__name__}: {exc}`\n\n"
               f"已切到预生成示例图（`{fig.name}`）。"
               "如果提示下载失败，说明现场没网且权重没预热——演讲前请跑 `--warmup`。")
        return None, str(fig) if fig.exists() else None, msg, None

    meta = {}
    try:
        import json

        meta = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    tm = meta.get("metrics", {}).get("test", {})
    n_test = meta.get("n_test", "?")

    verdict = "defect" if r["p_defect"] >= 0.5 else "nondefect"
    prob = {"defect": r["p_defect"], "nondefect": r["p_nondefect"]}

    note = (
        f"**判定：`{verdict}`** ｜ P(defect) = `{r['p_defect']:.3f}`\n\n"
        f"该模型在 D1 测试集（n={n_test}）上的表现："
        f"acc `{tm.get('accuracy', float('nan')):.3f}` · F1 `{tm.get('f1', float('nan')):.3f}` · "
        f"MCC `{tm.get('mcc', float('nan')):.3f}`。\n\n"
        "热力图是 Grad-CAM：红色区域是模型做判断时看得最重的地方。"
        "如果红色落在背景或反光上，说明它学到了伪相关。"
    )
    table = run_table(image_path) if compare_all else None
    return prob, r["overlay"], note, table


def demo_examples(per_class: int = 2) -> list[Path]:
    """界面上的示例图：优先用仓库自带的 samples/D1（224×224，未参与训练）。"""
    out = []
    for label in ("defect", "nondefect"):
        d = ROOT / "samples" / "D1" / label
        if d.exists():
            out += sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:per_class]
    return out


DEMO_EXAMPLES = demo_examples()


def build_ui(choices: list[str], default: str):
    import gradio as gr

    with gr.Blocks(title="3D 打印件缺陷检测 · 迁移学习复现") as demo:
        gr.Markdown(
            "# 3D 打印件缺陷检测 · 迁移学习复现\n"
            "**IDAT7215 Group Project · Topic 4** — 冻结 ImageNet 骨干，只训练论文同款分类头。\n\n"
            f"> 可推理的模型：{len(choices)} 个（D1 / Adam）。其余 run 只保存了指标，没有权重。"
            "  \n> 首次推理会载入骨干，约 3–8 秒；之后切换图片是秒级。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="filepath", label="上传 3D 打印件照片", height=320)
                if DEMO_EXAMPLES:
                    gr.Examples(examples=[[str(p)] for p in DEMO_EXAMPLES],
                                inputs=[img], label="示例（D1 测试集，未参与训练）")
                model_dd = gr.Dropdown(choices, value=default, label="模型")
                compare = gr.Checkbox(value=False, label="同时对比全部模型（慢约 5–10 秒）")
                btn = gr.Button("判定", variant="primary")
            with gr.Column(scale=1):
                out_label = gr.Label(num_top_classes=2, label="类别概率")
                out_img = gr.Image(label="Grad-CAM（红 = 模型关注区域）", height=320)
        out_note = gr.Markdown()
        out_table = gr.Dataframe(
            headers=["模型", "P(defect)", "判定", "D1 测试 acc"],
            datatype=["str", "number", "str", "number"],
            label="全部模型对比", wrap=True,
        )
        btn.click(ui_predict, [img, model_dd, compare], [out_label, out_img, out_note, out_table])
        img.change(ui_predict, [img, model_dd, compare], [out_label, out_img, out_note, out_table])
        gr.Markdown(
            "---\n"
            "**说明**：正类 = `defect`。预处理与训练一致（直接 resize 到 224×224、不裁剪长宽比）。"
            "D1 测试集只有 54 张，单张判错对总体指标影响很大，界面不承诺泛化性能。"
        )
    return demo


def warmup(verbose: bool = True) -> None:
    """把每个 checkpoint 的骨干都下载并跑一次前向，演讲前执行。"""
    import numpy as np
    import torch

    runs = available_runs()
    if not runs:
        print("[warmup] 没有找到 model.pt")
        return
    dummy = torch.zeros(1, 3, 224, 224)
    for i, run_dir in enumerate(runs, 1):
        try:
            model, ckpt = get_model(run_dir)
            with torch.no_grad():
                logits = model(dummy)
            if verbose:
                print(f"[warmup] {i}/{len(runs)} {label_of(run_dir)} ok "
                      f"(logits {logits.shape})", flush=True)
        except Exception as exc:
            print(f"[warmup] {i}/{len(runs)} {label_of(run_dir)} 失败：{type(exc).__name__}: {exc}",
                  flush=True)
    print("[warmup] 完成。现场请设置 HF_HUB_OFFLINE=1 再启动界面。")


def selftest() -> int:
    """不开界面：用合成图跑一次推理，验证载入 / 前向 / Grad-CAM 全链路。"""
    import numpy as np
    from PIL import Image

    runs = available_runs()
    if not runs:
        print("[selftest] 没有 model.pt")
        return 1
    # 用最小的 checkpoint（MobileNetV2）自检，避免一次拉 1 GB 骨干
    run_dir = min(runs, key=lambda p: (p / "model.pt").stat().st_size)
    tmp = Path(os.environ.get("TEMP", "/tmp")) / "am_defect_selftest.png"
    arr = (np.random.default_rng(0).random((480, 480, 3)) * 255).astype("uint8")
    Image.fromarray(arr).save(tmp)
    r = predict_one(run_dir, str(tmp))
    print(f"[selftest] {label_of(run_dir)} p_defect={r['p_defect']:.4f} "
          f"cam={r['cam'].shape} overlay={r['overlay'].size}")
    print("[selftest] 单模型链路 OK。要验证其余模型请跑 --warmup。")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="vgg16 / resnet50 / ...（默认第一个可用）")
    ap.add_argument("--ckpt", default=None, help="直接指定 run 目录")
    ap.add_argument("--warmup", action="store_true", help="只预热权重，不启动界面")
    ap.add_argument("--selftest", action="store_true", help="跑一次离线推理自检")
    ap.add_argument("--offline", action="store_true", help="设置 HF_HUB_OFFLINE=1（权重需已预热）")
    ap.add_argument("--mirror", action="store_true",
                    help="走 hf-mirror.com 下载 timm 权重（国内网络连不上 huggingface.co 时用）")
    ap.add_argument("--share", action="store_true", help="生成 Gradio 公网链接（需要联网）")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = 同网段手机可访问")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if args.mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    if args.warmup:
        warmup()
        return
    if args.selftest:
        raise SystemExit(selftest())

    runs = available_runs()
    if not runs:
        raise SystemExit("results/runs 下没有 model.pt，无法启动 demo。")
    if args.ckpt:
        run_dir = Path(args.ckpt)
        if not run_dir.is_absolute():
            run_dir = RUNS_DIR / args.ckpt
        runs = [run_dir] + [r for r in runs if r != run_dir]
    elif args.model:
        picked = [r for r in runs if r.name.split("__")[0] == args.model]
        if picked:
            runs = picked + [r for r in runs if r not in picked]

    # 默认用 MobileNetV2：CPU 上载入最快，现场切换不卡
    order = ["mobilenet_v2", "vgg16", "resnet50", "xception", "inception_resnet_v2"]
    runs = sorted(runs, key=lambda p: (order.index(p.name.split("__")[0])
                                       if p.name.split("__")[0] in order else 99))
    choices = [label_of(r) for r in runs]
    import gradio as gr

    kwargs = dict(server_name=args.host, server_port=args.port, share=args.share,
                  inbrowser=not args.no_browser, show_error=True)
    try:  # Gradio 6 起 theme 从 Blocks 移到 launch；旧版本仍在 Blocks 上
        build_ui(choices, choices[0]).launch(theme=gr.themes.Soft(), **kwargs)
    except TypeError:
        build_ui(choices, choices[0]).launch(**kwargs)


if __name__ == "__main__":
    main()
