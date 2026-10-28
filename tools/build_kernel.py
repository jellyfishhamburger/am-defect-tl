# Author: 组长 + agent | Task: 把代码打包成自包含 Kaggle kernel 并推送 | PR: 待提
# Status: 本组原创实现
"""把仓库代码打包成一个「自包含」的 Kaggle kernel 并推送执行。

原理：把 src/ configs/ splits/ requirements 打成一个 zip → base64 内嵌进 main.py →
上传到 Kaggle（只需 code_file，不依赖 git clone、不需要 token），运行时解压后执行命令。

用法：
    # 冒烟（只跑 1 个 run，1 epoch，20 张图）
    python tools/build_kernel.py --slug am-defect-smoke \
        --cmd "python -m src.batch --data-root /kaggle/input/datasets/kennethdeng/am-defect-v1 --out results --smoke --limit-runs 1"

    # 只打包不推送
    python tools/build_kernel.py --slug am-defect-smoke --no-push
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = "kennethdeng/am-defect-v1"
DEFAULT_DATA_ROOT = "/kaggle/input/datasets/kennethdeng/am-defect-v1"

RUNNER = '''# 由 tools/build_kernel.py 自动生成 —— 自包含运行器
import base64, io, os, shutil, subprocess, sys, zipfile

CODE_B64 = """__CODE_B64__"""
CMD = "__CMD__"
WORK = "/kaggle/working"
code_dir = os.path.join(WORK, "code")

try:
    import timm  # noqa: F401
except Exception:
    print("[runner] timm 缺失，尝试安装 ...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=False)

shutil.rmtree(code_dir, ignore_errors=True)
os.makedirs(code_dir, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_B64))) as z:
    z.extractall(code_dir)
os.chdir(code_dir)
sys.path.insert(0, code_dir)

import torch
print("[runner] python:", sys.version.split()[0], flush=True)
print("[runner] torch:", torch.__version__, "cuda:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("[runner] gpu:", torch.cuda.get_device_name(0), flush=True)
print("[runner] cwd:", os.getcwd(), flush=True)
print("[runner] cmd:", CMD, flush=True)

rc = subprocess.run(CMD, shell=True).returncode
print("[runner] exit code:", rc, flush=True)

# 把结果搬到 /kaggle/working/results，方便 kaggle kernels output 下载
# 若训练已经直接写 /kaggle/working/results，则跳过复制（路径相同）
src = os.path.join(code_dir, "results")
dst = os.path.join(WORK, "results")
if os.path.isdir(src) and os.path.abspath(src) != os.path.abspath(dst):
    shutil.copytree(src, dst, dirs_exist_ok=True)
if os.path.isdir(dst):
    n = sum(len(files) for _, _, files in os.walk(dst))
    print("[runner] results ->", dst, "files:", n, flush=True)
else:
    print("[runner] WARN: no results dir produced", flush=True)
sys.exit(rc)
'''


def build_zip() -> bytes:
    """把需要运行的文件打进 zip（不含 data、.git），并写入 VERSION 记录 commit。"""
    patterns = ["src/*.py", "configs/*.yaml", "splits/*.csv", "requirements.txt"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for pattern in patterns:
            for p in sorted(ROOT.glob(pattern)):
                z.write(p, p.relative_to(ROOT).as_posix())
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                    cwd=ROOT, capture_output=True, text=True, timeout=5
                                    ).stdout.strip()
        except Exception:
            commit = ""
        z.writestr("VERSION", commit or "unknown")
    return buf.getvalue()


def write_kernel(folder: Path, slug: str, cmd: str, gpu: bool, internet: bool) -> None:
    payload = base64.b64encode(build_zip()).decode("ascii")
    main_py = RUNNER.replace("__CODE_B64__", payload).replace("__CMD__", cmd)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "main.py").write_text(main_py, encoding="utf-8")
    meta = {
        "id": f"kennethdeng/{slug}",
        "title": slug,
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": internet,
        "dataset_sources": [DATASET],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def push(folder: Path) -> int:
    exe = shutil.which("kaggle") or "kaggle"
    print(f"[kernel] push {folder} ...", flush=True)
    return subprocess.run([exe, "kernels", "push", "-p", str(folder)]).returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, help="Kaggle kernel slug，如 am-defect-smoke")
    ap.add_argument("--cmd", default=(
        f"python -m src.batch --data-root {DEFAULT_DATA_ROOT} --out results --smoke --limit-runs 1"))
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--no-internet", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--keep", action="store_true", help="保留构建目录")
    args = ap.parse_args()

    folder = ROOT / "kaggle" / args.slug
    write_kernel(folder, args.slug, args.cmd, gpu=not args.no_gpu, internet=not args.no_internet)
    size_kb = (folder / "main.py").stat().st_size / 1024
    print(f"[kernel] 构建完成：{folder}  main.py={size_kb:.0f} KB", flush=True)

    if args.no_push:
        return
    rc = push(folder)
    if not args.keep:
        print(f"[kernel] 构建产物保留在 {folder}（方便复查；可手动删除）", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
