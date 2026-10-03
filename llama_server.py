#!/usr/bin/env python3
"""启动 Facet 专用的本地 llama-server 服务。

Usage:
    python llama_server.py

这是 Facet 的本地启动入口，不提供给其他应用复用；模型路径和服务参数
按当前机器的固定配置写在本文件中。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


MODEL_PATH = Path(
    "/Users/cuichen/.cache/huggingface/hub/models--ggml-org--qwen3.5-0.8b-gguf/"
    "snapshots/8fea620810c4afa23dd6443f999a48574c1611a3/"
    "Qwen3.5-0.8B-Q4_0.gguf"
)
MMPROJ_PATH = Path(
    "/Users/cuichen/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/"
    "snapshots/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0/"
    "mmproj-BF16.gguf"
)


def _build_command(server_path: str) -> list[str]:
    return [
        server_path,
        "-m",
        str(MODEL_PATH),
        "--mmproj",
        str(MMPROJ_PATH),
        "--alias",
        "Qwen3.5-0.8B",
        "--host",
        "127.0.0.1",
        "--port",
        "8765",
        "--no-ui",
        "--ctx-size",
        "4096",
        "--parallel",
        "1",
        "--threads",
        "4",
        "--threads-batch",
        "4",
        "--no-cont-batching",
        # Intel CPU 上将图片视觉 token（图像切分后的模型输入单位）限制为
        # 256；1024 会让 Qwen3.5-0.8B 的单张图片推理长时间不返回。
        "--image-min-tokens",
        "256",
        "--image-max-tokens",
        "256",
    ]


def main() -> int:
    server_path = shutil.which("llama-server")
    if not server_path:
        print("找不到 llama-server，请先确认它已安装并在 PATH 中。", file=sys.stderr)
        return 127

    missing = [path for path in (MODEL_PATH, MMPROJ_PATH) if not path.is_file()]
    if missing:
        print("模型文件不存在：", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    # 用 exec 替换当前 Python 进程，使 Ctrl-C 和退出码直接传给服务进程。
    os.execv(server_path, _build_command(server_path))
    return 0  # pragma: no cover - execv 成功后不会返回


if __name__ == "__main__":
    raise SystemExit(main())
