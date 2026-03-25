#!/usr/bin/env python3
"""命令行：将 SRanipal 37 列 CSV 转为 ARKit 51 列 CSV，用于验证映射与外部工具对接。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 保证以「仓库根」为 cwd 调用时也能 import 包。
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from face_blender_shape.blendshape_mapping import convert_sranipal_batch
from face_blender_shape.io.blendshape_csv import (
    load_blendshape_csv,
    save_arkit_blendshape_csv,
)


def main(argv: list[str] | None = None) -> int:
    """
    解析命令行，读取 SRanipal CSV，写出 ARKit CSV。
    argv: 传入则替代 sys.argv（供测试）；默认 None 使用进程参数。
    """
    parser = argparse.ArgumentParser(
        description="将 SRanipal（37 列）blendshape CSV 转为 ARKit（51 列）CSV。",
    )
    parser.add_argument(
        "input",
        type=str,
        help="输入 CSV 路径（37 列，列序同 BLENDSHAPE_NAMES）。",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="输出 ARKit CSV（51 列），须为可写路径，例如 data/sample_arkit.csv 或 ~/Desktop/out.csv。",
    )
    parser.add_argument(
        "--header",
        action="store_true",
        help="首行写入 ARKit 形态名（逗号分隔）。",
    )
    args = parser.parse_args(argv)
    try:
        data = load_blendshape_csv(args.input)
        arkit = convert_sranipal_batch(data)
        save_arkit_blendshape_csv(args.output, arkit, write_header=args.header)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"OK: {data.shape[0]} 帧 × 37 -> {arkit.shape[1]} 列 -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
