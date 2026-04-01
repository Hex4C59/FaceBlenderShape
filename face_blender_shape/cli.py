from __future__ import annotations

import argparse
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Callable, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from face_blender_shape.blender_runtime import FaceBlenderRuntime
from face_blender_shape.constants import DEFAULT_PLAYBACK_FPS, FRAME_WIDTH

CommandHandler: TypeAlias = Callable[[Namespace], int]


def load_blendshape_csv(path: str | Path) -> NDArray[np.float64]:
    csv_path = Path(path).expanduser()

    with open(csv_path, "r", encoding="utf-8") as file:
        first = file.readline().split(",")[0].strip()

    skiprows = 0
    try:
        float(first)
    except ValueError:
        skiprows = 1

    data = np.loadtxt(csv_path, delimiter=",", skiprows=skiprows)
    data = np.atleast_2d(data)

    if data.shape[1] != FRAME_WIDTH:
        raise ValueError(f"Expected {FRAME_WIDTH} columns, got {data.shape[1]}")

    return data.astype(float, copy=False)


def preview_sequence(
    path: str | Path,
    fps: float = DEFAULT_PLAYBACK_FPS,
    *,
    fbx_path: str | None = None,
    wireframe_head: bool = False,  # 是否头壳线框 + 舌实体。
) -> None:

    data: NDArray[np.float64] = load_blendshape_csv(path)

    runtime = FaceBlenderRuntime(
        path=fbx_path,
        wireframe_head=wireframe_head,  # True 时头壳 LineSet、舌三角面实体
    )
    frame_delay = (
        1.0 / fps if fps > 0 else 0.0
    )  # 相邻两帧之间的间隔（秒）；fps≤0 时不等待
    frame_total = data.shape[0]  # CSV 行数，即总帧数

    # 逐帧回放 CSV：每行是一帧的 blendshape 权重向量。
    for idx in range(frame_total):
        blendshapes: NDArray[np.float64] = data[idx]  # 当前帧，长度应为 FRAME_WIDTH
        print(f"frame {idx + 1}/{frame_total}")  # 终端进度
        runtime.update_visualizer(blendshapes)  # Blender 变形 + Open3D 刷新
        if frame_delay > 0:
            time.sleep(frame_delay)  # 按 fps 节流，避免刷得过快


def build_parser() -> argparse.ArgumentParser:

    parser: ArgumentParser = argparse.ArgumentParser(prog="face_blender_shape")
    subparsers = parser.add_subparsers(dest="command")

    preview_parser = subparsers.add_parser(
        name="preview",
        help="按 blendshape CSV 序列预览面部动画",
    )
    preview_parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="CSV 路径；每行 37 列，对应各 blendshape 权重",
    )
    preview_parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_PLAYBACK_FPS,
        help="播放帧率",
    )
    preview_parser.add_argument(
        "--fbx",
        type=str,
        help="覆盖默认头模 FBX 路径",
    )
    preview_parser.add_argument(
        "--wireframe-head",
        action="store_true",
        dest="wireframe_head",
        help="头壳仅画线框，舌保持实体网格（默认肤色），便于透视观察舌形变",
    )
    preview_parser.set_defaults(handler=handle_preview_command)
    return parser


def handle_preview_command(args: argparse.Namespace) -> int:

    preview_sequence(
        args.path,
        args.fps,
        fbx_path=args.fbx,
        wireframe_head=args.wireframe_head,
    )
    return 0


def main(argv: list[str] | None = None) -> int:

    parser = build_parser()
    args: Namespace = parser.parse_args(argv)
    handler = cast(CommandHandler | None, getattr(args, "handler", None))

    if handler is None:
        parser.print_help()
        return 0

    return handler(args)
