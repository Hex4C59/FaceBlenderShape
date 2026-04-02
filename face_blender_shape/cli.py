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
from face_blender_shape.landmarks import TONGUE_SLICE

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
    open3d_dual_view: bool = False,
    open3d_camera_zoom: float = 0.6,
    tongue_vertex_lo: int | None = None,
    tongue_vertex_hi: int | None = None,
    tongue_adjacency_expand: int = 0,
) -> None:

    data: NDArray[np.float64] = load_blendshape_csv(path)

    runtime = FaceBlenderRuntime(
        path=fbx_path,
        wireframe_head=wireframe_head,  # True 时头壳 LineSet、舌三角面实体
        open3d_dual_view=open3d_dual_view,
        open3d_camera_zoom=open3d_camera_zoom,
        tongue_vertex_lo=tongue_vertex_lo,
        tongue_vertex_hi=tongue_vertex_hi,
        tongue_adjacency_expand=tongue_adjacency_expand,
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
        help="CSV 路径；每行 52 列，与 constants.BLENDSHAPE_NAMES 顺序一致",
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
    preview_parser.add_argument(
        "--open3d-dual-view",
        action="store_true",
        dest="open3d_dual_view",
        help="Open3D 双窗：侧面 + 正面",
    )
    preview_parser.add_argument(
        "--open3d-camera-zoom",
        type=float,
        default=0.2,
        metavar="Z",
        help="Open3D 初始镜头缩放 set_zoom（默认 0.6）",
    )
    preview_parser.add_argument(
        "--tongue-lo",
        type=int,
        default=None,
        metavar="N",
        help=(
            "线框模式下舌顶点全局下标下界（含）；未指定时与默认 FBX 一致 "
            f"（当前默认 {TONGUE_SLICE.start}）"
        ),
    )
    preview_parser.add_argument(
        "--tongue-hi",
        type=int,
        default=None,
        metavar="N",
        help=(
            "线框模式下舌顶点全局下标上界（不含）；未指定时与默认 FBX 一致 "
            f"（当前默认 {TONGUE_SLICE.stop}）"
        ),
    )
    preview_parser.add_argument(
        "--tongue-adjacency-expand",
        type=int,
        default=0,
        metavar="ITERS",
        help=(
            "沿共享边扩展舌三角面轮数：用于顶点下标区间外的衔接面；"
            "过大可能把邻接口腔网格算进舌，建议从 1～3 试"
        ),
    )
    preview_parser.set_defaults(handler=handle_preview_command)
    return parser


def handle_preview_command(args: argparse.Namespace) -> int:

    preview_sequence(
        args.path,
        args.fps,
        fbx_path=args.fbx,
        wireframe_head=args.wireframe_head,
        open3d_dual_view=args.open3d_dual_view,
        open3d_camera_zoom=args.open3d_camera_zoom,
        tongue_vertex_lo=args.tongue_lo,
        tongue_vertex_hi=args.tongue_hi,
        tongue_adjacency_expand=args.tongue_adjacency_expand,
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
