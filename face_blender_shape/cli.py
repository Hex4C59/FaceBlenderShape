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
from face_blender_shape.io import load_blendshape_csv

CommandHandler: TypeAlias = Callable[[Namespace], int]


def preview_sequence(
    path: str | Path,
    fps: float = DEFAULT_PLAYBACK_FPS,
    *,
    fbx_path: str | None = None,
    texture_path: str | None = None,
    cutaway: bool = False,
    wireframe_head: bool = False,
) -> None:
    """按顺序预览一段 blendshape CSV。

    参数:
        path: 输入 CSV 文件路径。
        fps: 预览帧率；小于等于 0 时不主动 sleep。
        fbx_path: 可选的 FBX 路径，用于覆盖默认模型资源。
        texture_path: 可选的贴图路径。
        cutaway: 是否移除嘴部部分面片以观察口腔内部。
        wireframe_head: 是否头壳线框 + 舌实体（与 cutaway 互斥，开启时 render 忽略 cutaway）。
    """
    data: NDArray[np.float64] = load_blendshape_csv(path)
    # 在 Blender 中加载头模、绑定 blendshape，并创建 Open3D 预览窗口。
    runtime = FaceBlenderRuntime(
        path=fbx_path,
        enable_viewer=True,  # 启用 Open3D 网格查看器（关闭则无可视化）
        texture_path=texture_path,  # 外置 albedo；为 None 时尽量从材质读取或不加载
        cutaway=cutaway,  # True 时裁掉口腔区域三角面，便于看内部
        wireframe_head=wireframe_head,  # True 时头壳 LineSet、舌三角面实体
    )
    frame_delay = 1.0 / fps if fps > 0 else 0.0  # 相邻两帧之间的间隔（秒）；fps≤0 时不等待
    frame_total = data.shape[0]  # CSV 行数，即总帧数

    # 逐帧回放 CSV：每行是一帧的 blendshape 权重向量。
    for idx in range(frame_total):
        blendshapes: NDArray[np.float64] = data[idx]  # 当前帧，长度应为 FRAME_WIDTH
        print(f"frame {idx + 1}/{frame_total}")  # 终端进度
        runtime.update_visualizer(blendshapes)  # Blender 变形 + Open3D 刷新
        if frame_delay > 0:
            time.sleep(frame_delay)  # 按 fps 节流，避免刷得过快


def preview_all_shapes(
    *,
    fbx_path: str | None = None,
    texture_path: str | None = None,
    wireframe_head: bool = False,
) -> None:
    """用统一数值扫过全部 blendshape 通道，方便检查模型响应。

    参数:
        fbx_path: 可选的 FBX 路径，用于覆盖默认模型资源。
        texture_path: 可选的贴图路径。
        wireframe_head: 是否头壳线框 + 舌实体。
    """
    runtime: FaceBlenderRuntime = FaceBlenderRuntime(
        path=fbx_path,
        enable_viewer=True,
        texture_path=texture_path,
        wireframe_head=wireframe_head,
    )

    for value in np.linspace(0.0, 1.0, 100):
        current = float(value)
        print(current)
        runtime.update_visualizer(np.ones(FRAME_WIDTH, dtype=np.float64) * current)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器。

    参数:
        无。
    """
    parser: ArgumentParser = argparse.ArgumentParser(prog="face_blender_shape")
    subparsers = parser.add_subparsers(dest="command")

    preview_parser = subparsers.add_parser(
        name="preview",
        help="Preview a blendshape CSV or sweep all shapes",
    )
    preview_parser.add_argument(
        "--path",
        type=str,
        help="CSV sequence with 37 blendshape columns",
    )
    preview_parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_PLAYBACK_FPS,
        help="Playback FPS",
    )
    preview_parser.add_argument("--fbx", type=str, help="Override FBX path")
    preview_parser.add_argument(
        "--texture",
        type=str,
        help="Skin texture image path (optional; uses bpy material texture if omitted)",
    )
    preview_parser.add_argument(
        "--cutaway",
        action="store_true",
        help="移除唇部面片，露出口腔内舌头",
    )
    preview_parser.add_argument(
        "--wireframe-head",
        action="store_true",
        dest="wireframe_head",
        help="头壳仅画线框，舌保持实体贴图，便于透视观察舌形变",
    )
    preview_parser.set_defaults(handler=handle_preview_command)
    return parser


def handle_preview_command(args: argparse.Namespace) -> int:
    """执行 preview 子命令。

    参数:
        args: argparse 解析后的命令行参数对象。
    """
    if args.path:
        preview_sequence(
            args.path,
            args.fps,
            fbx_path=args.fbx,
            texture_path=args.texture,
            cutaway=args.cutaway,
            wireframe_head=args.wireframe_head,
        )
        return 0

    preview_all_shapes(
        fbx_path=args.fbx,
        texture_path=args.texture,
        wireframe_head=args.wireframe_head,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """解析命令行并分发到对应子命令。

    参数:
        argv: 可选的命令行参数列表；为 None 时读取进程实参。
    """
    parser = build_parser()
    args: Namespace = parser.parse_args(argv)
    handler = cast(CommandHandler | None, getattr(args, "handler", None))
    
    if handler is None:
        parser.print_help()
        return 0

    return handler(args)
