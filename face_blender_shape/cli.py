"""命令行入口：读取预览配置并启动播放。"""

from __future__ import annotations

import argparse
from pathlib import Path

from face_blender_shape.app.preview import preview_sequence
from face_blender_shape.io.preview_config import load_preview_config

DEFAULT_PREVIEW_CONFIG_NAME = "face_blender_preview.yaml"


def build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器。

    返回: 仅包含 `--config` 参数的解析器。
    """
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="根据 YAML 配置文件预览 blendshape CSV（见 face_blender_preview.yaml）。",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_PREVIEW_CONFIG_NAME,
        help=f"预览配置 YAML 路径（默认: {DEFAULT_PREVIEW_CONFIG_NAME}，相对当前工作目录）",
    )
    return parser


def resolve_config_path(config_path: str) -> Path:
    """把命令行参数中的配置路径解析为绝对路径。

    config_path: 命令行传入的配置文件路径。
    """
    path = Path(config_path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def main(argv: list[str] | None = None) -> int:
    """解析命令行并启动预览。

    argv: 自定义参数列表；为 None 时使用进程参数。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = resolve_config_path(args.config)
    config = load_preview_config(config_path)
    preview_sequence(
        config.path,
        config.fps,
        view_scale=config.view_scale,
        window_width=config.window_width,
        window_height=config.window_height,
        head_object_name=config.head,
        extra_mesh_names=config.extra_meshes,
        dual_view=config.dual_view,
        fbx_path=config.fbx,
    )
    return 0
