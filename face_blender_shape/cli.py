"""命令行入口：解析 YAML/参数，读取 CSV blendshape 序列并驱动 FaceBlenderRuntime 播放预览。"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from face_blender_shape.blender_runtime import FaceBlenderRuntime
from face_blender_shape.constants import (
    DEFAULT_OPEN3D_HEIGHT,
    DEFAULT_OPEN3D_WIDTH,
    DEFAULT_PLAYBACK_FPS,
    DEFAULT_VIEW_SCALE,
)
from face_blender_shape.io import load_blendshape_csv

# 默认预览配置文件名（相对当前工作目录）。
DEFAULT_PREVIEW_CONFIG_NAME = "face_blender_preview.yaml"


@dataclass
class PreviewConfig:
    """
    从 YAML 读取的预览参数集合。
    path: blendshape CSV 路径（必填）；每行一帧，列顺序与 SRanipal 37 维一致。
    fps: 播放节奏；大于 0 时按 1/fps 秒休眠逐帧，否则不延时连续刷帧。
    view_scale: Open3D 相机取景比例；数值越大脸在画面里越大，用于米级角色等场景。
    window_width / window_height: 预览窗口宽高（像素）。
    head: FBX 里承载 blendshape 的网格对象在 Blender 中的名称；null 时用 MetaHuman 默认头对象名。
    extra_meshes: 除面部外还要合并进同屏预览的网格对象名列表（如牙齿、头发）；null 时默认牙齿与双眼。
    """

    path: str
    fps: float
    view_scale: float
    window_width: int
    window_height: int
    head: str | None
    extra_meshes: tuple[str, ...] | None


def parse_extra_mesh_names(s: str | None) -> tuple[str, ...] | None:
    """
    解析逗号分隔的网格名字符串。
    s: 原始字符串；空或仅空白则返回 None。
    """
    if s is None or not str(s).strip():
        return None
    return tuple(x.strip() for x in str(s).split(",") if x.strip())


def normalize_extra_meshes_yaml(value: Any) -> tuple[str, ...] | None:
    """
    将 YAML 中的 extra_meshes 转为网格名元组；空则返回 None。
    value: YAML 中可为 list、逗号分隔字符串或 null。
    """
    if value is None:
        return None
    if isinstance(value, list):
        t = tuple(str(x).strip() for x in value if str(x).strip())
        return t if t else None
    return parse_extra_mesh_names(str(value) if value else None)


def load_preview_config(config_path: Path) -> PreviewConfig:
    """
    从 YAML 文件加载预览配置；缺省键使用与常量一致的默认值。
    config_path: YAML 文件路径（建议为绝对路径或已基于 cwd 解析）。
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("配置文件顶层必须是 YAML 映射（键值对）")
    path_val = raw.get("path")
    path_out = None if path_val is None else str(path_val).strip() or None
    if not path_out:
        raise ValueError("配置中 path 必须为非空字符串，指向 blendshape CSV 文件")
    head_val = raw.get("head")
    head_out = None if head_val is None else str(head_val).strip() or None
    return PreviewConfig(
        path=path_out,
        fps=float(raw.get("fps", DEFAULT_PLAYBACK_FPS)),
        view_scale=float(raw.get("view_scale", DEFAULT_VIEW_SCALE)),
        window_width=int(raw.get("window_width", DEFAULT_OPEN3D_WIDTH)),
        window_height=int(raw.get("window_height", DEFAULT_OPEN3D_HEIGHT)),
        head=head_out,
        extra_meshes=normalize_extra_meshes_yaml(raw.get("extra_meshes")),
    )


def preview_sequence(
    path: str | Path,
    fps: float = DEFAULT_PLAYBACK_FPS,
    *,
    view_scale: float = DEFAULT_VIEW_SCALE,
    window_width: int = DEFAULT_OPEN3D_WIDTH,
    window_height: int = DEFAULT_OPEN3D_HEIGHT,
    head_object_name: str | None = None,
    extra_mesh_names: tuple[str, ...] | None = None,
) -> None:
    """
    按 CSV 序列逐帧驱动 Open3D 预览。
    path: blendshape CSV 路径。
    fps: 每帧间隔由 fps 推算；<=0 则不 sleep。
    其余关键字参数含义与 FaceBlenderRuntime 一致。
    """
    data = load_blendshape_csv(path)
    runtime = FaceBlenderRuntime(
        enable_viewer=True,
        view_scale=view_scale,
        window_width=window_width,
        window_height=window_height,
        head_object_name=head_object_name,
        extra_mesh_names=extra_mesh_names,
    )
    frame_delay = 1.0 / fps if fps > 0 else 0.0

    for idx, blendshapes in enumerate(data, start=1):
        print(f"frame {idx}/{len(data)}")
        runtime.update_visualizer(blendshapes)
        if frame_delay > 0:
            time.sleep(frame_delay)


def build_parser() -> argparse.ArgumentParser:
    """
    构建仅含「配置文件路径」参数的解析器；业务参数全部在 YAML 中填写。
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


def run_preview(cfg: PreviewConfig) -> None:
    """
    按配置启动 CSV 序列预览。
    cfg: load_preview_config 得到的预览配置。
    """
    preview_sequence(
        cfg.path,
        cfg.fps,
        view_scale=cfg.view_scale,
        window_width=cfg.window_width,
        window_height=cfg.window_height,
        head_object_name=cfg.head,
        extra_mesh_names=cfg.extra_meshes,
    )


def main(argv: list[str] | None = None) -> int:
    """
    入口：解析可选 --config，读取 YAML 并运行预览。
    argv: 若传入则交给 argparse.parse_args；否则使用默认进程参数；用于测试。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    cfg = load_preview_config(config_path)
    run_preview(cfg)
    return 0


