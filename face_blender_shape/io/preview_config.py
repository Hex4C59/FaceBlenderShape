"""预览配置解析与归一化。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from face_blender_shape.core.viewer_defaults import (
    DEFAULT_OPEN3D_HEIGHT,
    DEFAULT_OPEN3D_WIDTH,
    DEFAULT_PLAYBACK_FPS,
    DEFAULT_VIEW_SCALE,
)


@dataclass
class PreviewConfig:
    """描述一次 CSV 预览所需参数。

    path: 输入 CSV 路径。
    fps: 播放帧率。
    view_scale: 取景比例。
    window_width: 窗口宽度。
    window_height: 窗口高度。
    head: 头部对象名。
    extra_meshes: 附加网格名称元组。
    dual_view: 是否启用侧视窗口。
    fbx: 自定义 FBX 路径。
    """

    path: str
    fps: float
    view_scale: float
    window_width: int
    window_height: int
    head: str | None
    extra_meshes: tuple[str, ...] | None
    dual_view: bool
    fbx: str | None


def parse_extra_mesh_names(value: str | None) -> tuple[str, ...] | None:
    """把逗号分隔字符串转成附加网格名称元组。

    value: 原始字符串。
    """
    if value is None:
        return None
    if not str(value).strip():
        return None
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def normalize_extra_meshes_yaml(value: Any) -> tuple[str, ...] | None:
    """把 YAML 中的 extra_meshes 统一转成元组。

    value: YAML 中的 extra_meshes 字段。
    """
    if value is None:
        return None
    if isinstance(value, list):
        items = tuple(str(item).strip() for item in value if str(item).strip())
        if not items:
            return None
        return items
    return parse_extra_mesh_names(str(value))


def _normalize_optional_string(value: Any) -> str | None:
    """把可选字符串字段归一化为非空字符串或 None。

    value: 原始字段值。
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def load_preview_config(config_path: Path) -> PreviewConfig:
    """从 YAML 文件读取预览配置。

    config_path: 配置文件路径。
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("配置文件顶层必须是 YAML 映射（键值对）")

    path_out = _normalize_optional_string(raw.get("path"))
    if path_out is None:
        raise ValueError("配置中 path 必须为非空字符串，指向 blendshape CSV 文件")

    return PreviewConfig(
        path=path_out,
        fps=float(raw.get("fps", DEFAULT_PLAYBACK_FPS)),
        view_scale=float(raw.get("view_scale", DEFAULT_VIEW_SCALE)),
        window_width=int(raw.get("window_width", DEFAULT_OPEN3D_WIDTH)),
        window_height=int(raw.get("window_height", DEFAULT_OPEN3D_HEIGHT)),
        head=_normalize_optional_string(raw.get("head")),
        extra_meshes=normalize_extra_meshes_yaml(raw.get("extra_meshes")),
        dual_view=bool(raw.get("dual_view", False)),
        fbx=_normalize_optional_string(raw.get("fbx")),
    )
