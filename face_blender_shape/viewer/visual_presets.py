"""Open3D 观感预设与程序化肤色模式。"""

from __future__ import annotations

from dataclasses import dataclass

from face_blender_shape.core.viewer_defaults import DEFAULT_OPEN3D_VISUAL_PRESET


@dataclass(frozen=True)
class Open3DViewerTune:
    """描述 Open3D 视图观感参数。"""

    background_rgb: tuple[float, float, float]
    baked_ambient: float
    baked_diffuse: float
    bake_key_frac: float
    bake_fill_frac: float
    warm_sss_rgb_scale: tuple[float, float, float]
    specular_scale: float
    matte_gamma: float


@dataclass(frozen=True)
class ProceduralSkinMode:
    """描述程序化肤色模式。"""

    detailed_zones: bool


_PRESETS: dict[str, tuple[Open3DViewerTune, ProceduralSkinMode]] = {
    "neutral_bust": (
        Open3DViewerTune(
            background_rgb=(0.24, 0.24, 0.26),
            baked_ambient=0.72,
            baked_diffuse=0.32,
            bake_key_frac=0.70,
            bake_fill_frac=0.30,
            warm_sss_rgb_scale=(0.03, 0.012, 0.008),
            specular_scale=0.15,
            matte_gamma=1.0,
        ),
        ProceduralSkinMode(detailed_zones=False),
    ),
    "detailed": (
        Open3DViewerTune(
            background_rgb=(0.86, 0.89, 0.93),
            baked_ambient=0.20,
            baked_diffuse=0.85,
            bake_key_frac=0.85,
            bake_fill_frac=0.15,
            warm_sss_rgb_scale=(0.12, 0.04, 0.02),
            specular_scale=1.0,
            matte_gamma=1.0,
        ),
        ProceduralSkinMode(detailed_zones=True),
    ),
}


def active_preset_name() -> str:
    """返回当前启用的预设名。"""
    name = (DEFAULT_OPEN3D_VISUAL_PRESET or "neutral_bust").strip().lower()
    if name not in _PRESETS:
        return "neutral_bust"
    return name


def get_open3d_viewer_tune() -> Open3DViewerTune:
    """返回当前预设对应的显示参数。"""
    return _PRESETS[active_preset_name()][0]


def get_procedural_skin_mode() -> ProceduralSkinMode:
    """返回当前预设对应的程序化肤色模式。"""
    return _PRESETS[active_preset_name()][1]
