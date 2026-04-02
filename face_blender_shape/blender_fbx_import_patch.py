"""修补 Blender 内置 FBX 导入器：不同版本灯光 RNA 不一致（如 ``cast_shadow``、``exposure``），
导入含灯光的 FBX 时按需跳过不存在的属性。"""

from __future__ import annotations

import math
from typing import Any

import bpy  # pyright: ignore[reportMissingModuleSource]
import io_scene_fbx.import_fbx as import_fbx  # pyright: ignore[reportMissingModuleSource]

_patch_done = False


def apply_blender_fbx_light_cast_shadow_patch() -> None:
    """在 ``bpy.ops.import_scene.fbx`` 之前调用；可重复调用，仅首次生效。"""
    global _patch_done
    if _patch_done:
        return

    def blen_read_light_fixed(fbx_tmpl: Any, fbx_obj: Any, settings: Any) -> Any:
        elem_name_utf8 = import_fbx.elem_name_ensure_class(fbx_obj, b"NodeAttribute")
        fbx_props = (
            import_fbx.elem_find_first(fbx_obj, b"Properties70"),
            import_fbx.elem_find_first(
                fbx_tmpl, b"Properties70", import_fbx.fbx_elem_nil
            ),
        )
        light_type = {
            0: "POINT",
            1: "SUN",
            2: "SPOT",
        }.get(import_fbx.elem_props_get_enum(fbx_props, b"LightType", 0), "POINT")

        lamp = bpy.data.lights.new(name=elem_name_utf8, type=light_type)

        if light_type == "SPOT":
            spot_size = import_fbx.elem_props_get_number(fbx_props, b"OuterAngle", None)
            if spot_size is None:
                spot_size = import_fbx.elem_props_get_number(
                    fbx_props, b"Cone angle", 45.0
                )
            lamp.spot_size = math.radians(spot_size)

            spot_blend = import_fbx.elem_props_get_number(
                fbx_props, b"InnerAngle", None
            )
            if spot_blend is None:
                spot_blend = import_fbx.elem_props_get_number(
                    fbx_props, b"HotSpot", 45.0
                )
            lamp.spot_blend = 1.0 - (spot_blend / spot_size)

        lamp.color = import_fbx.elem_props_get_color_rgb(
            fbx_props, b"Color", (1.0, 1.0, 1.0)
        )
        lamp.energy = (
            import_fbx.elem_props_get_number(fbx_props, b"Intensity", 100.0) / 100.0
        )
        exposure = import_fbx.elem_props_get_number(fbx_props, b"Exposure", 0.0)
        if hasattr(lamp, "exposure"):
            lamp.exposure = exposure
        use_shadow = import_fbx.elem_props_get_bool(fbx_props, b"CastShadow", True)
        if hasattr(lamp, "use_shadow"):
            lamp.use_shadow = use_shadow
        if hasattr(lamp, "cycles") and hasattr(lamp.cycles, "cast_shadow"):
            lamp.cycles.cast_shadow = use_shadow
        if hasattr(lamp, "shadow_color"):
            lamp.shadow_color = import_fbx.elem_props_get_color_rgb(
                fbx_props, b"ShadowColor", (0.0, 0.0, 0.0)
            )

        if settings.use_custom_props:
            import_fbx.blen_read_custom_properties(fbx_obj, lamp, settings)

        return lamp

    import_fbx.blen_read_light = blen_read_light_fixed
    _patch_done = True
