"""根据 Blender 材质推导 Open3D 顶点色。"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from mathutils import Vector  # pyright: ignore[reportMissingModuleSource]

_DEFAULT_PRINCIPLED_RGB = np.array([0.85, 0.85, 0.88], dtype=np.float64)
_PUPIL_COLOR_FLOOR = np.array([0.10, 0.085, 0.09], dtype=np.float64)
_IRIS_RGB_SCALE = np.array([0.94, 1.06, 1.04], dtype=np.float64)
_HAZEL_TINT = np.array([0.33, 0.23, 0.16], dtype=np.float64)
_SCLERA_SOFT = np.array([0.93, 0.94, 0.97], dtype=np.float64)


def principled_base_color_rgb(mat) -> np.ndarray:
    """读取材质的 Principled Base Color。

    mat: Blender 材质对象；允许为 None。
    """
    if mat is None:
        return _DEFAULT_PRINCIPLED_RGB.copy()
    if not getattr(mat, "use_nodes", False):
        return _DEFAULT_PRINCIPLED_RGB.copy()

    try:
        for node in mat.node_tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            color = node.inputs["Base Color"].default_value
            rgb = np.array(color[:3], dtype=np.float64)
            return np.clip(rgb, 0.0, 1.0)
    except (AttributeError, KeyError, TypeError):
        return _DEFAULT_PRINCIPLED_RGB.copy()

    return _DEFAULT_PRINCIPLED_RGB.copy()


def material_rgb_for_eye_viewport(mat) -> np.ndarray:
    """为眼球材质推导更适合 Open3D 预览的颜色。

    mat: Blender 材质对象；允许为 None。
    """
    rgb = principled_base_color_rgb(mat)
    if mat is None:
        return rgb

    name = (mat.name or "").lower()
    if "pupil" in name:
        return np.clip(np.maximum(rgb, _PUPIL_COLOR_FLOOR), 0.0, 1.0)
    if "iris" in name:
        rich = np.clip(rgb * _IRIS_RGB_SCALE, 0.0, 1.0)
        mixed = 0.70 * rich + 0.30 * _HAZEL_TINT
        return np.clip(mixed, 0.0, 1.0)
    if float(rgb.mean()) >= 0.45:
        return _SCLERA_SOFT.copy()
    return rgb


def mesh_vertex_colors_from_materials(mesh, materials) -> np.ndarray:
    """按面材质把颜色平均到各顶点。

    mesh: 已三角化的 Blender 网格。
    materials: 材质槽列表。
    """
    count = len(mesh.vertices)
    acc = np.zeros((count, 3), dtype=np.float64)
    weights = np.zeros(count, dtype=np.float64)

    for poly in mesh.polygons:
        if materials and len(materials) > 0:
            material_index = min(int(poly.material_index), len(materials) - 1)
            mat = materials[material_index]
        else:
            mat = None
        rgb = principled_base_color_rgb(mat)
        for vertex_index in poly.vertices:
            acc[vertex_index] += rgb
            weights[vertex_index] += 1.0

    weights = np.maximum(weights, 1.0)
    return acc / weights[:, None]


def mesh_vertex_colors_dominant_material(mesh, materials) -> np.ndarray:
    """按顶点邻接面积最大的材质为顶点着色。

    mesh: 已三角化的 Blender 网格。
    materials: 材质槽列表。
    """
    vertices = mesh.vertices
    weights_by_vertex = [defaultdict(float) for _ in range(len(vertices))]

    for poly in mesh.polygons:
        vertex_ids = list(poly.vertices)
        if len(vertex_ids) < 3:
            continue
        if materials and len(materials) > 0:
            material_index = min(int(poly.material_index), len(materials) - 1)
        else:
            material_index = -1
        c0 = Vector(vertices[vertex_ids[0]].co)
        c1 = Vector(vertices[vertex_ids[1]].co)
        c2 = Vector(vertices[vertex_ids[2]].co)
        area = (c1 - c0).cross(c2 - c0).length / 2.0
        for vertex_index in poly.vertices:
            weights_by_vertex[vertex_index][material_index] += float(area)

    out = np.zeros((len(vertices), 3), dtype=np.float64)
    for index, material_weights in enumerate(weights_by_vertex):
        if not material_weights:
            out[index] = material_rgb_for_eye_viewport(None)
            continue
        best_index = max(material_weights.items(), key=lambda item: item[1])[0]
        if best_index < 0 or best_index >= len(materials):
            mat = None
        else:
            mat = materials[best_index]
        out[index] = material_rgb_for_eye_viewport(mat)

    return out


def is_eye_mesh_object(obj) -> bool:
    """判断对象名是否像眼球网格。

    obj: Blender 对象；允许为 None。
    """
    if obj is None:
        return False
    if not getattr(obj, "name", None):
        return False
    return "eye" in obj.name.lower()
