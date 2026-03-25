"""把变形后的 Blender 网格组装成预览帧。"""

from __future__ import annotations

import bpy
import numpy as np

from face_blender_shape.landmarks import extract_default_landmarks
from face_blender_shape.runtime.material_colors import (
    is_eye_mesh_object,
    mesh_vertex_colors_dominant_material,
    mesh_vertex_colors_from_materials,
)
from face_blender_shape.runtime.mesh_eval import (
    get_mesh_data,
    get_modified_mesh,
    vertices_coords_numpy,
)
from face_blender_shape.viewer.shading import SKIN_TONE


def has_extra_meshes_for_view(append_parts) -> bool:
    """判断是否存在需要合并到视图中的附加网格。

    append_parts: 附加网格对象序列。
    """
    return any(part is not None for part in append_parts)


def build_combined_mesh_data(head_obj, append_parts) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把头部和附加网格合并成单个顶点、面和颜色数组。

    head_obj: 已带变形网格的头部临时对象。
    append_parts: 附加网格对象序列。
    """
    head_vertices, head_faces = get_mesh_data(head_obj)
    head_colors = np.tile(np.asarray(SKIN_TONE, dtype=np.float64), (len(head_vertices), 1))
    vertices_list = [head_vertices]
    faces_list = [head_faces]
    colors_list = [head_colors]
    offset = len(head_vertices)

    for part_obj in append_parts:
        if part_obj is None:
            continue
        part_mesh = get_modified_mesh(part_obj)
        try:
            slot_materials = part_obj.data.materials
            if is_eye_mesh_object(part_obj):
                vertex_colors = mesh_vertex_colors_dominant_material(part_mesh, slot_materials)
            else:
                vertex_colors = mesh_vertex_colors_from_materials(part_mesh, slot_materials)
            part_vertices = vertices_coords_numpy(part_mesh.vertices)
            part_faces = np.array([tuple(poly.vertices) for poly in part_mesh.polygons], dtype=int)
        finally:
            bpy.data.meshes.remove(part_mesh)

        if len(vertex_colors) != len(part_vertices):
            vertex_colors = np.tile(np.asarray(SKIN_TONE, dtype=np.float64), (len(part_vertices), 1))

        vertices_list.append(part_vertices)
        faces_list.append(part_faces + offset)
        colors_list.append(vertex_colors)
        offset += len(part_vertices)

    vertices = np.concatenate(vertices_list, axis=0)
    faces = np.concatenate(faces_list, axis=0)
    colors = np.concatenate(colors_list, axis=0)
    return vertices, faces, colors


def build_frame_payload(head_obj, append_parts) -> dict[str, np.ndarray]:
    """从头部对象和附加网格构建预览帧字典。

    head_obj: 已带变形网格的头部临时对象。
    append_parts: 附加网格对象序列。
    """
    if has_extra_meshes_for_view(append_parts):
        vertices, faces, vertex_colors = build_combined_mesh_data(head_obj, append_parts)
        landmarks = extract_default_landmarks(vertices)
        return {
            "vertices": vertices,
            "faces": faces,
            "vertex_colors": vertex_colors,
            **landmarks,
        }

    vertices, faces = get_mesh_data(head_obj)
    landmarks = extract_default_landmarks(vertices)
    return {"vertices": vertices, "faces": faces, **landmarks}
