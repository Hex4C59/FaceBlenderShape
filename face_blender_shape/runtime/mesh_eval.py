"""Blender 网格求值与 numpy 转换。"""

from __future__ import annotations

import bmesh  # pyright: ignore[reportMissingModuleSource]
import bpy
import numpy as np


def vertices_coords_numpy(vertices) -> np.ndarray:
    """把 Blender 顶点集合转成坐标数组。

    vertices: `bpy.types.Mesh.vertices` 顶点集合。
    """
    return np.array([tuple(vertex.co) for vertex in vertices], dtype=float)


def get_mesh_data(obj) -> tuple[np.ndarray, np.ndarray]:
    """读取对象当前网格数据块中的顶点和面。

    obj: Blender 网格对象。
    """
    vertices = vertices_coords_numpy(obj.data.vertices)
    faces = np.array([tuple(face.vertices) for face in obj.data.polygons], dtype=int)
    return vertices, faces


def get_modified_mesh(obj, cage: bool = False) -> bpy.types.Mesh:
    """在依赖图上求值对象几何并返回三角化后的新网格。

    obj: Blender 网格对象。
    cage: 是否使用 cage 模式求值。
    """
    bm = bmesh.new()
    bm.from_object(
        obj,
        bpy.context.evaluated_depsgraph_get(),
        cage=cage,
    )
    mesh = bpy.data.meshes.new("Deformed")
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    return mesh
