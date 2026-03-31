"""头舌网格拆分与线框边：舌顶点全局下标区间与默认 ``sranipal_head.fbx`` 一致。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# 舌区域顶点在全局顶点数组中的连续下标（与 assets 中 FBX 一致）。
TONGUE_SLICE = slice(180, 314)


def build_tongue_face_mask(
    faces: NDArray[np.int64],
    tongue_lo: int = TONGUE_SLICE.start,
    tongue_hi: int = TONGUE_SLICE.stop,
) -> NDArray[np.bool_]:
    """判断每个三角面是否含舌顶点；含则该面归入舌实体子网格，与头壳线框分离绘制。

    参数:
        faces: 三角面顶点索引，形状 (F, 3)。
        tongue_lo: 舌顶点在全局顶点数组中的起始下标。
        tongue_hi: 舌顶点区间右端（开区间），与 ``TONGUE_SLICE.stop`` 一致。

    返回:
        形状 (F,) 的 bool 数组，True 表示该面至少有一个顶点落在舌下标区间内。
    """
    return np.any((faces >= tongue_lo) & (faces < tongue_hi), axis=1)


def unique_edges_from_faces(faces: NDArray[np.int64]) -> NDArray[np.int64]:
    """从三角面表提取全部无向棱边并去重（下标与 faces 所用一致，可用于紧凑子网格线框）。

    参数:
        faces: 三角面顶点下标，形状 (F, 3)，可为全局或局部重编号后的索引。

    返回:
        形状 (E, 2) 的顶点对，每行一条无向边；无面时返回 (0, 2)。
    """
    if faces.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    ab = np.stack([faces[:, 0], faces[:, 1]], axis=1)
    bc = np.stack([faces[:, 1], faces[:, 2]], axis=1)
    ca = np.stack([faces[:, 2], faces[:, 0]], axis=1)
    edges = np.concatenate([ab, bc, ca], axis=0)
    lo = np.minimum(edges[:, 0], edges[:, 1])
    hi = np.maximum(edges[:, 0], edges[:, 1])
    pairs = np.column_stack([lo, hi])
    return np.unique(pairs, axis=0).astype(np.int64, copy=False)


def compact_mesh_by_face_mask(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
    face_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """按面掩码提取子网格，将顶点重编号为 0..K-1（紧凑存储）。

    参数:
        vertices: 全局顶点坐标，形状 (V, 3)。
        faces: 全局三角面顶点下标，形状 (F, 3)。
        face_mask: 形状 (F,) 的 bool，True 表示保留该三角面。

    返回:
        三元组 ``(new_vertices, new_faces, global_vertex_indices)``：
        ``new_vertices`` 形状 (K, 3)；``new_faces`` 形状 (M, 3)，下标指向 new_vertices；
        ``global_vertex_indices`` 形状 (K,)，``new_vertices[i] == vertices[global_vertex_indices[i]]``。
    """
    sub = faces[face_mask]
    if sub.size == 0:
        empty_v = np.zeros((0, 3), dtype=np.float64)
        empty_f = np.zeros((0, 3), dtype=np.int64)
        empty_i = np.zeros((0,), dtype=np.int64)
        return empty_v, empty_f, empty_i
    flat = sub.ravel()
    global_vertex_indices, inv = np.unique(flat, return_inverse=True)
    new_faces = inv.reshape(sub.shape).astype(np.int64, copy=False)
    v = np.asarray(vertices, dtype=np.float64)
    new_vertices = v[global_vertex_indices]
    return new_vertices, new_faces, global_vertex_indices.astype(np.int64, copy=False)


def split_head_tongue_meshes(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
    tongue_lo: int = TONGUE_SLICE.start,
    tongue_hi: int = TONGUE_SLICE.stop,
) -> tuple[
    tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]],
    tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]],
]:
    """将变形后的整头网格拆成「外壳」与「舌」两个独立紧凑网格（各自顶点表与面表）。

    与 ``build_tongue_face_mask`` 一致：含任一舌顶点的面归舌网格，否则归外壳网格。
    落在脸舌交界上的顶点会在两个子网格中各出现一次（坐标每帧仍一致）。

    参数:
        vertices: 当前帧全部顶点，形状 (V, 3)。
        faces: 三角面下标，形状 (F, 3)。
        tongue_lo: 舌顶点全局下标区间左端。
        tongue_hi: 舌顶点区间右端（开区间）。

    返回:
        ``(shell, tongue)``，其中 ``shell`` / ``tongue`` 均为
        ``(vertices, faces, global_vertex_indices)``，语义同 ``compact_mesh_by_face_mask``。
    """
    mask = build_tongue_face_mask(faces, tongue_lo, tongue_hi)
    shell = compact_mesh_by_face_mask(vertices, faces, ~mask)
    tongue = compact_mesh_by_face_mask(vertices, faces, mask)
    return shell, tongue
