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
    """判断每个三角面是否含舌顶点；含则该面归入舌实体子网格，与头壳线框分离绘制。"""
    return np.any((faces >= tongue_lo) & (faces < tongue_hi), axis=1)


def unique_edges_from_faces(faces: NDArray[np.int64]) -> NDArray[np.int64]:
    """从三角面表提取全部无向棱边并去重（下标与 faces 所用一致，可用于紧凑子网格线框）。"""
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
    """按面掩码提取子网格，将顶点重编号为 0..K-1（紧凑存储）。"""
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
    """将变形后的整头网格拆成「外壳」与「舌」两个独立紧凑网格（各自顶点表与面表）。"""
    mask = build_tongue_face_mask(faces, tongue_lo, tongue_hi)
    shell = compact_mesh_by_face_mask(vertices, faces, ~mask)
    tongue = compact_mesh_by_face_mask(vertices, faces, mask)
    return shell, tongue
