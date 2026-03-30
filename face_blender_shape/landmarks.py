"""从 SRanipal 拓扑头模顶点序列中按固定下标切片提取唇、舌、脸颊等区域，并生成口腔裁切面掩码。

``TONGUE_SLICE`` 等常量与默认 ``sranipal_head.fbx`` 网格顶点顺序绑定；更换模型拓扑时须同步修改下标。
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

VerticesInput = NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]


class LandmarkBundle(TypedDict):
    """单帧网格上的默认关键点集合（不含顶点和面索引）。"""

    lip: NDArray[np.float64]
    tongue: NDArray[np.float64]
    cheek: NDArray[np.float64]
    tongue_tip: NDArray[np.float64]
    cheek_keypoints: NDArray[np.float64]
    keypoints: NDArray[np.float64]


# 默认 SRanipal 头模：全头顶点数组中的连续下标区间（与 assets 中 FBX 一致）。
TONGUE_SLICE = slice(180, 314)
LIP_RIGHT_SLICE = slice(5977, 6015)
LIP_LEFT_SLICE = slice(7359, 7397)
CHEEK_RIGHT_SLICE = slice(6341, 6381)
CHEEK_LEFT_SLICE = slice(7723, 7763)

# 舌 / 脸颊子网格内的局部顶点下标（拼接顺序与 ``get_*_vertices`` 输出一致）。
_TONGUE_TIP_ROW = 59
_CHEEK_KP_A, _CHEEK_KP_B = 29, 69


def _coerce_vertices(vertices: VerticesInput) -> NDArray[np.float64]:
    """从顶点或 (顶点, 法向等) 元组中取出顶点数组。

    参数:
        vertices: 形状 (V, 3) 的顶点坐标，或首元素为上述数组的元组。

    返回:
        float64 的 (V, 3) 数组。
    """
    if isinstance(vertices, tuple):
        return vertices[0]
    return np.asarray(vertices, dtype=np.float64)


def get_lip_vertices(vertices: VerticesInput) -> NDArray[np.float64]:
    """提取唇部相关顶点（左右唇区域拼接）。

    参数:
        vertices: 全头网格顶点或 ``VerticesInput`` 元组。

    返回:
        形状 (N, 3) 的顶点坐标。
    """
    verts = _coerce_vertices(vertices)
    return np.concatenate([verts[LIP_RIGHT_SLICE], verts[LIP_LEFT_SLICE]], axis=0)


def get_tongue_vertices(vertices: VerticesInput) -> NDArray[np.float64]:
    """提取舌头区域顶点。

    参数:
        vertices: 全头网格顶点或 ``VerticesInput`` 元组。

    返回:
        形状 (TONGUE_SLICE 长度, 3) 的顶点坐标。
    """
    verts = _coerce_vertices(vertices)
    return verts[TONGUE_SLICE]


def get_cheek_vertices(vertices: VerticesInput) -> NDArray[np.float64]:
    """提取脸颊区域顶点（左右脸颊拼接）。

    参数:
        vertices: 全头网格顶点或 ``VerticesInput`` 元组。

    返回:
        形状 (N, 3) 的顶点坐标。
    """
    verts = _coerce_vertices(vertices)
    return np.concatenate([verts[CHEEK_RIGHT_SLICE], verts[CHEEK_LEFT_SLICE]], axis=0)


def get_tongue_tip(tongue_vertices: NDArray[np.float64]) -> NDArray[np.float64]:
    """取舌尖代表点（单点）。

    参数:
        tongue_vertices: ``get_tongue_vertices`` 输出的舌区域顶点。

    返回:
        形状 (1, 3) 的 float64 数组。
    """
    tongue = np.asarray(tongue_vertices, dtype=np.float64)
    return tongue[_TONGUE_TIP_ROW : _TONGUE_TIP_ROW + 1, :]


def get_cheek_keypoints(cheek_vertices: NDArray[np.float64]) -> NDArray[np.float64]:
    """从脸颊顶点中选取两个代表点。

    参数:
        cheek_vertices: ``get_cheek_vertices`` 输出的脸颊区域顶点。

    返回:
        形状 (2, 3) 的 float64 数组。
    """
    cheek = np.asarray(cheek_vertices, dtype=np.float64)
    return np.stack([cheek[_CHEEK_KP_A, :], cheek[_CHEEK_KP_B, :]], axis=0)


def extract_default_landmarks(vertices: VerticesInput) -> LandmarkBundle:
    """从整头网格提取默认关键点集合。

    参数:
        vertices: 全头网格顶点或 ``VerticesInput`` 元组。

    返回:
        含 lip、tongue、cheek、tongue_tip、cheek_keypoints 及拼接后的 keypoints。
    """
    lip = get_lip_vertices(vertices)
    tongue = get_tongue_vertices(vertices)
    cheek = get_cheek_vertices(vertices)
    tongue_tip = get_tongue_tip(tongue)
    cheek_keypoints = get_cheek_keypoints(cheek)
    keypoints = np.concatenate([lip, tongue_tip, cheek_keypoints], axis=0)
    return {
        "lip": lip,
        "tongue": tongue,
        "cheek": cheek,
        "tongue_tip": tongue_tip,
        "cheek_keypoints": cheek_keypoints,
        "keypoints": keypoints,
    }


def build_mouth_removal_mask(
    faces: NDArray[np.int64],
    vertices: NDArray[np.float64],
    *,
    margin: float = 0.01,
) -> NDArray[np.bool_]:
    """以舌头顶点包围盒为参考，标记要保留的三角面（侧面「开窗」观察口腔）。

    落在舌盒内且不含舌顶点的面视为牙齿/口腔内壁等并剔除；含舌顶点的面始终保留。

    参数:
        faces: 三角面顶点索引，形状 (F, 3)。
        vertices: 顶点坐标，形状 (V, 3)。
        margin: 包围盒各轴向外扩展的量（模型空间）；越大裁切窗口越宽松。

    返回:
        形状 (F,) 的 bool 数组，``True`` 表示保留该三角面。
    """
    tongue_lo, tongue_hi = TONGUE_SLICE.start, TONGUE_SLICE.stop
    tongue_verts = vertices[tongue_lo:tongue_hi]

    bbox_min = tongue_verts.min(axis=0) - margin
    bbox_max = tongue_verts.max(axis=0) + margin

    centroids = vertices[faces].mean(axis=1)
    inside_bbox = np.all((centroids >= bbox_min) & (centroids <= bbox_max), axis=1)

    is_tongue = np.any((faces >= tongue_lo) & (faces < tongue_hi), axis=1)

    return ~inside_bbox | is_tongue
