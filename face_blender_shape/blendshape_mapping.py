"""SRanipal（37 维）到 ARKit 命名空间权重的线性映射。

每个 SRanipal 通道可对应一个或多个 ARKit 目标，并带独立系数。
多个 SRanipal 通道驱动同一 ARKit 目标时，贡献在矩阵中先累加，再在
``convert_sranipal_to_arkit`` 中对结果按元素裁剪到 [0, 1]。

参考：VRCFaceTracking Unified Expressions、Apple ARKit ``ARFaceAnchor.BlendShapeLocation``。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from face_blender_shape.constants import BLENDSHAPE_NAMES, FRAME_WIDTH

# SRanipal 形状名 → [(ARKit 形状名, 权重), ...]；空列表表示当前管线不驱动任何 ARKit 通道。
# fmt: off
SRANIPAL_TO_ARKIT: dict[str, list[tuple[str, float]]] = {
    # ── 下颌 ──
    "Jaw_Left":              [("jawLeft", 1.0)],
    "Jaw_Right":             [("jawRight", 1.0)],
    "Jaw_Forward":           [("jawForward", 1.0)],
    "Jaw_Open":              [("jawOpen", 1.0)],

    # ── 嘴部：复合 / SRanipal 特有 ──
    "Mouth_Ape_Shape":       [("jawOpen", 0.5), ("mouthFunnel", 0.3)],
    "Mouth_Upper_Left":      [("mouthLeft", 0.5)],
    "Mouth_Upper_Right":     [("mouthRight", 0.5)],
    "Mouth_Lower_Left":      [("mouthLeft", 0.5)],
    "Mouth_Lower_Right":     [("mouthRight", 0.5)],
    "Mouth_Upper_Overturn":  [("mouthShrugUpper", 1.0)],
    "Mouth_Lower_Overturn":  [("mouthShrugLower", 1.0)],
    "Mouth_Pout":            [("mouthPucker", 1.0)],

    # ── 嘴部：笑 / 愁 ──
    "Mouth_Smile_Left":      [("mouthSmileLeft", 1.0)],
    "Mouth_Smile_Right":     [("mouthSmileRight", 1.0)],
    "Mouth_Sad_Left":        [("mouthFrownLeft", 1.0)],
    "Mouth_Sad_Right":       [("mouthFrownRight", 1.0)],

    # ── 脸颊 ──
    "Cheek_Puff_Left":       [("cheekPuff", 0.5)],
    "Cheek_Puff_Right":      [("cheekPuff", 0.5)],
    "Cheek_Suck":            [("mouthFunnel", 0.4)],

    # ── 嘴部：上抬 / 下压 ──
    "Mouth_Upper_UpLeft":    [("mouthUpperUpLeft", 1.0)],
    "Mouth_Upper_UpRight":   [("mouthUpperUpRight", 1.0)],
    "Mouth_Lower_DownLeft":  [("mouthLowerDownLeft", 1.0)],
    "Mouth_Lower_DownRight": [("mouthLowerDownRight", 1.0)],

    # ── 嘴部：内卷 / 闭合辅助 ──
    "Mouth_Upper_Inside":    [("mouthRollUpper", 1.0)],
    "Mouth_Lower_Inside":    [("mouthRollLower", 1.0)],
    "Mouth_Lower_Overlay":   [("mouthClose", 1.0)],

    # ── 舌头 ──
    # ARKit 仅有 tongueOut 等少量舌相关通道；伸舌步进用 jawOpen 弱耦合近似。
    # 左右/上下等细分舌形在 MetaHuman 无对应 shape key，故映射为空。
    "Tongue_LongStep1":      [("jawOpen", 0.15)],
    "Tongue_LongStep2":      [("jawOpen", 0.15)],
    "Tongue_Left":           [],
    "Tongue_Right":          [],
    "Tongue_Up":             [],
    "Tongue_Down":           [],
    "Tongue_Roll":           [],
    "Tongue_UpLeft_Morph":   [],
    "Tongue_UpRight_Morph":  [],
    "Tongue_DownLeft_Morph": [],
    "Tongue_DownRight_Morph":[],
}
# fmt: on

# 与目标 FBX（如 MetaHuman）中 ARKit 顺序 shape key 名称一致，长度须与矩阵列数相同。
ARKIT_SHAPE_NAMES: tuple[str, ...] = (
    "eyeBlinkLeft",
    "eyeLookDownLeft",
    "eyeLookInLeft",
    "eyeLookOutLeft",
    "eyeLookUpLeft",
    "eyeSquintLeft",
    "eyeWideLeft",
    "eyeBlinkRight",
    "eyeLookDownRight",
    "eyeLookInRight",
    "eyeLookOutRight",
    "eyeLookUpRight",
    "eyeSquintRight",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawRight",
    "jawOpen",
    "mouthClose",
    "mouthFunnel",
    "mouthPucker",
    "mouthRight",
    "mouthLeft",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "noseSneerLeft",
    "noseSneerRight",
)

# ARKit 名称 → 列下标，供填稀疏转换矩阵时 O(1) 查列。
_ARKIT_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(ARKIT_SHAPE_NAMES)}


def _build_sparse_matrix() -> NDArray[np.float64]:
    """
    预计算形状 (FRAME_WIDTH, len(ARKIT_SHAPE_NAMES)) 的转换矩阵：左乘 SRanipal 行向量得 ARKit 权重行向量。

    参数:
        无。

    返回:
        float64 二维数组；元素为 SRanipal 通道到 ARKit 通道的线性系数。
    """
    mat = np.zeros((FRAME_WIDTH, len(ARKIT_SHAPE_NAMES)), dtype=np.float64)
    for src_idx, src_name in enumerate(BLENDSHAPE_NAMES):
        for arkit_name, weight in SRANIPAL_TO_ARKIT.get(src_name, []):
            if arkit_name in _ARKIT_INDEX:
                mat[src_idx, _ARKIT_INDEX[arkit_name]] = weight
    return mat


_CONVERSION_MATRIX: NDArray[np.float64] = _build_sparse_matrix()


def convert_sranipal_to_arkit(sranipal_frame: ArrayLike) -> NDArray[np.float64]:
    """
    将单帧 SRanipal 权重（长度 FRAME_WIDTH）转为 ARKit 顺序的权重向量。

    参数:
        sranipal_frame: 可转为 1D 的输入（如 list、ndarray）；展平后长度须为 FRAME_WIDTH，否则与转换矩阵相乘会形状不匹配。

    返回:
        形状 (len(ARKIT_SHAPE_NAMES),) 的 float64 向量，各分量已裁剪到 [0, 1]。
    """
    frame = np.asarray(sranipal_frame, dtype=np.float64).reshape(-1)
    arkit = frame @ _CONVERSION_MATRIX
    return np.clip(arkit, 0.0, 1.0).astype(np.float64, copy=False)


def convert_sranipal_batch(data: ArrayLike) -> NDArray[np.float64]:
    """
    将多帧 SRanipal 数据批量映射为 ARKit 权重矩阵。

    参数:
        data: 可转为二维数组；行为帧，列为 FRAME_WIDTH（即每行 37 个 SRanipal 权重）。

    返回:
        形状 (N, len(ARKIT_SHAPE_NAMES)) 的 float64 矩阵，N 为帧数，元素已裁剪到 [0, 1]。
    """
    arkit = np.asarray(data, dtype=np.float64) @ _CONVERSION_MATRIX
    return np.clip(arkit, 0.0, 1.0).astype(np.float64, copy=False)
