"""SRanipal（37 维）到 ARKit 的 blendshape 名称映射、稀疏转换矩阵及逐帧/批量权重转换。

多路 SRanipal 形状驱动同一 ARKit 目标时权重相加并裁剪到 [0, 1]。
参考：VRCFaceTracking Unified Expressions、Apple ARFaceAnchor.BlendShapeLocation。
"""

from __future__ import annotations

import numpy as np

from face_blender_shape.constants import BLENDSHAPE_NAMES, FRAME_WIDTH

# SRanipal 37 维中与舌头对应的通道名；用于「直接写形态键」时从 ARKit 映射中整行剔除，避免与 jawOpen 等叠加。
SRANIPAL_TONGUE_SHAPE_NAMES: frozenset[str] = frozenset(
    n for n in BLENDSHAPE_NAMES if n.startswith("Tongue_")
)

# fmt: off
SRANIPAL_TO_ARKIT: dict[str, list[tuple[str, float]]] = {
    # ── Jaw ──
    "Jaw_Left":              [("jawLeft", 1.0)],
    "Jaw_Right":             [("jawRight", 1.0)],
    "Jaw_Forward":           [("jawForward", 1.0)],
    "Jaw_Open":              [("jawOpen", 1.0)],

    # ── Mouth – compound / unique to SRanipal ──
    "Mouth_Ape_Shape":       [("jawOpen", 0.5), ("mouthFunnel", 0.3)],
    "Mouth_Upper_Left":      [("mouthLeft", 0.5)],
    "Mouth_Upper_Right":     [("mouthRight", 0.5)],
    "Mouth_Lower_Left":      [("mouthLeft", 0.5)],
    "Mouth_Lower_Right":     [("mouthRight", 0.5)],
    "Mouth_Upper_Overturn":  [("mouthShrugUpper", 1.0)],
    "Mouth_Lower_Overturn":  [("mouthShrugLower", 1.0)],
    "Mouth_Pout":            [("mouthPucker", 1.0)],

    # ── Mouth – smile / frown ──
    "Mouth_Smile_Left":      [("mouthSmileLeft", 1.0)],
    "Mouth_Smile_Right":     [("mouthSmileRight", 1.0)],
    "Mouth_Sad_Left":        [("mouthFrownLeft", 1.0)],
    "Mouth_Sad_Right":       [("mouthFrownRight", 1.0)],

    # ── Cheek ──
    "Cheek_Puff_Left":       [("cheekPuff", 0.5)],
    "Cheek_Puff_Right":      [("cheekPuff", 0.5)],
    "Cheek_Suck":            [("mouthFunnel", 0.4)],

    # ── Mouth – lip raise / lower ──
    "Mouth_Upper_UpLeft":    [("mouthUpperUpLeft", 1.0)],
    "Mouth_Upper_UpRight":   [("mouthUpperUpRight", 1.0)],
    "Mouth_Lower_DownLeft":  [("mouthLowerDownLeft", 1.0)],
    "Mouth_Lower_DownRight": [("mouthLowerDownRight", 1.0)],

    # ── Mouth – lip roll / tuck ──
    "Mouth_Upper_Inside":    [("mouthRollUpper", 1.0)],
    "Mouth_Lower_Inside":    [("mouthRollLower", 1.0)],
    "Mouth_Lower_Overlay":   [("mouthClose", 1.0)],

    # ── Tongue ──
    # ARKit only has tongueOut; SRanipal has 9 detailed tongue shapes.
    # We map forward-extension shapes to tongueOut; directional shapes
    # are dropped (the MetaHuman model has no matching targets).
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

ARKIT_SHAPE_NAMES: tuple[str, ...] = (
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
    "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft",
    "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight",
    "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
    "jawForward", "jawLeft", "jawRight", "jawOpen",
    "mouthClose", "mouthFunnel", "mouthPucker",
    "mouthRight", "mouthLeft",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight",
)

_ARKIT_INDEX = {name: idx for idx, name in enumerate(ARKIT_SHAPE_NAMES)}


def _build_sparse_matrix(
    *,
    omit_sranipal_sources: frozenset[str] | None = None,
) -> np.ndarray:
    """
    预计算 (37, 51) 的 SRanipal → ARKit 权重矩阵。

    omit_sranipal_sources: 若为 frozenset，则这些 SRanipal 通道不参与映射（整行保持为 0），用于改由运行时直接写 Tongue_* 形态键。
    """
    mat = np.zeros((FRAME_WIDTH, len(ARKIT_SHAPE_NAMES)), dtype=float)
    omit = omit_sranipal_sources or frozenset()
    for src_idx, src_name in enumerate(BLENDSHAPE_NAMES):
        if src_name in omit:
            continue
        for arkit_name, weight in SRANIPAL_TO_ARKIT.get(src_name, []):
            if arkit_name in _ARKIT_INDEX:
                mat[src_idx, _ARKIT_INDEX[arkit_name]] = weight
    return mat


# 默认 SRanipal→ARKit 矩阵；当网格上存在 Tongue_* 形态键时改用 SRANIPAL_TO_ARKIT_MATRIX_EXCLUDING_TONGUE_SOURCES。
SRANIPAL_TO_ARKIT_MATRIX = _build_sparse_matrix()
SRANIPAL_TO_ARKIT_MATRIX_EXCLUDING_TONGUE_SOURCES = _build_sparse_matrix(
    omit_sranipal_sources=SRANIPAL_TONGUE_SHAPE_NAMES,
)


def convert_sranipal_to_arkit(
    sranipal_frame: np.ndarray,
    *,
    matrix: np.ndarray | None = None,
) -> np.ndarray:
    """
    将单帧 SRanipal (37,) 转为 ARKit (51,) 权重。

    sranipal_frame: 一维或可被 reshape 为 (37,) 的数组。
    matrix: 自定义转换矩阵；None 时使用 SRANIPAL_TO_ARKIT_MATRIX。
    """
    mat = SRANIPAL_TO_ARKIT_MATRIX if matrix is None else matrix
    frame = np.asarray(sranipal_frame, dtype=float).reshape(-1)
    arkit = frame @ mat
    return np.clip(arkit, 0.0, 1.0)


def convert_sranipal_batch(
    data: np.ndarray,
    *,
    matrix: np.ndarray | None = None,
) -> np.ndarray:
    """
    将 (N, 37) SRanipal 数据转为 (N, 51) ARKit 权重。

    data: 每行一帧。
    matrix: 自定义转换矩阵；None 时使用 SRANIPAL_TO_ARKIT_MATRIX。
    """
    mat = SRANIPAL_TO_ARKIT_MATRIX if matrix is None else matrix
    arkit = np.asarray(data, dtype=float) @ mat
    return np.clip(arkit, 0.0, 1.0)
