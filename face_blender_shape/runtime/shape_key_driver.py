"""Blendshape 帧校验、映射与形态键写入。"""

from __future__ import annotations

import numpy as np

from face_blender_shape.blendshape_mapping import (
    ARKIT_SHAPE_NAMES,
    SRANIPAL_TONGUE_SHAPE_NAMES,
    SRANIPAL_TO_ARKIT_MATRIX,
    SRANIPAL_TO_ARKIT_MATRIX_EXCLUDING_TONGUE_SOURCES,
    convert_sranipal_to_arkit,
)
from face_blender_shape.core.blendshape_schema import BLENDSHAPE_INDEX, FRAME_WIDTH


def validate_frame(blendshapes: np.ndarray | list[float]) -> np.ndarray:
    """校验输入帧长度并转成一维浮点数组。

    blendshapes: 输入的一帧 blendshape 权重。
    """
    frame = np.asarray(blendshapes, dtype=float).reshape(-1)
    if frame.size != FRAME_WIDTH:
        raise ValueError(f"Expected {FRAME_WIDTH} blendshape values, got {frame.size}")
    return frame


def write_shape_key_values(key_blocks, names: np.ndarray, values: np.ndarray) -> None:
    """把权重写入已存在的形态键。

    key_blocks: `shape_keys.key_blocks` 映射。
    names: 形态键名称数组。
    values: 对应权重数组。
    """
    for name, value in zip(names, values, strict=True):
        if name in key_blocks:
            key_blocks[name].value = float(value)


def resolve_sranipal_mapping(active_obj, append_parts) -> tuple[bool, np.ndarray]:
    """根据场景是否存在 Tongue_* 键决定映射矩阵。

    active_obj: 主头部对象。
    append_parts: 附加网格对象序列。
    """
    to_scan = [active_obj, *append_parts]
    use_direct = False
    for obj in to_scan:
        if obj is None:
            continue
        shape_keys = obj.data.shape_keys
        if shape_keys is None:
            continue
        key_blocks = shape_keys.key_blocks
        if any(name in key_blocks for name in SRANIPAL_TONGUE_SHAPE_NAMES):
            use_direct = True
            break

    if use_direct:
        matrix = SRANIPAL_TO_ARKIT_MATRIX_EXCLUDING_TONGUE_SOURCES
    else:
        matrix = SRANIPAL_TO_ARKIT_MATRIX

    return use_direct, matrix


def apply_direct_sranipal_tongue(
    frame: np.ndarray,
    active_obj,
    append_parts,
    *,
    enabled: bool,
) -> None:
    """把 Tongue_* 通道直接写到场景中的同名形态键。

    frame: 经过校验的一维 SRanipal 帧。
    active_obj: 主头部对象。
    append_parts: 附加网格对象序列。
    enabled: 是否启用直写模式。
    """
    if not enabled:
        return

    for obj in (active_obj, *append_parts):
        if obj is None:
            continue
        shape_keys = obj.data.shape_keys
        if shape_keys is None:
            continue
        key_blocks = shape_keys.key_blocks
        for name in SRANIPAL_TONGUE_SHAPE_NAMES:
            if name in key_blocks:
                key_blocks[name].value = float(frame[BLENDSHAPE_INDEX[name]])


def apply_arkit_shapes(active_obj, arkit_names: np.ndarray, arkit_values: np.ndarray) -> None:
    """把 ARKit 权重写入主头部对象。

    active_obj: 主头部对象。
    arkit_names: ARKit 形态键名称数组。
    arkit_values: ARKit 权重数组。
    """
    shape_keys = active_obj.data.shape_keys
    if shape_keys is None:
        return
    write_shape_key_values(shape_keys.key_blocks, arkit_names, arkit_values)


def apply_arkit_to_secondary_meshes(append_parts, arkit_names: np.ndarray, arkit_values: np.ndarray) -> None:
    """把 ARKit 权重同步到附加网格对象。

    append_parts: 附加网格对象序列。
    arkit_names: ARKit 形态键名称数组。
    arkit_values: ARKit 权重数组。
    """
    for obj in append_parts:
        if obj is None:
            continue
        shape_keys = obj.data.shape_keys
        if shape_keys is None:
            continue
        write_shape_key_values(shape_keys.key_blocks, arkit_names, arkit_values)


def apply_sranipal_frame(
    active_obj,
    append_parts,
    blendshapes: np.ndarray | list[float],
    *,
    matrix: np.ndarray,
    use_direct_sranipal_tongue: bool,
    arkit_names: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """把一帧 SRanipal 权重应用到主对象与附加网格。

    active_obj: 主头部对象。
    append_parts: 附加网格对象序列。
    blendshapes: 输入 SRanipal 帧。
    matrix: SRanipal 到 ARKit 的映射矩阵。
    use_direct_sranipal_tongue: 是否启用 Tongue_* 直写。
    arkit_names: ARKit 形态键名称数组；不传时使用默认列表。
    """
    frame = validate_frame(blendshapes)
    names = np.asarray(ARKIT_SHAPE_NAMES if arkit_names is None else arkit_names)
    arkit_values = convert_sranipal_to_arkit(frame, matrix=matrix)
    apply_arkit_shapes(active_obj, names, arkit_values)
    apply_arkit_to_secondary_meshes(append_parts, names, arkit_values)
    apply_direct_sranipal_tongue(
        frame,
        active_obj,
        append_parts,
        enabled=use_direct_sranipal_tongue,
    )
    return frame, arkit_values
