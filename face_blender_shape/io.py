from __future__ import annotations

from pathlib import Path

import numpy as np

from face_blender_shape.constants import BLENDSHAPE_NAMES, FRAME_WIDTH
from face_blender_shape.paths import resolve_input_csv_path, resolve_output_path


def load_blendshape_csv(path: str | Path) -> np.ndarray:
    resolved = resolve_input_csv_path(path)
    data = np.loadtxt(resolved, delimiter=",")
    data = np.atleast_2d(data)

    if data.shape[1] != FRAME_WIDTH:
        raise ValueError(f"Expected {FRAME_WIDTH} columns, got {data.shape[1]}")

    return data.astype(float, copy=False)


def save_keypoints_npz(
    input_path: str | Path,
    *,
    blendshapes: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    lip: np.ndarray,
    tongue_tip: np.ndarray,
    cheek_keypoints: np.ndarray,
    keypoints: np.ndarray,
    output_path: str | Path | None = None,
) -> Path:
    resolved_output = resolve_output_path(input_path=input_path, output_path=output_path, suffix=".npz")
    np.savez_compressed(
        resolved_output,
        blendshapes=blendshapes,
        vertices=vertices,
        faces=faces,
        lip=lip,
        tongue_tip=tongue_tip,
        cheek_keypoints=cheek_keypoints,
        keypoints=keypoints,
        blendshape_names=np.array(BLENDSHAPE_NAMES),
    )
    return resolved_output
