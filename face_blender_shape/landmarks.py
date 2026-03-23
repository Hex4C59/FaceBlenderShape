from __future__ import annotations

import numpy as np

TONGUE_SLICE = slice(180, 314)
LIP_RIGHT_SLICE = slice(5977, 6015)
LIP_LEFT_SLICE = slice(7359, 7397)
CHEEK_RIGHT_SLICE = slice(6341, 6381)
CHEEK_LEFT_SLICE = slice(7723, 7763)


def _coerce_vertices(vertices: np.ndarray | tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    if isinstance(vertices, tuple):
        return vertices[0]
    return np.asarray(vertices)


def get_lip_vertices(vertices: np.ndarray | tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    verts = _coerce_vertices(vertices)
    return np.concatenate([verts[LIP_RIGHT_SLICE], verts[LIP_LEFT_SLICE]], axis=0)


def get_tongue_vertices(vertices: np.ndarray | tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    verts = _coerce_vertices(vertices)
    return verts[TONGUE_SLICE]


def get_cheek_vertices(vertices: np.ndarray | tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    verts = _coerce_vertices(vertices)
    return np.concatenate([verts[CHEEK_RIGHT_SLICE], verts[CHEEK_LEFT_SLICE]], axis=0)


def get_tongue_tip(tongue_vertices: np.ndarray) -> np.ndarray:
    tongue = np.asarray(tongue_vertices)
    return tongue[59:60, :]


def get_cheek_keypoints(cheek_vertices: np.ndarray) -> np.ndarray:
    cheek = np.asarray(cheek_vertices)
    return np.stack([cheek[29, :], cheek[69, :]], axis=0)


def extract_default_landmarks(vertices: np.ndarray | tuple[np.ndarray, np.ndarray]) -> dict[str, np.ndarray]:
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
