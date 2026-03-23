import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from face_blender_shape.constants import BLENDSHAPE_INDEX, FRAME_WIDTH
from face_blender_shape.paths import OUTPUTS_DIR


def base_frame(jaw_open: float = 0.18, mouth_ape: float = 0.04) -> np.ndarray:
    frame = np.zeros(FRAME_WIDTH, dtype=float)
    frame[BLENDSHAPE_INDEX["Jaw_Open"]] = jaw_open
    frame[BLENDSHAPE_INDEX["Mouth_Ape_Shape"]] = mouth_ape
    return frame


def open_mouth_frame(jaw_open: float, mouth_ape: float) -> np.ndarray:
    frame = base_frame(jaw_open, mouth_ape)
    frame[BLENDSHAPE_INDEX["Mouth_Upper_UpLeft"]] = 0.18
    frame[BLENDSHAPE_INDEX["Mouth_Upper_UpRight"]] = 0.18
    frame[BLENDSHAPE_INDEX["Mouth_Lower_DownLeft"]] = 0.32
    frame[BLENDSHAPE_INDEX["Mouth_Lower_DownRight"]] = 0.32
    frame[BLENDSHAPE_INDEX["Mouth_Upper_Inside"]] = 0.12
    frame[BLENDSHAPE_INDEX["Mouth_Lower_Inside"]] = 0.12
    return frame


def eased(n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return t * t * (3.0 - 2.0 * t)


def segment_open_mouth(n: int) -> np.ndarray:
    start = base_frame(0.12, 0.02)
    end = open_mouth_frame(0.72, 0.22)
    t = eased(n)[:, None]
    return start[None, :] * (1.0 - t) + end[None, :] * t


def segment_extend(n: int) -> np.ndarray:
    seq = np.repeat(open_mouth_frame(0.74, 0.24)[None, :], n, axis=0)
    t = eased(n)
    seq[:, BLENDSHAPE_INDEX["Tongue_LongStep1"]] = 0.92 * t
    seq[:, BLENDSHAPE_INDEX["Tongue_LongStep2"]] = 0.62 * t
    seq[:, BLENDSHAPE_INDEX["Tongue_Up"]] = 0.16 * t
    return seq


def segment_side_to_side(n: int) -> np.ndarray:
    seq = np.repeat(open_mouth_frame(0.74, 0.22)[None, :], n, axis=0)
    wave = np.sin(np.linspace(0.0, 2.0 * np.pi, n))
    seq[:, BLENDSHAPE_INDEX["Tongue_LongStep1"]] = 0.9
    seq[:, BLENDSHAPE_INDEX["Tongue_LongStep2"]] = 0.6
    seq[:, BLENDSHAPE_INDEX["Tongue_Up"]] = 0.14
    seq[:, BLENDSHAPE_INDEX["Tongue_Left"]] = np.clip(wave, 0.0, None) * 0.78
    seq[:, BLENDSHAPE_INDEX["Tongue_Right"]] = np.clip(-wave, 0.0, None) * 0.78
    return seq


def segment_up_down(n: int) -> np.ndarray:
    seq = np.repeat(open_mouth_frame(0.76, 0.24)[None, :], n, axis=0)
    wave = np.sin(np.linspace(0.0, 2.0 * np.pi, n))
    seq[:, BLENDSHAPE_INDEX["Tongue_LongStep1"]] = 0.88
    seq[:, BLENDSHAPE_INDEX["Tongue_LongStep2"]] = 0.56
    seq[:, BLENDSHAPE_INDEX["Tongue_Up"]] = np.clip(wave, 0.0, None) * 0.82
    seq[:, BLENDSHAPE_INDEX["Tongue_Down"]] = np.clip(-wave, 0.0, None) * 0.82
    return seq


def segment_roll(n: int) -> np.ndarray:
    seq = np.repeat(open_mouth_frame(0.72, 0.22)[None, :], n, axis=0)
    t = np.sin(np.linspace(0.0, np.pi, n))
    seq[:, BLENDSHAPE_INDEX["Tongue_LongStep1"]] = 0.86
    seq[:, BLENDSHAPE_INDEX["Tongue_LongStep2"]] = 0.54
    seq[:, BLENDSHAPE_INDEX["Tongue_Roll"]] = 0.92 * t
    return seq


def segment_diagonals(n: int) -> np.ndarray:
    seq = np.repeat(open_mouth_frame(0.74, 0.22)[None, :], n, axis=0)
    seq[:, BLENDSHAPE_INDEX["Tongue_LongStep1"]] = 0.86
    seq[:, BLENDSHAPE_INDEX["Tongue_LongStep2"]] = 0.52
    block = max(1, n // 4)
    morphs = [
        "Tongue_UpLeft_Morph",
        "Tongue_UpRight_Morph",
        "Tongue_DownLeft_Morph",
        "Tongue_DownRight_Morph",
    ]
    for segment_idx, morph in enumerate(morphs):
        start = segment_idx * block
        end = n if segment_idx == len(morphs) - 1 else min(n, (segment_idx + 1) * block)
        t = np.sin(np.linspace(0.0, np.pi, end - start))
        seq[start:end, BLENDSHAPE_INDEX[morph]] = 0.85 * t
    return seq


def segment_release(n: int, start_frame: np.ndarray) -> np.ndarray:
    target = base_frame(0.12, 0.02)
    t = eased(n)[:, None]
    return start_frame[None, :] * (1.0 - t) + target[None, :] * t


def build_demo_sequence() -> np.ndarray:
    sections = [
        np.repeat(base_frame(0.12, 0.02)[None, :], 10, axis=0),
        segment_open_mouth(26),
        np.repeat(open_mouth_frame(0.72, 0.22)[None, :], 12, axis=0),
        segment_extend(24),
        segment_side_to_side(48),
        segment_up_down(36),
        segment_roll(24),
        segment_diagonals(32),
    ]
    release = segment_release(24, sections[-1][-1])
    sections.append(release)
    return np.vstack(sections)


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / "tongue_demo.csv"
    data = build_demo_sequence()
    np.savetxt(output_path, data, fmt="%.4f", delimiter=",")
    print(f"wrote {output_path.name} to {output_path} with shape {data.shape}")


if __name__ == "__main__":
    main()
