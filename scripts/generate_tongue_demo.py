"""生成舌头相关 Blendshape 演示序列，写入 outputs/tongue_demo.csv。"""
import sys
from pathlib import Path

import numpy as np

# 项目根目录，用于以脚本方式运行时能 import face_blender_shape
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from face_blender_shape.constants import BLENDSHAPE_INDEX, FRAME_WIDTH
from face_blender_shape.paths import OUTPUTS_DIR


def base_frame(jaw_open: float = 0.18, mouth_ape: float = 0.04) -> np.ndarray:
    """构建单帧基础口型（下巴与嘴型），不含舌头通道。

    jaw_open: Jaw_Open 通道权重。
    mouth_ape: Mouth_Ape_Shape 通道权重。
    """
    frame = np.zeros(FRAME_WIDTH, dtype=float)
    frame[BLENDSHAPE_INDEX["Jaw_Open"]] = jaw_open
    frame[BLENDSHAPE_INDEX["Mouth_Ape_Shape"]] = mouth_ape
    return frame


def open_mouth_frame(jaw_open: float, mouth_ape: float) -> np.ndarray:
    """在基础帧上叠加张口、唇角与口内等通道，便于后续叠加舌头动画。

    jaw_open: Jaw_Open 通道权重。
    mouth_ape: Mouth_Ape_Shape 通道权重。
    """
    frame = base_frame(jaw_open, mouth_ape)
    frame[BLENDSHAPE_INDEX["Mouth_Upper_UpLeft"]] = 0.18
    frame[BLENDSHAPE_INDEX["Mouth_Upper_UpRight"]] = 0.18
    frame[BLENDSHAPE_INDEX["Mouth_Lower_DownLeft"]] = 0.32
    frame[BLENDSHAPE_INDEX["Mouth_Lower_DownRight"]] = 0.32
    frame[BLENDSHAPE_INDEX["Mouth_Upper_Inside"]] = 0.12
    frame[BLENDSHAPE_INDEX["Mouth_Lower_Inside"]] = 0.12
    return frame


def eased(n: int) -> np.ndarray:
    """返回 smoothstep 缓动曲线在 [0,1] 上的 n 个采样点。

    n: 采样个数。
    """
    t = np.linspace(0.0, 1.0, n)
    return t * t * (3.0 - 2.0 * t)


def segment_open_mouth(n: int) -> np.ndarray:
    """从微张口插值到大张口，共 n 帧。

    n: 该段帧数。
    """
    start = base_frame(0.12, 0.02)
    end = open_mouth_frame(0.72, 0.22)
    t = eased(n)[:, None]
    return start[None, :] * (1.0 - t) + end[None, :] * t


def segment_extend(n: int) -> np.ndarray:
    """保持大张嘴，舌头伸出（LongStep、Up）随缓动渐强，共 n 帧。

    n: 该段帧数。
    """
    seq = np.repeat(open_mouth_frame(0.74, 0.24)[None, :], n, axis=0)
    t = eased(n)
    seq[:, BLENDSHAPE_INDEX["Tongue_LongStep1"]] = 0.92 * t
    seq[:, BLENDSHAPE_INDEX["Tongue_LongStep2"]] = 0.62 * t
    seq[:, BLENDSHAPE_INDEX["Tongue_Up"]] = 0.16 * t
    return seq


def segment_side_to_side(n: int) -> np.ndarray:
    """大张嘴下舌头左右摆动（正弦驱动 Tongue_Left / Tongue_Right），共 n 帧。

    n: 该段帧数。
    """
    seq = np.repeat(open_mouth_frame(0.74, 0.22)[None, :], n, axis=0)
    wave = np.sin(np.linspace(0.0, 2.0 * np.pi, n))
    seq[:, BLENDSHAPE_INDEX["Tongue_LongStep1"]] = 0.9
    seq[:, BLENDSHAPE_INDEX["Tongue_LongStep2"]] = 0.6
    seq[:, BLENDSHAPE_INDEX["Tongue_Up"]] = 0.14
    seq[:, BLENDSHAPE_INDEX["Tongue_Left"]] = np.clip(wave, 0.0, None) * 0.78
    seq[:, BLENDSHAPE_INDEX["Tongue_Right"]] = np.clip(-wave, 0.0, None) * 0.78
    return seq


def segment_up_down(n: int) -> np.ndarray:
    """大张嘴下舌头上、下摆动（正弦驱动 Tongue_Up / Tongue_Down），共 n 帧。

    n: 该段帧数。
    """
    seq = np.repeat(open_mouth_frame(0.76, 0.24)[None, :], n, axis=0)
    wave = np.sin(np.linspace(0.0, 2.0 * np.pi, n))
    seq[:, BLENDSHAPE_INDEX["Tongue_LongStep1"]] = 0.88
    seq[:, BLENDSHAPE_INDEX["Tongue_LongStep2"]] = 0.56
    seq[:, BLENDSHAPE_INDEX["Tongue_Up"]] = np.clip(wave, 0.0, None) * 0.82
    seq[:, BLENDSHAPE_INDEX["Tongue_Down"]] = np.clip(-wave, 0.0, None) * 0.82
    return seq


def segment_roll(n: int) -> np.ndarray:
    """大张嘴下舌头卷起（Tongue_Roll 半周期正弦），共 n 帧。

    n: 该段帧数。
    """
    seq = np.repeat(open_mouth_frame(0.72, 0.22)[None, :], n, axis=0)
    t = np.sin(np.linspace(0.0, np.pi, n))
    seq[:, BLENDSHAPE_INDEX["Tongue_LongStep1"]] = 0.86
    seq[:, BLENDSHAPE_INDEX["Tongue_LongStep2"]] = 0.54
    seq[:, BLENDSHAPE_INDEX["Tongue_Roll"]] = 0.92 * t
    return seq


def segment_diagonals(n: int) -> np.ndarray:
    """大张嘴下按时间分四段依次激活四个对角舌头 morph，共 n 帧。

    n: 该段帧数。
    """
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
    """从起始单帧用缓动插值回默认微张口，共 n 帧。

    n: 过渡帧数。
    start_frame: 起始一帧的 blendshape 向量，长度须为 FRAME_WIDTH。
    """
    target = base_frame(0.12, 0.02)
    t = eased(n)[:, None]
    return start_frame[None, :] * (1.0 - t) + target[None, :] * t


def build_demo_sequence() -> np.ndarray:
    """按顺序拼接各演示段落并追加收尾过渡，返回完整帧序列。"""
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
    """确保输出目录存在，生成演示序列并写入 tongue_demo.csv。"""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / "tongue_demo.csv"
    data = build_demo_sequence()
    np.savetxt(output_path, data, fmt="%.4f", delimiter=",")
    print(f"wrote {output_path.name} to {output_path} with shape {data.shape}")


if __name__ == "__main__":
    main()
