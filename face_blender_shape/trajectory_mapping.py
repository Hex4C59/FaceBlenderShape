"""平面轨迹 (x, y) 到 SRanipal 37 维系数的启发式映射，供 mock / 标定联调。"""

from __future__ import annotations

import numpy as np

from face_blender_shape.core.blendshape_schema import BLENDSHAPE_INDEX, FRAME_WIDTH


def mock_trajectory_from_mock_audio(
    n_frames: int,
    fps: float,
    *,
    audio_freq_hz: float = 2.0,
    carrier_hz: float = 0.7,
) -> np.ndarray:
    """
    用简单正弦包络模拟「语音能量」，再生成归一化平面轨迹，便于不接真实音频时联调。

    n_frames: 总帧数。
    fps: 帧率，用于把时间轴换成秒。
    audio_freq_hz: 包络变化频率（模拟语速/音节）。
    carrier_hz: 轨迹在平面内转动的载波频率。
    返回: 形状 (n_frames, 2) 的 float64，每行 (x, y) 约在 [0, 1]。
    """
    t = np.arange(n_frames, dtype=np.float64) / max(float(fps), 1e-6)
    env = 0.5 + 0.5 * np.sin(2.0 * np.pi * float(audio_freq_hz) * t)
    phase = 2.0 * np.pi * float(carrier_hz) * t
    x = 0.5 + 0.28 * env * np.sin(phase)
    y = 0.5 + 0.28 * env * np.cos(phase * 1.3)
    return np.stack([x, y], axis=1)


def trajectory_to_sranipal_frames(
    xy: np.ndarray,
    *,
    ref_xy: np.ndarray | None = None,
    scale_xy: tuple[float, float] | None = None,
    gain: float = 1.0,
    long_gain: float = 0.9,
    jaw_open_base: float = 0.38,
    jaw_open_k: float = 0.55,
    y_up_positive: bool = False,
) -> np.ndarray:
    """
    将归一化或像素级轨迹映射为逐帧 SRanipal 向量（主要写舌头 11 维 + 可选下颌张开）。

    xy: 形状 (N, 2)，每行 (x, y)；若未与 ref 相减，建议已是图像归一化坐标。
    ref_xy: 参考点，长度 2；None 时用 xy 的均值。
    scale_xy: (sx, sy) 用于把位移除到约 [-1, 1]；None 时用 xy 相对 ref 的绝对值最大半幅。
    gain: 舌头主方向通道总增益，乘在 max(0,±dx), max(0,±dy) 上。
    long_gain: 径向伸出 Tongue_LongStep1/2 的增益。
    jaw_open_base: 基础 Jaw_Open，便于侧视观察舌头（模拟张嘴）。
    jaw_open_k: 由轨迹径向幅度叠加到 Jaw_Open 上的系数。
    y_up_positive: True 表示 y 增大对应解剖「上」；False（默认）表示图像坐标 y 向下增大，
        此时 Tongue_Up 用 -dy 正部、Tongue_Down 用 +dy 正部。
    返回: 形状 (N, FRAME_WIDTH)，每行一帧，列顺序同 BLENDSHAPE_NAMES。
    """
    p = np.asarray(xy, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"xy 须为 (N, 2)，当前 {p.shape}")

    if ref_xy is None:
        r0 = p.mean(axis=0)
    else:
        r0 = np.asarray(ref_xy, dtype=np.float64).reshape(2)

    d = p - r0
    if scale_xy is not None:
        sx, sy = float(scale_xy[0]), float(scale_xy[1])
        sx = max(sx, 1e-9)
        sy = max(sy, 1e-9)
    else:
        ax = np.maximum(np.abs(d[:, 0]).max(), 1e-9)
        ay = np.maximum(np.abs(d[:, 1]).max(), 1e-9)
        sx, sy = ax, ay

    nx = np.clip(d[:, 0] / sx, -1.0, 1.0)
    ny = np.clip(d[:, 1] / sy, -1.0, 1.0)

    g = max(float(gain), 0.0)
    w_r = g * np.clip(nx, 0.0, 1.0)
    w_l = g * np.clip(-nx, 0.0, 1.0)
    if y_up_positive:
        w_u = g * np.clip(ny, 0.0, 1.0)
        w_d = g * np.clip(-ny, 0.0, 1.0)
    else:
        w_u = g * np.clip(-ny, 0.0, 1.0)
        w_d = g * np.clip(ny, 0.0, 1.0)

    r = np.sqrt(np.clip(nx * nx + ny * ny, 0.0, 2.0))
    lg = max(float(long_gain), 0.0)
    long1 = np.clip(lg * r, 0.0, 1.0)
    long2 = np.clip(lg * np.maximum(r - 0.45, 0.0) * 2.0, 0.0, 1.0)

    k_corner = 1.25
    ul = np.clip(k_corner * w_u * w_l, 0.0, 1.0)
    ur = np.clip(k_corner * w_u * w_r, 0.0, 1.0)
    dl = np.clip(k_corner * w_d * w_l, 0.0, 1.0)
    dr = np.clip(k_corner * w_d * w_r, 0.0, 1.0)

    n = p.shape[0]
    out = np.zeros((n, FRAME_WIDTH), dtype=np.float64)
    ji = BLENDSHAPE_INDEX["Jaw_Open"]
    out[:, ji] = np.clip(
        float(jaw_open_base) + float(jaw_open_k) * r,
        0.0,
        1.0,
    )

    ti = BLENDSHAPE_INDEX
    out[:, ti["Tongue_LongStep1"]] = long1
    out[:, ti["Tongue_LongStep2"]] = long2
    out[:, ti["Tongue_Left"]] = w_l
    out[:, ti["Tongue_Right"]] = w_r
    out[:, ti["Tongue_Up"]] = w_u
    out[:, ti["Tongue_Down"]] = w_d
    out[:, ti["Tongue_UpLeft_Morph"]] = ul
    out[:, ti["Tongue_UpRight_Morph"]] = ur
    out[:, ti["Tongue_DownLeft_Morph"]] = dl
    out[:, ti["Tongue_DownRight_Morph"]] = dr

    return out


def save_sranipal_csv(path: str, frames: np.ndarray) -> None:
    """
    将 (N, FRAME_WIDTH) 写入无表头逗号分隔 CSV，与现有 preview 输入一致。

    path: 输出文件路径。
    frames: 每行一帧的系数矩阵。
    """
    arr = np.asarray(frames, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != FRAME_WIDTH:
        raise ValueError(f"frames 形状须为 (N, {FRAME_WIDTH})")
    np.savetxt(path, arr, delimiter=",", fmt="%.4f")
