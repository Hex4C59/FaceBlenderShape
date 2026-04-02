"""将 tongue_regions.csv（舌背矢状面三点 2D 轨迹）启发式映射为
52 列 blendshape CSV（首行为 BLENDSHAPE_NAMES 列名），可直接喂给本项目的 preview 管线。

输入语义
--------
- ``root`` / ``body`` / ``tip``：沿舌背（超声矢状切面）由后向前的三个采样点，列名沿用
  ``root_* / body_* / tip_*``，与追踪管线一致。

映射思路
--------
- root→tip 距离  →  Tongue_LongStep1 / LongStep2（舌背前伸在 2D 上的投影幅度）
- tip 相对 root 的仰角变化  →  Tongue_Up / Tongue_Down
- body 相对 root→tip 弦的拱起（两头低、背峰高）→ 主要并入 Tongue_Up（Roll/对称 Up morph
  在该 FBX 上易把背中拉成凹槽，故拱起不再强写 Roll）
- 垂距幅值 → 仅作 Tongue_Roll 的弱输入（与拱起解耦）
- root 的垂直位移（下沉≈张嘴）  →  Jaw_Open 等口型通道

局限
----
- 仅舌背一条 2D 曲线，不是完整舌体 3D；左右运动不可观 → Tongue_Left/Right 恒为 0
- 超声矢状面侧视，坐标→blendshape 为启发式，不等同于真实 SRAnipal 采集
- 生成 CSV 时按嘴张程度收缩舌通道，减轻预览时舌穿出嘴唇（可用 CLI 关闭）
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from numpy.typing import NDArray

from face_blender_shape.constants import BLENDSHAPE_INDEX, BLENDSHAPE_NAMES, FRAME_WIDTH

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_TONGUE_REGIONS_CSV = PROJECT_ROOT / "data" / "tongue_regions.csv"

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_tongue_regions(csv_path: str | Path) -> dict[str, NDArray[np.float64]]:
    """读取 tongue_regions.csv，按 video 分组返回坐标数组。

    csv_path: 输入 CSV 文件路径。
    返回: {video_id: ndarray(N, 6)}，列为舌背三点：root、body、tip 的 x/y。
    """
    groups: dict[str, list[list[float]]] = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["video"]
            coords = [
                float(row["root_x"]),
                float(row["root_y"]),
                float(row["body_x"]),
                float(row["body_y"]),
                float(row["tip_x"]),
                float(row["tip_y"]),
            ]
            groups.setdefault(vid, []).append(coords)
    return {vid: np.asarray(rows, dtype=np.float64) for vid, rows in groups.items()}


# ---------------------------------------------------------------------------
# 几何特征
# ---------------------------------------------------------------------------


def compute_features(coords: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    """从舌背三点坐标提取帧级几何特征（均为 shape (N,)）。

    coords: shape (N, 6)，列序 root、body、tip 各 (x, y)，点为矢状舌背轨迹。
    返回: 各几何特征字典。
    """
    root = coords[:, 0:2]
    body = coords[:, 2:4]
    tip = coords[:, 4:6]

    # root→tip：舌背主弦长与朝向（2D 投影）
    ext_vec = tip - root
    # 沿着（N，2），第1维求欧式距离
    ext_len = np.linalg.norm(ext_vec, axis=1)

    # root→tip 向量的仰角（弧度），y 轴向下时负角 = tip 在 root 上方
    angle = np.arctan2(-(ext_vec[:, 1]), ext_vec[:, 0])

    # body 偏离 root→tip 连线的有符号垂距（2D 叉积）
    line_dir = ext_vec / (ext_len[:, None] + 1e-8)
    rb = body - root
    curvature = line_dir[:, 0] * rb[:, 1] - line_dir[:, 1] * rb[:, 0]

    # 舌背相对弦「抬高」：图像 y 向下时，拱起则 body 在弦上方 → on_chord_y > body_y
    t = np.sum(rb * line_dir, axis=1) / (ext_len + 1e-8)
    t = np.clip(t, 0.0, 1.0)
    on_chord = root + t[:, None] * ext_vec
    dorsum_arch = on_chord[:, 1] - body[:, 1]

    # root 垂直位移（向下越大≈张嘴程度越大）
    root_y = root[:, 1]

    return {
        "ext_len": ext_len,
        "angle": angle,
        "curvature": curvature,
        "dorsum_arch": dorsum_arch,
        "root_y": root_y,
    }


# ---------------------------------------------------------------------------
# 归一化 & 平滑
# ---------------------------------------------------------------------------


def norm_deviation(
    arr: NDArray[np.float64], scale: float = 1.0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """以中位数为中性，将正/负偏差分别归一化到 [0, 1] 并乘以 scale。

    arr: 输入一维数组。
    scale: 最大输出值。
    返回: (positive_part, negative_part)，各 shape 同 arr。
    """
    med = np.median(arr)
    dev = arr - med
    pos = np.clip(dev, 0, None)
    neg = np.clip(-dev, 0, None)

    def _norm(a: NDArray[np.float64]) -> NDArray[np.float64]:
        hi = np.percentile(a, 97) if a.max() > 0 else 1.0
        return np.clip(a / (hi + 1e-8), 0.0, 1.0) * scale

    return _norm(pos), _norm(neg)


def norm_range(
    arr: NDArray[np.float64], lo_pct: float = 3, hi_pct: float = 97
) -> NDArray[np.float64]:
    """按百分位归一化到 [0, 1]。

    arr: 输入一维数组。
    lo_pct: 下界百分位。
    hi_pct: 上界百分位。
    """
    lo, hi = np.percentile(arr, lo_pct), np.percentile(arr, hi_pct)
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def smooth(arr: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
    """一维高斯平滑（零填充边界），sigma <= 0 时不做处理。

    arr: 输入一维数组。
    sigma: 高斯核标准差（帧数）。
    """
    if sigma <= 0 or len(arr) < 3:
        return arr
    from scipy.ndimage import gaussian_filter1d  # type: ignore[import-untyped]

    return np.asarray(
        gaussian_filter1d(arr, sigma=sigma, mode="nearest"),
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# 核心映射
# ---------------------------------------------------------------------------


def constrain_tongue_to_mouth_room(frames: NDArray[np.float64]) -> None:
    """用嘴张相关通道限制舌权重，使 CSV 驱动预览时舌不易穿出嘴唇（原地修改）。"""
    jaw = frames[:, BLENDSHAPE_INDEX["Jaw_Open"]]
    ape = frames[:, BLENDSHAPE_INDEX["Mouth_Ape_Shape"]]
    room = jaw + ape * 0.5

    ls1_i = BLENDSHAPE_INDEX["Tongue_LongStep1"]
    ls2_i = BLENDSHAPE_INDEX["Tongue_LongStep2"]
    ls1 = frames[:, ls1_i]
    ls2 = frames[:, ls2_i]
    total = ls1 + ls2
    cap_long = room * 0.72
    scale = np.where(total > cap_long, cap_long / (total + 1e-12), 1.0)
    frames[:, ls1_i] = ls1 * scale
    frames[:, ls2_i] = ls2 * scale

    frames[:, BLENDSHAPE_INDEX["Tongue_Up"]] = np.minimum(
        frames[:, BLENDSHAPE_INDEX["Tongue_Up"]], room * 0.78
    )
    frames[:, BLENDSHAPE_INDEX["Tongue_Down"]] = np.minimum(
        frames[:, BLENDSHAPE_INDEX["Tongue_Down"]], room * 0.70
    )
    frames[:, BLENDSHAPE_INDEX["Tongue_Roll"]] = np.minimum(
        frames[:, BLENDSHAPE_INDEX["Tongue_Roll"]], room * 0.72
    )
    for name in (
        "Tongue_UpLeft_Morph",
        "Tongue_UpRight_Morph",
        "Tongue_DownLeft_Morph",
        "Tongue_DownRight_Morph",
    ):
        i = BLENDSHAPE_INDEX[name]
        frames[:, i] = np.minimum(frames[:, i], room * 0.58)


def features_to_blendshapes(
    feats: dict[str, NDArray[np.float64]],
    n: int,
    *,
    sigma: float = 1.5,
    flip_y: bool = False,
    tongue_mouth_constraint: bool = True,
) -> NDArray[np.float64]:
    """将几何特征映射为 52 列 blendshape 权重。

    feats: compute_features 返回的特征字典。
    n: 帧数。
    sigma: 时域高斯平滑核宽度（帧），<= 0 不平滑。
    flip_y: 若为 True 则反转 y 轴方向（默认假设 y 向下 = 图像坐标）。
    tongue_mouth_constraint: 是否按嘴张程度限制舌列（默认 True）。
    返回: shape (n, FRAME_WIDTH) blendshape 数组。
    """
    frames = np.zeros((n, FRAME_WIDTH), dtype=np.float64)

    ext = norm_range(feats["ext_len"])
    angle = feats["angle"] * (-1 if flip_y else 1)
    curv_mag = np.abs(feats["curvature"])
    arch_raw = feats["dorsum_arch"] * (-1 if flip_y else 1)
    arch = np.clip(arch_raw, 0.0, None)
    root_y = feats["root_y"] * (-1 if flip_y else 1)

    arch_n = norm_range(arch)
    damp_long = smooth(1.0 - 0.22 * arch_n, sigma)

    # --- 舌头伸出（拱起时略减前伸，避免尖部过度上挑） ---
    ls1 = smooth(ext * 0.92, sigma) * damp_long
    ls2 = smooth(np.clip((ext - 0.25) / 0.75, 0, 1) * 0.65, sigma) * damp_long
    frames[:, BLENDSHAPE_INDEX["Tongue_LongStep1"]] = ls1
    frames[:, BLENDSHAPE_INDEX["Tongue_LongStep2"]] = ls2

    # --- 背峰：此头模上 Roll/双侧 Up morph 易成「侧高中凹」，拱起改主要由 Tongue_Up 表达 ---
    up_angle, down = norm_deviation(angle, scale=0.40)
    up_combined = np.clip(up_angle + arch_n * 0.72, 0.0, 1.0)
    frames[:, BLENDSHAPE_INDEX["Tongue_Up"]] = smooth(up_combined, sigma)
    frames[:, BLENDSHAPE_INDEX["Tongue_Down"]] = smooth(down, sigma)

    curv_n = norm_range(curv_mag)
    roll = norm_range(curv_mag) * 0.36
    frames[:, BLENDSHAPE_INDEX["Tongue_Roll"]] = smooth(roll, sigma)

    angle_pos, angle_neg = norm_deviation(angle, scale=1.0)
    frames[:, BLENDSHAPE_INDEX["Tongue_UpLeft_Morph"]] = smooth(
        angle_pos * curv_n * 0.22, sigma
    )
    frames[:, BLENDSHAPE_INDEX["Tongue_UpRight_Morph"]] = smooth(
        angle_pos * curv_n * 0.12, sigma
    )
    frames[:, BLENDSHAPE_INDEX["Tongue_DownLeft_Morph"]] = smooth(
        angle_neg * curv_n * 0.14, sigma
    )
    frames[:, BLENDSHAPE_INDEX["Tongue_DownRight_Morph"]] = smooth(
        angle_neg * curv_n * 0.22, sigma
    )

    # --- 张嘴口型：root 越下沉，嘴巴越张 ---
    jaw = norm_range(root_y)
    jaw_open = 0.12 + jaw * 0.50
    frames[:, BLENDSHAPE_INDEX["Jaw_Open"]] = smooth(jaw_open, sigma)
    frames[:, BLENDSHAPE_INDEX["Mouth_Ape_Shape"]] = smooth(jaw * 0.12, sigma)

    # --- 唇部辅助通道，让嘴型更自然 ---
    lip_scale = smooth(jaw, sigma)
    frames[:, BLENDSHAPE_INDEX["Mouth_Upper_UpLeft"]] = lip_scale * 0.10
    frames[:, BLENDSHAPE_INDEX["Mouth_Upper_UpRight"]] = lip_scale * 0.10
    frames[:, BLENDSHAPE_INDEX["Mouth_Lower_DownLeft"]] = lip_scale * 0.22
    frames[:, BLENDSHAPE_INDEX["Mouth_Lower_DownRight"]] = lip_scale * 0.22

    if tongue_mouth_constraint:
        constrain_tongue_to_mouth_room(frames)

    return np.clip(frames, 0.0, 1.0).astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def convert_tongue_regions(
    input_csv: str | Path,
    output_dir: str | Path | None = None,
    *,
    sigma: float = 1.5,
    flip_y: bool = False,
    tongue_mouth_constraint: bool = True,
) -> list[Path]:
    """读取 tongue_regions.csv，为每个 video 片段生成一份 blendshape CSV。

    input_csv: 输入 CSV 路径。
    output_dir: 输出目录，默认 outputs/。
    sigma: 时域平滑参数（帧），<= 0 不平滑。
    flip_y: 是否翻转 y 轴。
    tongue_mouth_constraint: 是否对舌列做口张约束。
    返回: 生成的文件路径列表。
    """
    out = Path(output_dir) if output_dir else OUTPUTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    groups = load_tongue_regions(input_csv)
    paths: list[Path] = []

    for vid, coords in groups.items():
        feats = compute_features(coords)
        bs = features_to_blendshapes(
            feats,
            len(coords),
            sigma=sigma,
            flip_y=flip_y,
            tongue_mouth_constraint=tongue_mouth_constraint,
        )
        dst = out / f"{vid}_blendshapes.csv"
        with open(dst, "w", encoding="utf-8", newline="") as f:
            f.write(",".join(BLENDSHAPE_NAMES) + "\n")
            np.savetxt(f, bs, fmt="%.4f", delimiter=",")
        print(f"[{vid}] {len(coords)} frames → {dst}")
        paths.append(dst)

    return paths


def main() -> None:
    """解析命令行并执行 tongue_regions → blendshape CSV 转换。"""
    parser = argparse.ArgumentParser(
        description="将 tongue_regions.csv 的舌头三点坐标映射为 blendshape CSV",
    )
    parser.add_argument(
        "-i",
        "--input",
        default=str(DEFAULT_TONGUE_REGIONS_CSV),
        help="输入 tongue_regions.csv 路径（默认 data/tongue_regions.csv）",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="输出目录（默认 outputs/）",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.5,
        help="时域高斯平滑标准差（帧），0 = 不平滑（默认 1.5）",
    )
    parser.add_argument(
        "--flip-y",
        action="store_true",
        help="翻转 y 轴方向（若坐标系 y 向上则需要此选项）",
    )
    parser.add_argument(
        "--no-tongue-mouth-constraint",
        action="store_true",
        dest="no_tongue_mouth_constraint",
        help="不按嘴张程度限制舌列（默认会限制，减轻舌穿出嘴唇）",
    )
    args = parser.parse_args()

    paths = convert_tongue_regions(
        args.input,
        args.output_dir,
        sigma=args.sigma,
        flip_y=args.flip_y,
        tongue_mouth_constraint=not args.no_tongue_mouth_constraint,
    )
    print(f"\n共转换 {len(paths)} 个片段，可用以下命令预览：")
    for p in paths:
        print(f"  uv run face-blender-shape preview --path {p}")


if __name__ == "__main__":
    main()
