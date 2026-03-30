"""将 tongue_regions.csv（舌头三点 2D 坐标轨迹）启发式映射为
37 列 blendshape CSV（首行为 BLENDSHAPE_NAMES 列名），可直接喂给本项目的 preview / convert 管线。

映射思路
--------
- root→tip 距离  →  Tongue_LongStep1 / LongStep2（舌头伸出程度）
- tip 相对 root 的仰角变化  →  Tongue_Up / Tongue_Down
- body 偏离 root→tip 连线的弯曲度  →  Tongue_Roll（近似）
- root 的垂直位移（下沉≈张嘴）  →  Jaw_Open 等口型通道

局限
----
- 超声矢状面只有二维侧视，无法推断左右运动 → Tongue_Left/Right 恒为 0
- 坐标→blendshape 的映射是启发式的，不等同于真实 SRAnipal 采集
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

from face_blender_shape.constants import BLENDSHAPE_INDEX, BLENDSHAPE_NAMES, FRAME_WIDTH

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_tongue_regions(csv_path: str | Path) -> dict[str, np.ndarray]:
    """读取 tongue_regions.csv，按 video 分组返回坐标数组。

    csv_path: 输入 CSV 文件路径。
    返回: {video_id: ndarray(N, 6)}，6 列为 root_x/y, body_x/y, tip_x/y。
    """
    groups: dict[str, list[list[float]]] = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["video"]
            coords = [
                float(row["root_x"]), float(row["root_y"]),
                float(row["body_x"]), float(row["body_y"]),
                float(row["tip_x"]), float(row["tip_y"]),
            ]
            groups.setdefault(vid, []).append(coords)
    return {vid: np.array(rows) for vid, rows in groups.items()}


# ---------------------------------------------------------------------------
# 几何特征
# ---------------------------------------------------------------------------

def compute_features(coords: np.ndarray) -> dict[str, np.ndarray]:
    """从三点坐标提取帧级几何特征（均为 shape (N,)）。

    coords: shape (N, 6)，列序 root_x, root_y, body_x, body_y, tip_x, tip_y。
    返回: 各几何特征字典。
    """
    root = coords[:, 0:2]
    body = coords[:, 2:4]
    tip = coords[:, 4:6]

    # root→tip 距离（舌头整体延伸长度）
    ext_vec = tip - root
    # 沿着（N，2），第1维求欧式距离
    ext_len = np.linalg.norm(ext_vec, axis=1)


    # root→tip 向量的仰角（弧度），y 轴向下时负角 = tip 在 root 上方
    angle = np.arctan2(-(ext_vec[:, 1]), ext_vec[:, 0])

    # body 偏离 root→tip 连线的有符号垂距（2D 叉积）
    line_dir = ext_vec / (ext_len[:, None] + 1e-8)
    rb = body - root
    curvature = line_dir[:, 0] * rb[:, 1] - line_dir[:, 1] * rb[:, 0]

    # root 垂直位移（向下越大≈张嘴程度越大）
    root_y = root[:, 1]

    return {
        "ext_len": ext_len,
        "angle": angle,
        "curvature": curvature,
        "root_y": root_y,
    }


# ---------------------------------------------------------------------------
# 归一化 & 平滑
# ---------------------------------------------------------------------------

def norm_deviation(arr: np.ndarray, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """以中位数为中性，将正/负偏差分别归一化到 [0, 1] 并乘以 scale。

    arr: 输入一维数组。
    scale: 最大输出值。
    返回: (positive_part, negative_part)，各 shape 同 arr。
    """
    med = np.median(arr)
    dev = arr - med
    pos = np.clip(dev, 0, None)
    neg = np.clip(-dev, 0, None)

    def _norm(a: np.ndarray) -> np.ndarray:
        hi = np.percentile(a, 97) if a.max() > 0 else 1.0
        return np.clip(a / (hi + 1e-8), 0.0, 1.0) * scale

    return _norm(pos), _norm(neg)


def norm_range(arr: np.ndarray, lo_pct: float = 3, hi_pct: float = 97) -> np.ndarray:
    """按百分位归一化到 [0, 1]。

    arr: 输入一维数组。
    lo_pct: 下界百分位。
    hi_pct: 上界百分位。
    """
    lo, hi = np.percentile(arr, lo_pct), np.percentile(arr, hi_pct)
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    """一维高斯平滑（零填充边界），sigma <= 0 时不做处理。

    arr: 输入一维数组。
    sigma: 高斯核标准差（帧数）。
    """
    if sigma <= 0 or len(arr) < 3:
        return arr
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(arr, sigma=sigma, mode="nearest")


# ---------------------------------------------------------------------------
# 核心映射
# ---------------------------------------------------------------------------

def features_to_blendshapes(
    feats: dict[str, np.ndarray],
    n: int,
    *,
    sigma: float = 1.5,
    flip_y: bool = False,
) -> np.ndarray:
    """将几何特征映射为 37 列 blendshape 权重。

    feats: compute_features 返回的特征字典。
    n: 帧数。
    sigma: 时域高斯平滑核宽度（帧），<= 0 不平滑。
    flip_y: 若为 True 则反转 y 轴方向（默认假设 y 向下 = 图像坐标）。
    返回: shape (n, 37) blendshape 数组。
    """
    frames = np.zeros((n, FRAME_WIDTH), dtype=float)

    ext = norm_range(feats["ext_len"])
    angle = feats["angle"] * (-1 if flip_y else 1)
    curvature = np.abs(feats["curvature"])
    root_y = feats["root_y"] * (-1 if flip_y else 1)

    # --- 舌头伸出 ---
    ls1 = smooth(ext * 0.92, sigma)
    ls2 = smooth(np.clip((ext - 0.25) / 0.75, 0, 1) * 0.65, sigma)
    frames[:, BLENDSHAPE_INDEX["Tongue_LongStep1"]] = ls1
    frames[:, BLENDSHAPE_INDEX["Tongue_LongStep2"]] = ls2

    # --- 舌尖上翘 / 下压（用 root→tip 仰角偏差） ---
    up, down = norm_deviation(angle, scale=0.80)
    frames[:, BLENDSHAPE_INDEX["Tongue_Up"]] = smooth(up, sigma)
    frames[:, BLENDSHAPE_INDEX["Tongue_Down"]] = smooth(down, sigma)

    # --- 卷舌（body 弯曲度） ---
    roll = norm_range(curvature) * 0.45
    frames[:, BLENDSHAPE_INDEX["Tongue_Roll"]] = smooth(roll, sigma)

    # --- 对角 morph：用 angle 偏差 × 曲率 给一点微动，增加丰富度 ---
    curv_n = norm_range(curvature)
    angle_pos, angle_neg = norm_deviation(angle, scale=1.0)
    frames[:, BLENDSHAPE_INDEX["Tongue_UpLeft_Morph"]] = smooth(angle_pos * curv_n * 0.25, sigma)
    frames[:, BLENDSHAPE_INDEX["Tongue_DownRight_Morph"]] = smooth(angle_neg * curv_n * 0.25, sigma)

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

    # --- 防穿模：舌头运动幅度不超过 Jaw_Open 提供的空间 ---
    jaw_val = frames[:, BLENDSHAPE_INDEX["Jaw_Open"]]
    td_idx = BLENDSHAPE_INDEX["Tongue_Down"]
    ls1_idx = BLENDSHAPE_INDEX["Tongue_LongStep1"]
    ls2_idx = BLENDSHAPE_INDEX["Tongue_LongStep2"]
    frames[:, td_idx] = np.minimum(frames[:, td_idx], jaw_val * 0.9)
    frames[:, ls1_idx] = np.minimum(frames[:, ls1_idx], jaw_val * 1.2)
    frames[:, ls2_idx] = np.minimum(frames[:, ls2_idx], jaw_val * 0.85)

    return np.clip(frames, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def convert_tongue_regions(
    input_csv: str | Path,
    output_dir: str | Path | None = None,
    *,
    sigma: float = 1.5,
    flip_y: bool = False,
) -> list[Path]:
    """读取 tongue_regions.csv，为每个 video 片段生成一份 blendshape CSV。

    input_csv: 输入 CSV 路径。
    output_dir: 输出目录，默认 outputs/。
    sigma: 时域平滑参数（帧），<= 0 不平滑。
    flip_y: 是否翻转 y 轴。
    返回: 生成的文件路径列表。
    """
    out = Path(output_dir) if output_dir else OUTPUTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    groups = load_tongue_regions(input_csv)
    paths: list[Path] = []

    for vid, coords in groups.items():
        feats = compute_features(coords)
        bs = features_to_blendshapes(feats, len(coords), sigma=sigma, flip_y=flip_y)
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
        "-i", "--input",
        default=str(PROJECT_ROOT / "tongue_regions.csv"),
        help="输入 tongue_regions.csv 路径（默认仓库根目录下的 tongue_regions.csv）",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="输出目录（默认 outputs/）",
    )
    parser.add_argument(
        "--sigma", type=float, default=1.5,
        help="时域高斯平滑标准差（帧），0 = 不平滑（默认 1.5）",
    )
    parser.add_argument(
        "--flip-y", action="store_true",
        help="翻转 y 轴方向（若坐标系 y 向上则需要此选项）",
    )
    args = parser.parse_args()

    paths = convert_tongue_regions(
        args.input,
        args.output_dir,
        sigma=args.sigma,
        flip_y=args.flip_y,
    )
    print(f"\n共转换 {len(paths)} 个片段，可用以下命令预览：")
    for p in paths:
        print(f"  uv run face-blender-shape preview --path {p}")


if __name__ == "__main__":
    main()
