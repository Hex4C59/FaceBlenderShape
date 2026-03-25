#!/usr/bin/env python3
"""
联调步骤 3–5（mock）：模拟语音驱动 → 平面轨迹 → 映射 SRanipal → 双视窗预览。

不接真实音频时，用正弦包络代替「音频特征」；后续可换成 FINAL 的 wav2vec .p 再驱动轨迹。
默认使用 models/Metahuman_Head.fbx（勿传 --fbx / --head）；SRanipal 舌头通道在 MetaHuman 上映射弱，主要看下颌与嘴部。
若日后换成二进制导出的其它 FBX，可用 --fbx 与 --head。

用法（在项目根）:
  uv run python scripts/mock_talk_pipeline.py --dual --save-csv outputs/mock_from_traj.csv
  # 无图形界面（SSH、未设 DISPLAY）时加 --headless，或脚本在 Linux 上会自动无窗口运行
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# 从 scripts/ 直接运行时，Python 不会把仓库根目录加入 path；须手动加入才能 import face_blender_shape。
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from face_blender_shape.blender_runtime import FaceBlenderRuntime
from face_blender_shape.constants import DEFAULT_OPEN3D_HEIGHT, DEFAULT_OPEN3D_WIDTH, DEFAULT_VIEW_SCALE
from face_blender_shape.trajectory_mapping import (
    mock_trajectory_from_mock_audio,
    save_sranipal_csv,
    trajectory_to_sranipal_frames,
)


def _gui_likely_available() -> bool:
    """
    判断当前进程是否可能弹出 Open3D 窗口。
    Linux 无 DISPLAY/WAYLAND 时 GLFW 无法初始化；macOS/Windows 默认尝试创建窗口。
    """
    if sys.platform == "darwin" or sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mock：模拟音频→轨迹→blendshape→Open3D 主视+可选侧视")
    p.add_argument("--frames", type=int, default=240, help="总帧数")
    p.add_argument("--fps", type=float, default=30.0, help="播放帧率")
    p.add_argument(
        "--headless",
        action="store_true",
        help="不创建 Open3D 窗口，仅 Blender 逐帧求值（适合 SSH/无图形环境）",
    )
    p.add_argument("--dual", action="store_true", help="打开侧视窗口")
    p.add_argument(
        "--fbx",
        type=str,
        default=None,
        help="FBX 路径；省略则加载 models/Metahuman_Head.fbx（与 FaceBlenderRuntime 默认一致）",
    )
    p.add_argument("--head", type=str, default=None, help="头部网格对象名（自定义 FBX 时常必填）")
    p.add_argument(
        "--extra-meshes",
        type=str,
        default=None,
        help="逗号分隔的附加网格名（牙齿/眼等），可选",
    )
    p.add_argument("--jaw-base", type=float, default=0.38, help="基础 Jaw_Open，便于观察口腔内")
    p.add_argument("--save-csv", type=str, default=None, help="若指定路径则写出 37 维 CSV")
    p.add_argument("--view-scale", type=float, default=DEFAULT_VIEW_SCALE, help="Open3D 取景比例")
    return p.parse_args()


def _extra_tuple(s: str | None) -> tuple[str, ...] | None:
    if not s or not str(s).strip():
        return None
    return tuple(x.strip() for x in str(s).split(",") if x.strip())


def main() -> None:
    args = _parse_args()
    xy = mock_trajectory_from_mock_audio(int(args.frames), float(args.fps))
    frames = trajectory_to_sranipal_frames(
        xy,
        jaw_open_base=float(args.jaw_base),
        y_up_positive=False,
    )

    if args.save_csv:
        out = Path(args.save_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_sranipal_csv(str(out), frames)
        print(f"已写入 CSV: {out.resolve()}")

    use_viewer = not args.headless
    if use_viewer and not _gui_likely_available():
        print("未检测到 DISPLAY/WAYLAND_DISPLAY，自动以无窗口模式运行（等价 --headless）。")
        use_viewer = False

    fbx = Path(args.fbx).resolve() if args.fbx else None
    runtime = FaceBlenderRuntime(
        enable_viewer=use_viewer,
        enable_side_viewer=use_viewer and bool(args.dual),
        view_scale=float(args.view_scale),
        window_width=DEFAULT_OPEN3D_WIDTH,
        window_height=DEFAULT_OPEN3D_HEIGHT,
        head_object_name=args.head,
        extra_mesh_names=_extra_tuple(args.extra_meshes),
        fbx_path=fbx,
    )

    delay = 1.0 / float(args.fps) if float(args.fps) > 0 else 0.0
    n = frames.shape[0]
    for i in range(n):
        print(f"frame {i + 1}/{n}")
        runtime.update_visualizer(frames[i])
        if delay > 0:
            time.sleep(delay)


if __name__ == "__main__":
    main()
