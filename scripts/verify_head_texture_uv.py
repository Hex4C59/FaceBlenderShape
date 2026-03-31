"""导入 sranipal_head.fbx，按与 blender_runtime 相同的规则从给定 PNG 采样 UV，用于核对贴图是否匹配。"""

from __future__ import annotations

import argparse
from pathlib import Path

import bpy
import numpy as np


def _bpy_image_to_rgb_u8(img: object) -> np.ndarray:
    """
    将 bpy 图像转为 H×W×3 的 uint8 RGB（与 blender_runtime 中 flip 规则一致）。

    参数:
        img: bpy.types.Image。
    """
    w, h = img.size
    ch = img.channels
    pixels = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, ch)
    pixels = np.flipud(pixels)
    if ch >= 4:
        pixels = pixels[:, :, :3]
    return np.clip(pixels * 255.0, 0, 255).astype(np.uint8)


def _head_fan_triangle_uvs(mesh: object) -> np.ndarray:
    """
    对 Head 网格做扇形三角化，收集每个三角三个角点的 (u,v)，形状 (3*F, 2)。

    参数:
        mesh: bpy.types.Mesh，须含活动 UV 层。
    """
    uv_layer = mesh.uv_layers.active
    if uv_layer is None:
        raise RuntimeError("网格无活动 UV 层")
    out: list[tuple[float, float]] = []
    for poly in mesh.polygons:
        li = list(poly.loop_indices)
        n = len(li)
        if n < 3:
            continue
        if n == 3:
            for idx in li:
                uv = uv_layer.data[idx].uv
                out.append((float(uv[0]), float(uv[1])))
        else:
            for i in range(1, n - 1):
                for idx in (li[0], li[i], li[i + 1]):
                    uv = uv_layer.data[idx].uv
                    out.append((float(uv[0]), float(uv[1])))
    return np.array(out, dtype=np.float64)


def _sample_centroids(
    triangle_uvs: np.ndarray, texture: np.ndarray
) -> np.ndarray:
    """
    对每个三角面的 UV 重心在贴图上采样 RGB，返回 (F, 3) uint8。

    参数:
        triangle_uvs: (3*F, 2) 每行一个角点 UV。
        texture: H×W×3 uint8。
    """
    h, w, _ = texture.shape
    tri = triangle_uvs.reshape(-1, 3, 2)
    cent = tri.mean(axis=1)
    us = np.clip(cent[:, 0], 0.0, 1.0)
    vs = np.clip(cent[:, 1], 0.0, 1.0)
    px = (us * (w - 1)).astype(np.int64)
    py = ((1.0 - vs) * (h - 1)).astype(np.int64)
    return texture[py, px]


def _report(name: str, texture: np.ndarray, triangle_uvs: np.ndarray) -> None:
    """
    打印贴图尺寸与在 Head 三角重心上的采样统计。

    参数:
        name: 贴图标识字符串。
        texture: H×W×3 uint8。
        triangle_uvs: (3*F, 2) UV 展开数组。
    """
    h, w, _ = texture.shape
    colors = _sample_centroids(triangle_uvs, texture)
    print(f"--- {name} ---")
    print(f"  图像尺寸: {w}×{h}")
    print(f"  三角面数: {len(colors)}")
    print(f"  重心采样 RGB 均值: {colors.mean(axis=0)}")
    print(f"  重心采样 RGB 标准差: {colors.std(axis=0)}")
    print(f"  各通道 min: {colors.min(axis=0)}, max: {colors.max(axis=0)}")


def main() -> None:
    """
    解析路径，导入 FBX，对 Head 网格验证 UV 与一张或多张 PNG 的采样统计。

    参数: 无（使用 argparse）。
    """
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fbx",
        type=Path,
        default=root / "assets/models/sranipal_head.fbx",
        help="头模 FBX 路径",
    )
    parser.add_argument(
        "--texture",
        type=Path,
        action="append",
        default=[],
        help="可多次指定，每张 PNG 做一遍采样统计",
    )
    args = parser.parse_args()
    fbx: Path = args.fbx.expanduser()
    tex_paths: list[Path] = list(args.texture) if args.texture else []

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(fbx))

    head = bpy.data.objects.get("Head")
    if head is None or head.type != "MESH":
        raise SystemExit("场景中未找到名为 Head 的网格对象")

    mesh = head.data
    if not mesh.uv_layers:
        raise SystemExit("Head 无 UV 层")

    tu = _head_fan_triangle_uvs(mesh)
    print("FBX:", fbx.resolve())
    print("Head: 扇形三角化后 UV 角点行数 =", tu.shape[0], "(应为 3×三角面数)")
    print("UV 分量范围: u ∈", float(tu[:, 0].min()), "…", float(tu[:, 0].max()))
    print("              v ∈", float(tu[:, 1].min()), "…", float(tu[:, 1].max()))
    outside = np.sum((tu < 0) | (tu > 1))
    if outside:
        print("注意: 有", outside, "个分量落在 [0,1] 外（已按运行时规则 clip）")

    if not tex_paths:
        tex_paths = [
            root / "assets/textures/fairy_body_Albedo.png",
            root / "assets/textures/Head_b_albedo.png",
        ]

    for p in tex_paths:
        p = p.expanduser()
        if not p.is_file():
            print(f"\n跳过（文件不存在）: {p}")
            continue
        img = bpy.data.images.load(str(p))
        if img.size[0] == 0:
            print(f"\n无法加载图像: {p}")
            continue
        rgb = _bpy_image_to_rgb_u8(img)
        _report(str(p), rgb, tu)


if __name__ == "__main__":
    main()
