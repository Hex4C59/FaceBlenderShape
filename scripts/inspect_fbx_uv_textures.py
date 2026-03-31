"""用 bpy 导入 FBX 并打印 UV 层与材质 TEX_IMAGE（供本地 uv run 检查）。"""

from __future__ import annotations

import argparse
from pathlib import Path

import bpy


def main() -> None:
    """
    解析命令行，导入 FBX 后打印各网格 UV 与材质图像节点信息。

    参数: 无（来自 argparse）。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fbx",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent.parent / "assets/models/sranipal_head.fbx",
        help="FBX 路径（须为二进制 FBX；PyPI bpy 不支持 ASCII FBX）",
    )
    args = parser.parse_args()
    fbx: Path = args.fbx.expanduser()
    if not fbx.is_file():
        raise SystemExit(f"找不到文件: {fbx}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(fbx))

    print("FBX:", fbx)
    print("--- 网格与 UV ---")
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        mesh = obj.data
        uvs = mesh.uv_layers
        print(f"  {obj.name!r}: verts={len(mesh.vertices)}, faces={len(mesh.polygons)}")
        if not uvs:
            print("    UV: 无")
        else:
            for ul in uvs:
                print(
                    f"    UV 层 {ul.name!r}: loops={len(ul.data)} "
                    f"(mesh.loops={len(mesh.loops)})"
                )

    print("--- 材质 TEX_IMAGE ---")
    seen: set[object] = set()
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None or mat in seen:
                continue
            seen.add(mat)
            print(f"  材质 {mat.name!r}, use_nodes={mat.use_nodes}")
            if not mat.use_nodes:
                continue
            for node in mat.node_tree.nodes:
                if node.type != "TEX_IMAGE":
                    continue
                img = node.image
                if img is None:
                    print(f"    TEX_IMAGE {node.name!r}: image=None")
                else:
                    w, h = img.size
                    has_px = w > 0 and h > 0 and len(img.pixels) > 0
                    print(
                        f"    TEX_IMAGE {node.name!r}: {img.name!r} "
                        f"size=({w},{h}) filepath={img.filepath!r} "
                        f"packed={img.packed_file is not None} has_pixels={has_px}"
                    )

    print("--- bpy.data.images ---")
    for img in bpy.data.images:
        if img.name in ("Render Result", "Viewer Node"):
            continue
        w, h = img.size
        print(f"  {img.name!r}: ({w},{h}) {img.filepath!r}")


if __name__ == "__main__":
    main()
