#!/usr/bin/env python3
"""List mesh object names inside an FBX (for use with --head / --extra-meshes)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root on path when run as `uv run python scripts/list_fbx_meshes.py`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import bpy

from face_blender_shape.paths import resolve_fbx_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fbx", type=str, help="Path to .fbx file")
    parser.add_argument(
        "--shape-keys",
        action="store_true",
        help="Also print shape key counts on each mesh",
    )
    args = parser.parse_args()

    path = resolve_fbx_path(args.fbx)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.import_scene.fbx(filepath=str(path))

    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    meshes.sort(key=lambda o: o.name.lower())
    for o in meshes:
        sk = o.data.shape_keys
        n_shapes = len(sk.key_blocks) - 1 if sk else 0
        if args.shape_keys:
            print(f"{o.name}\tverts={len(o.data.vertices)}\tblendshapes={n_shapes}")
        else:
            print(o.name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
