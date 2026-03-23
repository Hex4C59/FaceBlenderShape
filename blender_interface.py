import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from face_blender_shape.blender_runtime import FaceBlenderRuntime
from face_blender_shape.cli import parse_extra_mesh_names, preview_all_shapes, preview_sequence
from face_blender_shape.constants import (
    DEFAULT_OPEN3D_HEIGHT,
    DEFAULT_OPEN3D_WIDTH,
    DEFAULT_PLAYBACK_FPS,
    DEFAULT_VIEW_SCALE,
)

EMBlender = FaceBlenderRuntime


def play_sequence(
    path: str,
    fps: float = DEFAULT_PLAYBACK_FPS,
    *,
    fbx_path: str | None = None,
    texture_path: str | None = None,
    model: str = "sranipal",
    view_scale: float = DEFAULT_VIEW_SCALE,
    window_width: int = DEFAULT_OPEN3D_WIDTH,
    window_height: int = DEFAULT_OPEN3D_HEIGHT,
    head_object_name: str | None = None,
    extra_mesh_names: tuple[str, ...] | None = None,
):
    preview_sequence(
        path,
        fps,
        fbx_path=fbx_path,
        texture_path=texture_path,
        model=model,
        view_scale=view_scale,
        window_width=window_width,
        window_height=window_height,
        head_object_name=head_object_name,
        extra_mesh_names=extra_mesh_names,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="CSV sequence with 37 blendshape columns")
    parser.add_argument("--fps", type=float, default=DEFAULT_PLAYBACK_FPS, help="Playback FPS for CSV sequences")
    parser.add_argument("--fbx", type=str, help="Override FBX path")
    parser.add_argument("--texture", type=str, help="Skin texture image path (auto-detects from assets/textures/)")
    parser.add_argument("--model", type=str, default="sranipal", choices=["sranipal", "metahuman"], help="Model backend (default: sranipal)")
    parser.add_argument("--view-scale", type=float, default=DEFAULT_VIEW_SCALE, help="Larger = bigger face (meter-scale / MetaHuman)")
    parser.add_argument("--window-width", type=int, default=DEFAULT_OPEN3D_WIDTH, help="Viewer window width (pixels)")
    parser.add_argument("--window-height", type=int, default=DEFAULT_OPEN3D_HEIGHT, help="Viewer window height (pixels)")
    parser.add_argument("--head", type=str, default=None, help="Face mesh object name in FBX (custom ARKit avatar)")
    parser.add_argument("--extra-meshes", type=str, default=None, help="Comma-separated meshes to merge (teeth, hair, …)")
    args = parser.parse_args()
    extra = parse_extra_mesh_names(args.extra_meshes)

    if args.path:
        play_sequence(
            args.path,
            args.fps,
            fbx_path=args.fbx,
            texture_path=args.texture,
            model=args.model,
            view_scale=args.view_scale,
            window_width=args.window_width,
            window_height=args.window_height,
            head_object_name=args.head,
            extra_mesh_names=extra,
        )
    else:
        preview_all_shapes(
            fbx_path=args.fbx,
            texture_path=args.texture,
            model=args.model,
            view_scale=args.view_scale,
            window_width=args.window_width,
            window_height=args.window_height,
            head_object_name=args.head,
            extra_mesh_names=extra,
        )


if __name__ == "__main__":
    main()
