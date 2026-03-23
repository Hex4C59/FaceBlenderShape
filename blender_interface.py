import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from face_blender_shape.blender_runtime import FaceBlenderRuntime
from face_blender_shape.cli import preview_all_shapes, preview_sequence
from face_blender_shape.constants import DEFAULT_PLAYBACK_FPS

EMBlender = FaceBlenderRuntime


def play_sequence(path: str, fps: float = DEFAULT_PLAYBACK_FPS, *, fbx_path: str | None = None, texture_path: str | None = None, model: str = "sranipal"):
    preview_sequence(path, fps, fbx_path=fbx_path, texture_path=texture_path, model=model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="CSV sequence with 37 blendshape columns")
    parser.add_argument("--fps", type=float, default=DEFAULT_PLAYBACK_FPS, help="Playback FPS for CSV sequences")
    parser.add_argument("--fbx", type=str, help="Override FBX path")
    parser.add_argument("--texture", type=str, help="Skin texture image path (auto-detects from assets/textures/)")
    parser.add_argument("--model", type=str, default="sranipal", choices=["sranipal", "metahuman"], help="Model backend (default: sranipal)")
    args = parser.parse_args()

    if args.path:
        play_sequence(args.path, args.fps, fbx_path=args.fbx, texture_path=args.texture, model=args.model)
    else:
        preview_all_shapes(fbx_path=args.fbx, texture_path=args.texture, model=args.model)


if __name__ == "__main__":
    main()
