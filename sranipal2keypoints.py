import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from face_blender_shape.cli import convert_csv_to_keypoints


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", "--path", dest="path", required=True, type=str, help="Input CSV path")
    parser.add_argument("--output", type=str, help="Output NPZ path")
    parser.add_argument("--fbx", type=str, help="Override FBX path")
    parser.add_argument("--visualize", action="store_true", help="Render while converting")
    args = parser.parse_args()

    convert_csv_to_keypoints(
        args.path,
        output_path=args.output,
        fbx_path=args.fbx,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    main()
