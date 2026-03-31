"""在 Blender 内作为入口：播放 CSV blendshape 序列。"""
import argparse

from face_blender_shape.cli import preview_sequence
from face_blender_shape.constants import DEFAULT_PLAYBACK_FPS


def play_sequence(
    path: str,
    fps: float = DEFAULT_PLAYBACK_FPS,
    *,
    fbx_path: str | None = None,
    wireframe_head: bool = False,
) -> None:
    """按指定帧率播放 CSV blendshape 序列。

    参数:
        path: CSV 文件路径（37 列 blendshape）。
        fps: 播放帧率。
        fbx_path: 可选，覆盖默认 FBX 模型路径。
        wireframe_head: 可选，头壳线框 + 舌实体。
    """
    preview_sequence(
        path,
        fps,
        fbx_path=fbx_path,
        wireframe_head=wireframe_head,
    )


def main() -> None:
    """解析命令行并播放 CSV 序列。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="CSV sequence with 37 blendshape columns",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_PLAYBACK_FPS, help="Playback FPS for CSV sequences")
    parser.add_argument("--fbx", type=str, help="Override FBX path")
    parser.add_argument(
        "--wireframe-head",
        action="store_true",
        dest="wireframe_head",
        help="头壳线框、舌实体",
    )
    args = parser.parse_args()

    play_sequence(
        args.path,
        args.fps,
        fbx_path=args.fbx,
        wireframe_head=args.wireframe_head,
    )


if __name__ == "__main__":
    main()
