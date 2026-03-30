"""在 Blender 内作为入口：播放 CSV 序列或逐个预览 blendshape（由 CLI 参数决定）。"""
import argparse

from face_blender_shape.cli import preview_all_shapes, preview_sequence
from face_blender_shape.constants import DEFAULT_PLAYBACK_FPS

def play_sequence(
    path: str,
    fps: float = DEFAULT_PLAYBACK_FPS,
    *,
    fbx_path: str | None = None,
    texture_path: str | None = None,
    model: str = "sranipal",
    ):
    """按指定帧率播放 CSV blendshape 序列。

    path: CSV 文件路径（37 列 blendshape）。
    fps: 播放帧率。
    fbx_path: 可选，覆盖默认 FBX 模型路径。
    texture_path: 可选，皮肤贴图路径。
    model: 后端模型标识，如 sranipal / metahuman。
    """
    preview_sequence(
        path,
        fps,
        fbx_path=fbx_path,
        texture_path=texture_path, 
        model=model
        )


def main() -> None:
    """解析命令行：若提供 --path 则播放序列，否则进入全 shape 预览模式。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="CSV sequence with 37 blendshape columns")
    parser.add_argument("--fps", type=float, default=DEFAULT_PLAYBACK_FPS, help="Playback FPS for CSV sequences")
    parser.add_argument("--fbx", type=str, help="Override FBX path")
    parser.add_argument("--texture", type=str, help="Skin texture image path (optional; uses bpy material texture if omitted)")
    parser.add_argument("--model", type=str, default="sranipal", choices=["sranipal", "metahuman"], help="Model backend (default: sranipal)")
    args = parser.parse_args()

    if args.path:
        play_sequence(args.path, args.fps, fbx_path=args.fbx, texture_path=args.texture, model=args.model)
    else:
        preview_all_shapes(fbx_path=args.fbx, texture_path=args.texture, model=args.model)


if __name__ == "__main__":
    main()
