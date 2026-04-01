"""在 Blender 内作为入口：播放 CSV blendshape 序列。"""
import argparse

from face_blender_shape.cli import preview_sequence
from face_blender_shape.constants import DEFAULT_PLAYBACK_FPS
from face_blender_shape.landmarks import TONGUE_SLICE


def play_sequence(
    path: str,
    fps: float = DEFAULT_PLAYBACK_FPS,
    *,
    fbx_path: str | None = None,
    wireframe_head: bool = False,
    tongue_vertex_lo: int | None = None,
    tongue_vertex_hi: int | None = None,
    tongue_adjacency_expand: int = 0,
) -> None:
    """按指定帧率播放 CSV blendshape 序列。

    参数:
        path: CSV 文件路径（52 列 blendshape，与 Head 可驱动形态键一致）。
        fps: 播放帧率。
        fbx_path: 可选，覆盖默认 FBX 模型路径。
        wireframe_head: 可选，头壳线框 + 舌实体。
        tongue_vertex_lo / tongue_vertex_hi: 线框模式下舌顶点下标范围。
        tongue_adjacency_expand: 舌面沿邻接边扩展轮数。
    """
    preview_sequence(
        path,
        fps,
        fbx_path=fbx_path,
        wireframe_head=wireframe_head,
        tongue_vertex_lo=tongue_vertex_lo,
        tongue_vertex_hi=tongue_vertex_hi,
        tongue_adjacency_expand=tongue_adjacency_expand,
    )


def main() -> None:
    """解析命令行并播放 CSV 序列。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="CSV sequence with 52 blendshape columns",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_PLAYBACK_FPS, help="Playback FPS for CSV sequences")
    parser.add_argument("--fbx", type=str, help="Override FBX path")
    parser.add_argument(
        "--wireframe-head",
        action="store_true",
        dest="wireframe_head",
        help="头壳线框、舌实体",
    )
    parser.add_argument(
        "--tongue-lo",
        type=int,
        default=None,
        metavar="N",
        help=f"舌顶点下界（含）；默认 {TONGUE_SLICE.start}",
    )
    parser.add_argument(
        "--tongue-hi",
        type=int,
        default=None,
        metavar="N",
        help=f"舌顶点上界（不含）；默认 {TONGUE_SLICE.stop}",
    )
    parser.add_argument(
        "--tongue-adjacency-expand",
        type=int,
        default=0,
        metavar="ITERS",
        help="舌三角面邻接扩展轮数（默认 0）",
    )
    args = parser.parse_args()

    play_sequence(
        args.path,
        args.fps,
        fbx_path=args.fbx,
        wireframe_head=args.wireframe_head,
        tongue_vertex_lo=args.tongue_lo,
        tongue_vertex_hi=args.tongue_hi,
        tongue_adjacency_expand=args.tongue_adjacency_expand,
    )


if __name__ == "__main__":
    main()
