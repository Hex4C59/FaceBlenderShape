from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from face_blender_shape.blender_runtime import FaceBlenderRuntime
from face_blender_shape.constants import (
    DEFAULT_OPEN3D_HEIGHT,
    DEFAULT_OPEN3D_WIDTH,
    DEFAULT_PLAYBACK_FPS,
    DEFAULT_VIEW_SCALE,
    FRAME_WIDTH,
)
from face_blender_shape.io import load_blendshape_csv, save_keypoints_npz


def parse_extra_mesh_names(s: str | None) -> tuple[str, ...] | None:
    if s is None or not str(s).strip():
        return None
    return tuple(x.strip() for x in str(s).split(",") if x.strip())


def preview_sequence(
    path: str | Path,
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
) -> None:
    data = load_blendshape_csv(path)
    runtime = FaceBlenderRuntime(
        path=fbx_path,
        enable_viewer=True,
        texture_path=texture_path,
        model=model,
        view_scale=view_scale,
        window_width=window_width,
        window_height=window_height,
        head_object_name=head_object_name,
        extra_mesh_names=extra_mesh_names,
    )
    frame_delay = 1.0 / fps if fps > 0 else 0.0

    for idx, blendshapes in enumerate(data, start=1):
        print(f"frame {idx}/{len(data)}")
        runtime.update_visualizer(blendshapes)
        if frame_delay > 0:
            time.sleep(frame_delay)


def preview_all_shapes(
    *,
    fbx_path: str | None = None,
    texture_path: str | None = None,
    model: str = "sranipal",
    view_scale: float = DEFAULT_VIEW_SCALE,
    window_width: int = DEFAULT_OPEN3D_WIDTH,
    window_height: int = DEFAULT_OPEN3D_HEIGHT,
    head_object_name: str | None = None,
    extra_mesh_names: tuple[str, ...] | None = None,
) -> None:
    runtime = FaceBlenderRuntime(
        path=fbx_path,
        enable_viewer=True,
        texture_path=texture_path,
        model=model,
        view_scale=view_scale,
        window_width=window_width,
        window_height=window_height,
        head_object_name=head_object_name,
        extra_mesh_names=extra_mesh_names,
    )
    for value in np.linspace(0.0, 1.0, 100):
        print(value)
        runtime.update_visualizer(np.ones(FRAME_WIDTH) * value)


def convert_csv_to_keypoints(
    path: str | Path,
    *,
    output_path: str | Path | None = None,
    fbx_path: str | None = None,
    visualize: bool = False,
    model: str = "sranipal",
    head_object_name: str | None = None,
    extra_mesh_names: tuple[str, ...] | None = None,
) -> Path:
    data = load_blendshape_csv(path)
    runtime = FaceBlenderRuntime(
        path=fbx_path,
        enable_viewer=visualize,
        model=model,
        head_object_name=head_object_name,
        extra_mesh_names=extra_mesh_names,
    )

    vertices_frames = []
    lip_frames = []
    tongue_tip_frames = []
    cheek_keypoint_frames = []
    keypoint_frames = []
    faces = None

    for idx, blendshapes in enumerate(data, start=1):
        print(f"extracting frame {idx}/{len(data)}")
        frame = runtime.extract_frame(blendshapes)
        vertices_frames.append(frame["vertices"])
        lip_frames.append(frame["lip"])
        tongue_tip_frames.append(frame["tongue_tip"])
        cheek_keypoint_frames.append(frame["cheek_keypoints"])
        keypoint_frames.append(frame["keypoints"])
        if faces is None:
            faces = frame["faces"]
        if visualize:
            runtime.render(
                frame["vertices"],
                frame["faces"],
                vertex_colors=frame.get("vertex_colors"),
            )

    if faces is None:
        raise RuntimeError("No frames were extracted from the input CSV")

    output = save_keypoints_npz(
        path,
        blendshapes=data,
        vertices=np.stack(vertices_frames, axis=0),
        faces=faces,
        lip=np.stack(lip_frames, axis=0),
        tongue_tip=np.stack(tongue_tip_frames, axis=0),
        cheek_keypoints=np.stack(cheek_keypoint_frames, axis=0),
        keypoints=np.stack(keypoint_frames, axis=0),
        output_path=output_path,
    )
    print(f"saved keypoints to {output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="face_blender_shape")
    subparsers = parser.add_subparsers(dest="command")

    preview_parser = subparsers.add_parser("preview", help="Preview a blendshape CSV or sweep all shapes")
    preview_parser.add_argument("--path", type=str, help="CSV sequence with 37 blendshape columns")
    preview_parser.add_argument("--fps", type=float, default=DEFAULT_PLAYBACK_FPS, help="Playback FPS")
    preview_parser.add_argument("--fbx", type=str, help="Override FBX path")
    preview_parser.add_argument("--texture", type=str, help="Skin texture image path (auto-detects from assets/textures/)")
    preview_parser.add_argument("--model", type=str, default="sranipal", choices=["sranipal", "metahuman"], help="Model backend (default: sranipal)")
    preview_parser.add_argument("--view-scale", type=float, default=DEFAULT_VIEW_SCALE, help="Closer framing for meter-scale models (MetaHuman); larger = bigger face")
    preview_parser.add_argument("--window-width", type=int, default=DEFAULT_OPEN3D_WIDTH, help="Viewer window width in pixels")
    preview_parser.add_argument("--window-height", type=int, default=DEFAULT_OPEN3D_HEIGHT, help="Viewer window height in pixels")
    preview_parser.add_argument(
        "--head",
        type=str,
        default=None,
        help="Face mesh object name inside FBX (metahuman / custom ARKit; default: bundled MetaHuman or Head)",
    )
    preview_parser.add_argument(
        "--extra-meshes",
        type=str,
        default=None,
        help='Comma-separated mesh names to merge for preview (e.g. teeth,eyes,hair). See docs/custom-digital-human.md',
    )
    preview_parser.set_defaults(handler=handle_preview_command)

    convert_parser = subparsers.add_parser("convert", help="Convert a blendshape CSV into NPZ keypoints")
    convert_parser.add_argument("--path", required=True, type=str, help="Input CSV path")
    convert_parser.add_argument("--output", type=str, help="Output NPZ path")
    convert_parser.add_argument("--fbx", type=str, help="Override FBX path")
    convert_parser.add_argument("--visualize", action="store_true", help="Render while converting")
    convert_parser.add_argument("--model", type=str, default="sranipal", choices=["sranipal", "metahuman"])
    convert_parser.add_argument("--head", type=str, default=None, help="Face mesh object name (see preview --head)")
    convert_parser.add_argument("--extra-meshes", type=str, default=None, help="Comma-separated extra meshes (see preview)")
    convert_parser.set_defaults(handler=handle_convert_command)

    return parser


def handle_preview_command(args: argparse.Namespace) -> int:
    extra = parse_extra_mesh_names(getattr(args, "extra_meshes", None))
    if args.path:
        preview_sequence(
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
    return 0


def handle_convert_command(args: argparse.Namespace) -> int:
    convert_csv_to_keypoints(
        args.path,
        output_path=args.output,
        fbx_path=args.fbx,
        visualize=args.visualize,
        model=args.model,
        head_object_name=args.head,
        extra_mesh_names=parse_extra_mesh_names(getattr(args, "extra_meshes", None)),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "handler"):
        parser.print_help()
        return 0

    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
