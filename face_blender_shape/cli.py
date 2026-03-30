from __future__ import annotations

import argparse
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from face_blender_shape.blender_runtime import FaceBlenderRuntime
from face_blender_shape.constants import DEFAULT_PLAYBACK_FPS, FRAME_WIDTH
from face_blender_shape.io import load_blendshape_csv, save_keypoints_npz


def preview_sequence(
    path: str | Path,
    fps: float = DEFAULT_PLAYBACK_FPS,
    *,
    fbx_path: str | None = None,
    texture_path: str | None = None,
    model: str = "sranipal",
    cutaway: bool = False,
) -> None:
    data: NDArray[np.float64] = load_blendshape_csv(path)
    runtime = FaceBlenderRuntime(
        path=fbx_path,
        enable_viewer=True,
        texture_path=texture_path,
        model=model,
        cutaway=cutaway,
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
) -> None:
    runtime = FaceBlenderRuntime(
        path=fbx_path, enable_viewer=True, texture_path=texture_path, model=model
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
) -> Path:
    data: NDArray[np.float64] = load_blendshape_csv(path)
    runtime = FaceBlenderRuntime(path=fbx_path, enable_viewer=visualize)

    vertices_frames: list[NDArray[np.float64]] = []
    lip_frames: list[NDArray[np.float64]] = []
    tongue_tip_frames: list[NDArray[np.float64]] = []
    cheek_keypoint_frames: list[NDArray[np.float64]] = []
    keypoint_frames: list[NDArray[np.float64]] = []
    faces: NDArray[np.int64] | None = None

    for idx in range(data.shape[0]):
        blendshapes = cast(NDArray[np.float64], data[idx])
        print(f"extracting frame {idx + 1}/{data.shape[0]}")
        frame = runtime.extract_frame(blendshapes)
        vertices_frames.append(frame["vertices"])
        lip_frames.append(frame["lip"])
        tongue_tip_frames.append(frame["tongue_tip"])
        cheek_keypoint_frames.append(frame["cheek_keypoints"])
        keypoint_frames.append(frame["keypoints"])
        if faces is None:
            faces = frame["faces"]
        if visualize:
            runtime.render(frame["vertices"], frame["faces"])

    if faces is None:
        raise RuntimeError("No frames were extracted from the input CSV")

    output = save_keypoints_npz(
        input_path=path,
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

    preview_parser = subparsers.add_parser(
        "preview", help="Preview a blendshape CSV or sweep all shapes"
    )
    preview_parser.add_argument(
        "--path", type=str, help="CSV sequence with 37 blendshape columns"
    )
    preview_parser.add_argument(
        "--fps", type=float, default=DEFAULT_PLAYBACK_FPS, help="Playback FPS"
    )
    preview_parser.add_argument("--fbx", type=str, help="Override FBX path")
    preview_parser.add_argument(
        "--texture",
        type=str,
        help="Skin texture image path (auto-detects from assets/textures/)",
    )
    preview_parser.add_argument(
        "--model",
        type=str,
        default="sranipal",
        choices=["sranipal", "metahuman"],
        help="Model backend (default: sranipal)",
    )
    preview_parser.add_argument(
        "--cutaway", action="store_true", help="移除唇部面片，露出口腔内舌头"
    )
    preview_parser.set_defaults(handler=handle_preview_command)

    return parser


def handle_preview_command(args: argparse.Namespace) -> int:
    if args.path:
        preview_sequence(
            args.path,
            args.fps,
            fbx_path=args.fbx,
            texture_path=args.texture,
            model=args.model,
            cutaway=args.cutaway,
        )
    else:
        preview_all_shapes(
            fbx_path=args.fbx, texture_path=args.texture, model=args.model
        )
    return 0


def handle_convert_command(args: argparse.Namespace) -> int:
    convert_csv_to_keypoints(
        args.path, output_path=args.output, fbx_path=args.fbx, visualize=args.visualize
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser: ArgumentParser = build_parser()
    args: Namespace = parser.parse_args(argv)

    if not hasattr(args, "handler"):
        parser.print_help()
        return 0

    return args.handler(args)
