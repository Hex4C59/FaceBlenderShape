"""
将 SRAnipal 等 blendshape CSV 转为关键点 NPZ 的独立入口脚本。

依赖 face_blender_shape.cli.convert_csv_to_keypoints；通过调整 sys.path 支持在未 pip 安装时从仓库内直接运行。
"""
import argparse
import sys
from pathlib import Path

# 将仓库根目录加入 sys.path，便于以「python scripts/sranipal2keypoints.py」方式运行时导入 face_blender_shape。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from face_blender_shape.cli import convert_csv_to_keypoints


def main() -> None:
    """
    解析命令行并调用 CSV→关键点 转换逻辑。

    命令行参数（由 argparse 注入，无显式形参）：
    - path：输入 blendshape CSV 文件路径（必填）。
    - output：输出 NPZ 路径；省略则使用 CLI 默认规则。
    - fbx：覆盖默认 FBX 模型路径；省略则使用项目内默认资源。
    - visualize：若指定则在转换过程中打开 Blender 可视化。
    """
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
