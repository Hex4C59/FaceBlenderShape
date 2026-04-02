# Face Blender Shape Python

## 简介

如果你使用 Vive Facial Tracker 作为真值系统，并希望获取毫米尺度的人脸关键点数据，或者你想在不依赖 Unity 的情况下可视化 SRanipal Blender Shape 的面部效果，那么这个仓库可能会对你有帮助。

## 安装

本仓库使用 `pyproject.toml + uv.lock` 作为**唯一依赖事实来源**，并且需要 Python `3.10`，因为 `bpy==4.0.0` 仅为该 Python 版本提供了预编译 wheel。

推荐安装方式：

```bash
uv sync
```

如果你希望强制 `uv` 显式使用 Python 3.10 创建环境：

```bash
uv sync --python 3.10
```

仓库中还包含一个固定为 `3.10` 的 `.python-version` 文件，因此 `uv sync` 创建的虚拟环境以及你在该环境中执行的 `python` 会对应 Python 3.10。

如果你想手动激活环境：

```bash
source .venv/bin/activate
```

### pip 兼容安装

如果你不用 `uv`，也可以使用兼容用的 `requirements.txt`：

```bash
pip install -r requirements.txt
```

不过需要注意：
- `requirements.txt` 只是一个兼容入口；
- 真正维护的依赖定义仍在 `pyproject.toml`；
- 更新依赖时应修改 `pyproject.toml`，然后重新生成/同步锁文件，而不是手改 `requirements.txt`。

## 目录概览

```text
FaceBlenderShape/
├── blender_interface.py
├── sranipal2keypoints.py
├── face_blender_shape/
├── scripts/
├── assets/models/
├── data/examples/
├── outputs/
└── docs/assets/
```

- `blender_interface.py`：保留的顶层兼容入口，用于 Blender 预览
- `sranipal2keypoints.py`：保留的顶层兼容入口，用于关键点导出
- `face_blender_shape/`：核心运行时、路径、IO、viewer、landmark 模块
- `scripts/`：辅助脚本，例如 demo 数据生成
- `assets/models/`：运行依赖的 FBX 模型
- `data/examples/`：手工维护的示例输入
- `outputs/`：脚本运行时生成的输出文件
- `docs/assets/`：README 使用的文档资源

## 人脸网格可视化器

统一 CLI 入口为 `face_blender_shape.cli`（`python -m face_blender_shape` 或安装后的 `face-blender-shape`）。当前仅提供子命令 `preview`，与 `face_blender_shape/cli.py` 一致。

```bash
python -m face_blender_shape preview --path data/examples/sample_data.csv
# 若已通过 pip/uv 安装包，也可：
# face-blender-shape preview --path data/examples/sample_data.csv
```

查看全部参数：

```bash
python -m face_blender_shape preview -h
```

`preview` 选项说明：

| 选项 | 说明 |
|------|------|
| `--path` | 必填。CSV 路径；每行列数与 `constants.FRAME_WIDTH`（与 `BLENDSHAPE_NAMES` 顺序一致）相同。 |
| `--fps` | 播放帧率，默认 `30.0`。 |
| `--fbx` | 覆盖默认头模 FBX 路径。 |
| `--wireframe-head` | 头壳仅线框，舌保持实体网格，便于观察舌形变。 |
| `--open3d-dual-view` | Open3D 双窗：侧面 + 正面。 |
| `--open3d-camera-zoom Z` | Open3D `set_zoom` 初始缩放（默认 `0.2`）。 |
| `--tongue-lo N` | 线框模式下舌顶点全局下标下界（含）；缺省与当前默认 FBX 一致。 |
| `--tongue-hi N` | 线框模式下舌顶点全局下标上界（不含）；缺省与当前默认 FBX 一致。 |
| `--tongue-adjacency-expand ITERS` | 沿共享边扩展舌三角面轮数，用于衔接区间外顶点；建议从小值试起。 |

兼容脚本 `blender_interface.py` 调用同一套 `preview_sequence`，参数含义与上表相同，但**无子命令**，且 `--path` 必填。`--open3d-camera-zoom` 在该脚本中默认 `0.6`（与统一 CLI 的默认值不同）。

```bash
python blender_interface.py --path data/examples/sample_data.csv
```

<img src="docs/assets/facevis.gif" alt="drawing" width="200" height="320"/>

## Blender Shape 转关键点

统一 CLI **没有** `convert` 子命令；请使用顶层兼容脚本：

```bash
python sranipal2keypoints.py --path data/examples/sample_data.csv
```

默认会在 `outputs/` 下生成对应的 `.npz` 文件，例如 `outputs/sample_data.npz`。

## 生成 tongue demo

```bash
python scripts/generate_tongue_demo.py
```

生成结果会写入 `outputs/tongue_demo.csv`。
