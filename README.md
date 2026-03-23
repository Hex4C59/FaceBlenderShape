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

仓库中还包含一个固定为 `3.10` 的 `.python-version` 文件，因此默认执行 `uv sync` 和 `uv run` 时会自动使用正确的解释器。

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

直接运行兼容脚本：

```bash
uv run python blender_interface.py
```

播放示例 CSV：

```bash
uv run python blender_interface.py --path data/examples/sample_data.csv
```

也可以使用统一 CLI：

```bash
uv run face-blender-shape preview --path data/examples/sample_data.csv
```

<img src="docs/assets/facevis.gif" alt="drawing" width="200" height="320"/>

## Blender Shape 转关键点

使用兼容脚本：

```bash
uv run python sranipal2keypoints.py --path data/examples/sample_data.csv
```

或使用统一 CLI：

```bash
uv run face-blender-shape convert --path data/examples/sample_data.csv
```

默认会在 `outputs/` 下生成对应的 `.npz` 文件，例如 `outputs/sample_data.npz`。

## 生成 tongue demo

```bash
uv run python scripts/generate_tongue_demo.py
```

生成结果会写入 `outputs/tongue_demo.csv`。
