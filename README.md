# Face Blender Shape Python

## 简介

这是一个基于 Python 的面部 BlendShape 预览工具：

- 使用 `bpy` 导入并驱动仓库内置的 `sranipal_head.fbx`
- 读取每帧 52 列的 BlendShape CSV
- 通过 Blender 求值得到变形后的网格
- 使用 Open3D 实时预览头模形变

当前仓库的**稳定公开能力**聚焦在“预览 CSV 驱动的面部动画”。README 只描述当前已经落地并能从仓库中直接找到的功能。

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

## 当前目录概览

```text
FaceBlenderShape/
├── blender_interface.py
├── face_blender_shape/
├── assets/models/
├── docs/assets/
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

- `blender_interface.py`：顶层兼容入口，用于播放 BlendShape CSV
- `face_blender_shape/`：核心运行时、CLI、Open3D viewer、常量与网格拆分逻辑
- `assets/models/`：运行依赖的 FBX 模型
- `docs/assets/`：README 使用的文档资源

## 输入格式

输入 CSV 的每一行表示一帧，每帧必须包含 **52 列**，列顺序需要与代码中的 `face_blender_shape/constants.py` 里的 `BLENDSHAPE_NAMES` 完全一致。

如果首行不是数值，程序会把它当作表头并自动跳过。

## 预览头模动画

### 方式一：使用兼容入口脚本

```bash
uv run python blender_interface.py --path /path/to/your_blendshape.csv
```

### 方式二：使用统一 CLI

```bash
uv run face-blender-shape preview --path /path/to/your_blendshape.csv
```

### 常用参数

```bash
uv run face-blender-shape preview \
  --path /path/to/your_blendshape.csv \
  --fps 30 \
  --fbx /path/to/your_head.fbx \
  --wireframe-head \
  --tongue-lo 180 \
  --tongue-hi 314 \
  --tongue-adjacency-expand 1
```

参数说明：

- `--path`：必填，BlendShape CSV 路径
- `--fps`：播放帧率
- `--fbx`：覆盖默认 FBX 路径
- `--wireframe-head`：头壳使用线框、舌体保持实体网格，便于观察舌部形变
- `--tongue-lo` / `--tongue-hi`：在线框模式下指定舌区域顶点下标范围
- `--tongue-adjacency-expand`：沿共享边扩展舌面邻接轮数，用于补齐舌体边界三角面

<img src="docs/assets/facevis.gif" alt="Face preview" width="200" height="320"/>

## 当前默认模型约定

默认使用的模型位于：

```text
assets/models/sranipal_head.fbx
```

当前运行时对默认模型有以下约定：

- 头部对象名为 `Head`
- BlendShape 通道与 `BLENDSHAPE_NAMES` 一一对应
- 舌区域的默认顶点下标范围与仓库内置模型一致

这意味着它目前更适合驱动与默认 FBX 拓扑兼容的头模，而不是任意未知拓扑的人脸模型。

## 当前未在 README 中承诺的能力

当前 README **不再声明**以下能力，因为它们未作为当前仓库主线能力稳定暴露：

- 关键点导出命令
- `face-blender-shape convert` 子命令
- `sranipal2keypoints.py` 顶层脚本
- 仓库内置示例 CSV
- tongue demo 生成脚本

如果后续这些能力重新加入主线接口，再补回 README。
