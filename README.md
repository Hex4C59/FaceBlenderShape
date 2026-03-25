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

更新依赖时：改 `pyproject.toml` 后执行 `uv lock` / `uv sync`，锁文件由 `uv` 维护，不要手改 `uv.lock`。

## 目录概览

```text
FaceBlenderShape/
├── blender_interface.py
├── sranipal2keypoints.py
├── face_blender_shape/
├── scripts/
├── models/
├── data/
├── outputs/
└── docs/assets/
```

- `blender_interface.py`：保留的顶层兼容入口，用于 Blender 预览
- `sranipal2keypoints.py`：保留的顶层兼容入口，用于关键点导出
- `face_blender_shape/`：核心运行时、路径、IO、viewer、landmark 模块
- `scripts/`：辅助脚本，例如 demo 数据生成
- `models/`：运行依赖的 FBX 模型
- `data/`：CSV 等输入数据（如 `sample_data.csv`）
- `outputs/`：脚本运行时生成的输出文件
- `docs/assets/`：README 使用的文档资源

## 人脸网格可视化器

直接运行兼容脚本：

```bash
uv run python blender_interface.py
```

播放示例 CSV：

```bash
uv run python blender_interface.py --path data/sample_data.csv
```

也可以使用统一 CLI：

```bash
uv run face-blender-shape preview --path data/sample_data.csv
```

<img src="docs/assets/facevis.gif" alt="drawing" width="200" height="320"/>

### 默认头模为什么看起来「不像真人」？

默认的 **SRanipal 头**（`models/sranipal_head.fbx`）是为 **面部追踪实时性能** 做的低面数模型：网格棱角明显、细节少；眼睛区域往往是简化几何 + 顶点色，不会像照片级角色。这是设计取舍，不是渲染 bug。

### 更接近真人的头模（推荐）

本仓库已包含 **MetaHuman 风格、带 52 个 ARKit blendshape** 的高面数头部（`models/Metahuman_Head.fbx`，约 2.4 万顶点，含牙齿与眼球子网格）。用 **SRanipal 的 37 维 CSV** 仍可驱动，内部会做 **SRanipal → ARKit** 映射：

```bash
uv run --no-sync python blender_interface.py --model metahuman --path data/sample_data.csv --fps 15
```

需要更大脸部占比时可调 `--view-scale`（例如 `3.0`）和窗口大小，见脚本 `--help`。

**说明**：再往上要「照片级真人」，通常要在 Blender / UE 里换 **高精度扫描或 Metahuman 完整流程**（皮肤纹理、毛发、眼球折射等）。任意好看的头模若要接 Vive 数据，仍需 **与 SRanipal 或 ARKit 等 blendshape 命名兼容**，否则要自己重做映射或重定向。

### 自带贴图 / 毛发网格的数字人（自定义 FBX）

若资产是 **ARKit 标准 blendshape** 的头部 FBX（可含牙齿、眼球、**多边形头发** 等子网格），可用同一套 **SRanipal CSV** 驱动，并指定物体名：

```bash
uv run --no-sync python scripts/list_fbx_meshes.py /path/to/avatar.fbx --shape-keys

uv run --no-sync python blender_interface.py \
  --model metahuman \
  --fbx /path/to/avatar.fbx \
  --head YourFaceMeshName \
  --extra-meshes "Teeth_Low,Eye_L,Eye_R,Hair_Cards" \
  --path data/sample_data.csv
```

详细流程、毛发与展示级渲染说明见 **[docs/custom-digital-human.md](docs/custom-digital-human.md)**。

### 为什么预览仍可能「有点吓人」？

本项目的 Open3D 窗口是 **科研向快速预览**：顶点色 + 简单光照，**没有** PBR、次表面散射、睫毛、头发、眼球折射或电影级材质。人脑对「像人但缺细节的脸」特别敏感（恐怖谷）。若要给他人做 **展示或录制**，更合适的做法是：在此导出网格/曲线后，在 **Blender / Unreal** 里绑定材质与灯光再渲染。

为减轻默认预览的「油光塑料感」，查看器会对顶点色做轻度 **压亮（matte gamma）**，可在 `constants.DEFAULT_OPEN3D_VERTEX_MATTE_GAMMA` 微调（约 `1.0`～`1.25`，越大越哑光、整体越暗）。

默认还会把 **Half-Lambert 漫反射** 烘焙进顶点色并 **关闭场景灯**（避免 Open3D 高光叠加成塑料感），可在 `constants` 里开关与强度：`DEFAULT_OPEN3D_BAKED_SHADING`、`DEFAULT_OPEN3D_BAKED_AMBIENT`、`DEFAULT_OPEN3D_BAKED_DIFFUSE`。

## Blender Shape 转关键点

使用兼容脚本：

```bash
uv run python sranipal2keypoints.py --path data/sample_data.csv
```

或使用统一 CLI：

```bash
uv run face-blender-shape convert --path data/sample_data.csv
```

默认会在 `outputs/` 下生成对应的 `.npz` 文件，例如 `outputs/sample_data.npz`。

## 生成 tongue demo

```bash
uv run python scripts/generate_tongue_demo.py
```

生成结果会写入 `outputs/tongue_demo.csv`。

## Agent 组件架构示意

![Agent 组件：用户目标、LLM Core、Executor 等](docs/assets/agent_components.svg)
