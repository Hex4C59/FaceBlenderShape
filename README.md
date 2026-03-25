# Face Blender Shape Python

## 简介

如果你使用 Vive Facial Tracker 作为真值系统，并希望获取毫米尺度的人脸关键点数据，或者你想在不依赖 Unity 的情况下可视化 SRanipal Blender Shape 的面部效果，那么这个仓库可能会对你有帮助。

当前仓库的**日常启动方式**是：编辑根目录下的 `face_blender_preview.yaml`，再执行 **`run.py`**（见下文「启动预览」）。业务参数集中在 YAML 里，命令行只负责指定配置文件路径。

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
├── run.py
├── face_blender_preview.yaml
├── face_blender_shape/
├── models/
├── data/
├── outputs/
└── docs/assets/
```

- `run.py`：命令行入口，读取 YAML 后启动 Open3D 预览（实现上调用 `face_blender_shape.cli`）。
- `face_blender_preview.yaml`：默认预览配置；可用 `run.py --config` 换成其它路径。
- `face_blender_shape/`：核心运行时、路径、IO、viewer、landmark 与 CLI 解析逻辑。
- `models/`：运行依赖的 FBX 模型（当前运行时固定加载 MetaHuman 头模，见下文）。
- `data/`：CSV 等输入数据（如 `sample_data.csv`）。
- `outputs/`：若你自行编写导出逻辑，可将输出放在此目录（本仓库默认预览不写关键点 npz）。
- `docs/assets/`：README 使用的文档资源。

## 启动预览

1. 在 **`face_blender_preview.yaml`** 中设置 `path`（必填）：指向 SRanipal 37 维 blendshape CSV；其余键（`fps`、`view_scale`、`window_width`、`window_height`、`head`、`extra_meshes`）可选，含义见该文件内注释与下表。

2. 在仓库根目录执行：

```bash
uv run python run.py
```

若已 `source .venv/bin/activate`，可直接：

```bash
python run.py
```

3. 使用其它配置文件时：

```bash
uv run python run.py --config path/to/your_preview.yaml
```

### YAML 常用字段

| 字段 | 说明 |
|------|------|
| `path` | blendshape CSV 路径（必填）。 |
| `fps` | 播放帧率；`<=0` 时不按帧延时。 |
| `view_scale` | Open3D 取景比例，越大脸部在画面中越大。 |
| `window_width` / `window_height` | 预览窗口宽高（像素）。 |
| `head` | 主脸网格在场景中的对象名；省略则用 MetaHuman 默认头对象名。 |
| `extra_meshes` | 额外合并进预览的网格名，逗号分隔字符串或 YAML 列表（如牙齿、头发子网格）。 |
| `dual_view` | `true` 时除正脸外再开一个 **侧视** Open3D 窗口（同一网格同步刷新）。 |
| `fbx` | 自定义头部 FBX 路径；省略则使用 `resolve_fbx_path(None)` 的默认（通常为 `models/Metahuman_Head.fbx`）。 |

<img src="docs/assets/facevis.gif" alt="drawing" width="200" height="320"/>

### 默认头模为什么看起来「不像真人」？

内置的 **MetaHuman 风格头**（`models/Metahuman_Head.fbx`）相对 SRanipal 官方低模已更细腻，但 Open3D 窗口仍是 **科研向快速预览**：顶点色 + 简单光照，**没有** PBR、次表面散射、毛发、眼球折射或电影级材质。人脑对「像人但缺细节的脸」特别敏感（恐怖谷）。若要给他人做 **展示或录制**，更合适的做法是：在此校验形变后，在 **Blender / Unreal** 里用同一套数据与资产渲染。

为减轻默认预览的「油光塑料感」，查看器会对顶点色做轻度 **压亮（matte gamma）**，可在 `constants.DEFAULT_OPEN3D_VERTEX_MATTE_GAMMA` 微调（约 `1.0`～`1.25`，越大越哑光、整体越暗）。

默认还会把 **Half-Lambert 漫反射** 烘焙进顶点色并 **关闭场景灯**（避免 Open3D 高光叠加成塑料感），可在 `constants` 里开关与强度：`DEFAULT_OPEN3D_BAKED_SHADING`、`DEFAULT_OPEN3D_BAKED_AMBIENT`、`DEFAULT_OPEN3D_BAKED_DIFFUSE`。

### 数字人资产与自定义网格名

默认从 `models/Metahuman_Head.fbx` 加载（若不存在则需在 YAML 里设置 `fbx` 指向仓库内其它头模，如 `models/Avatar_Shieh_V2.fbx` 并正确填写 `head`）。CSV 仍为 **SRanipal 37 维**，MetaHuman 路径下会做 **SRanipal → ARKit** 映射；**舌头多通道在 MetaHuman 上映射很弱**，要看舌头形变宜使用带完整 `Tongue_*` shape key 的 FBX。更完整的说明见 **[docs/custom-digital-human.md](docs/custom-digital-human.md)**。

### 平面轨迹 → blendshape 与 mock 联调（步骤 3–5）

- **映射**：[`face_blender_shape/trajectory_mapping.py`](face_blender_shape/trajectory_mapping.py) 将一系列 `(x, y)`（归一化平面坐标）经启发式规则写入 37 维中的舌头与 `Jaw_Open`（默认带基础张嘴，便于侧视观察）。
- **一键联调脚本**（模拟「说话 → 轨迹 → 驱动」；尚未接真实音频/超声）：

```bash
# 默认头模为 models/Metahuman_Head.fbx，无需 --fbx / --head
uv run python scripts/mock_talk_pipeline.py --dual --save-csv outputs/mock_from_traj.csv
```

`--dual` 打开 **主视 + 侧视** 两个窗口。后续可把 `mock_trajectory_from_mock_audio` 换成「读 FINAL 提取的 wav2vec + 你自己的轨迹规则」，再把真实超声特征接进同一映射函数。

**manifest 式训练数据**若你自行采集：配对 `video_path,blendshape_csv_path` 的约定仍可用于有监督回归（见仓库内其它文档/计划）；本段仅覆盖 **无训练集时的几何映射 + 双视预览**。

### 为什么预览仍可能「有点吓人」？

说明同上：Open3D 仅为快速校验，并非成片渲染器。

## Agent 组件架构示意

![Agent 组件：用户目标、LLM Core、Executor 等](docs/assets/agent_components.svg)
