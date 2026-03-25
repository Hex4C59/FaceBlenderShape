# 数字人预览（当前仓库行为说明）

运行时 **固定加载** `models/Metahuman_Head.fbx`，CSV 仍为 **SRanipal 37 维**，内部 **SRanipal → ARKit** 映射后驱动该头模。

若 **`head` 网格或 `extra_meshes` 中任一对象**带有与 SRanipal 列名一致的 **`Tongue_*` 形态键**（见 `constants.BLENDSHAPE_NAMES`），启动时会自动启用 **舌头直连**：这些通道按 CSV 数值写入对应形态键，且 **SRanipal 中全部舌头行不再参与 ARKit 映射**（避免 `Tongue_LongStep*` 再叠加到 `jawOpen`）。无 `Tongue_*` 键时行为与此前一致。

## 配置方式

使用项目根目录 **`face_blender_preview.yaml`**（或 `run.py --config` 指向其它 YAML）：

- **`path`**：blendshape CSV（必填）。
- **`head`**：可选；覆盖 FBX 里作为主脸的网格对象名（默认 MetaHuman 头对象名见 `constants.py`）。
- **`extra_meshes`**：可选；逗号分隔或 YAML 列表，指定额外合并进 Open3D 的网格名（默认仍为牙齿与双眼）。

**不再支持**：在配置或代码中切换其它 FBX 路径、外部 `texture` 贴图加载与 UV 烘焙；头部无合并附件时由 Open3D 使用默认肤色顶点色，附件部件仍按材质 Base Color 近似。

## 查看 FBX 中的物体名

本仓库不再附带「列出网格 / shape key」的辅助脚本；需要核对对象名时，请在 **Blender** 中打开对应 FBX，在 Outliner 与 Shape Keys 面板中查看（运行时仍只加载内置 MetaHuman FBX）。

## 展示级画质

Open3D 仅为快速校验形变；成片请在 Blender / Unreal 中使用同一套数据与资产渲染。
