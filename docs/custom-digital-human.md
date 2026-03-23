# 接入「带贴图 / 毛发」的数字人（ARKit blendshape）

本仓库的 **CSV 仍是 SRanipal 37 维**；若你的角色使用 **ARKit 标准 52 形变**（或兼容子集），可用 `--model metahuman` 走内置的 **SRanipal → ARKit** 映射，并把 **任意 FBX** 指给运行时。

## 你需要什么资产

1. **面部主网格**（`--head`）：含 **ARKit 命名** 的 shape keys（与 Apple `ARFaceAnchor` / PerfectSync 常见命名一致，如 `jawOpen`、`mouthSmileLeft`…）。子集也可，缺的形变会保持 0。
2. **可选附加网格**（`--extra-meshes`）：牙齿、眼球、**多边形头发** 等，按 **逗号分隔的物体名** 列出。它们会与头部 **合并** 到 Open3D 里一起显示。
3. **贴图**：优先使用 FBX 里材质已引用的贴图（自定义 FBX 时会尝试从 bpy 读取）；也可继续用 `--texture` 指定一张头部 albedo。

## 毛发注意事项

- **Unreal Groom / 粒子发** 不会直接进本项目的 Open3D 预览。需要先在 **Blender / UE** 里 **转成网格**（发卡片、发束烘焙、或第三方插件），再随 FBX 导出。
- 头发网格通常 **没有** 面部 52 形变：不会随嘴型动，这是正常现象；若要与头皮一起变形，需在 DCC 里把相同形变拷到发片或使用蒙皮跟随（超出本文范围）。

## 查看 FBX 里的物体名

```bash
uv run --no-sync python scripts/list_fbx_meshes.py /path/to/avatar.fbx
uv run --no-sync python scripts/list_fbx_meshes.py /path/to/avatar.fbx --shape-keys
```

## 运行示例

```bash
uv run --no-sync python blender_interface.py \
  --model metahuman \
  --fbx /path/to/your_avatar.fbx \
  --head Face_Geo \
  --extra-meshes "Teeth_Geo,Eye_L,Eye_R,Hair_Geo" \
  --path data/examples/sample_data.csv \
  --fps 15
```

- `--head`：驱动 blendshape 的主物体（必须存在且为 MESH）。
- `--extra-meshes`：按 **显示顺序** 追加合并；写错的名称会打印 warning 并跳过。

## 展示级画质

Open3D 仅为快速校验形变；**成片展示**请在 Blender / Unreal 中导入同一套 FBX，用 **同一套 CSV 或 Live Link** 驱动，以获得 PBR、毛发、景深与正确眼球。

## 仍是 SRanipal 形变命名怎么办？

若资产只有 **SRanipal 37** 名称，请使用默认 `--model sranipal`（或保证 `--head` 物体上是这 37 个 key）。ARKit 与 SRanipal **混用** 需自行在 Blender 里重定向或在本仓库扩展映射表（`blendshape_mapping.py`）。
