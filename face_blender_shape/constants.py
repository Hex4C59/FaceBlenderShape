"""全项目共用的 BlendShape 名称顺序与默认配置。

``BLENDSHAPE_NAMES`` 顺序与 ``sranipal_head.fbx`` 导入 Blender 后、除 Basis 外的
形态键顺序一致（口周 → 眼 → 舌），末尾追加本项目自定义通道。
CSV 每行须与此顺序一一对应。

Basis 为基准形，不由 CSV 驱动。

勿在元组中间增删改名称，否则旧 CSV 与已导出帧数据会与通道错位。
"""

from __future__ import annotations

# ---------- 形态键：前 52 个与 sranipal_head.fbx 一致，末尾为自定义扩展 ----------

BLENDSHAPE_NAMES: tuple[str, ...] = (
    # --- 口颌与面颊（与旧版 SRanipal 37 通道中前 26 个顺序相同）---
    "Jaw_Left",  # 下颌向左
    "Jaw_Right",  # 下颌向右
    "Jaw_Forward",  # 下颌前伸
    "Jaw_Open",  # 张嘴（下颌张开）
    "Mouth_Ape_Shape",  # 嘴大幅张开（类似噘张大口）
    "Mouth_Upper_Left",  # 上唇左侧
    "Mouth_Upper_Right",  # 上唇右侧
    "Mouth_Lower_Left",  # 下唇左侧
    "Mouth_Lower_Right",  # 下唇右侧
    "Mouth_Upper_Overturn",  # 上唇外翻
    "Mouth_Lower_Overturn",  # 下唇外翻
    "Mouth_Pout",  # 撅嘴
    "Mouth_Smile_Left",  # 左侧微笑
    "Mouth_Smile_Right",  # 右侧微笑
    "Mouth_Sad_Left",  # 左侧嘴角下垂/撇嘴
    "Mouth_Sad_Right",  # 右侧嘴角下垂/撇嘴
    "Cheek_Puff_Left",  # 左腮鼓起
    "Cheek_Puff_Right",  # 右腮鼓起
    "Cheek_Suck",  # 吸腮（两颊内收）
    "Mouth_Upper_UpLeft",  # 上唇左上提拉
    "Mouth_Upper_UpRight",  # 上唇右上提拉
    "Mouth_Lower_DownLeft",  # 下唇左下压
    "Mouth_Lower_DownRight",  # 下唇右下压
    "Mouth_Upper_Inside",  # 上唇向内卷
    "Mouth_Lower_Inside",  # 下唇向内卷
    "Mouth_Lower_Overlay",  # 下唇叠在上唇之上
    # --- 眼部（15）；旧版 37 列 CSV 无此段，迁移时中间补 0 ---
    "Eye_Left_Blink",  # 左眼眨眼
    "Eye_Left_Wide",  # 左眼睁大
    "Eye_Left_Right",  # 左眼朝右看
    "Eye_Left_Left",  # 左眼朝左看
    "Eye_Left_Up",  # 左眼朝上看
    "Eye_Left_Down",  # 左眼朝下看
    "Eye_Right_Blink",  # 右眼眨眼
    "Eye_Right_Wide",  # 右眼睁大
    "Eye_Right_Right",  # 右眼朝右看
    "Eye_Right_Left",  # 右眼朝左看
    "Eye_Right_Up",  # 右眼朝上看
    "Eye_Right_Down",  # 右眼朝下看
    "Eye_Frown",  # 眉间皱眉
    "Eye_Left_squeeze",  # 左眼眯眼
    "Eye_Right_squeeze",  # 右眼眯眼
    # --- 舌（11）；与旧版 37 列中最后 11 个顺序相同 ---
    "Tongue_LongStep1",  # 舌头伸出（第一段）
    "Tongue_LongStep2",  # 舌头伸出（第二段，更长）
    "Tongue_Left",  # 舌头向左
    "Tongue_Right",  # 舌头向右
    "Tongue_Up",  # 舌头上翘
    "Tongue_Down",  # 舌头下压
    "Tongue_Roll",  # 卷舌
    "Tongue_UpLeft_Morph",  # 舌左上方向形变
    "Tongue_UpRight_Morph",  # 舌右上方向形变
    "Tongue_DownLeft_Morph",  # 舌左下方向形变
    "Tongue_DownRight_Morph",  # 舌右下方向形变
    # --- 自定义扩展（FBX 中不存在，由 blender_runtime 自动创建） ---
    "Tongue_Dorsum_Arch",  # 舌背拱起：两头低中间高（bell curve 位移）
)

BLENDSHAPE_INDEX: dict[str, int] = {
    name: idx for idx, name in enumerate(BLENDSHAPE_NAMES)
}
FRAME_WIDTH: int = len(BLENDSHAPE_NAMES)

BASIS_SHAPE_KEY_NAME: str = "Basis"
# FBX 中原有的标准形态键数（不含 Basis，不含自定义扩展）
SRANIPAL_STANDARD_SHAPE_KEY_COUNT: int = 52

# ---------- 预览与可视化默认 ----------

DEFAULT_PLAYBACK_FPS: float = 30.0  # CLI 顺序预览时相邻帧 sleep 依据
# 预览时若按 fps 算出的整段时长短于此值（秒），则自动拉长每帧间隔（帧少则放慢）
PREVIEW_MIN_SEQUENCE_SECONDS: float = 5
DEFAULT_OPEN3D_WINDOW_NAME: str = "Face Blender Shape Viewer"
