from __future__ import annotations

# MetaHuman / SRanipal 风格面部 BlendShape 名称顺序（与 CSV 列一致）
BLENDSHAPE_NAMES = (
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
)

BLENDSHAPE_INDEX = {name: idx for idx, name in enumerate(BLENDSHAPE_NAMES)}
FRAME_WIDTH = len(BLENDSHAPE_NAMES)
DEFAULT_PLAYBACK_FPS = 30.0
DEFAULT_OPEN3D_WINDOW_NAME = "Face Blender Shape Viewer"
DEFAULT_HEAD_OBJECT_NAME = "Head"

METAHUMAN_HEAD_OBJECT_NAME = "head_lod0_ORIGINAL"
METAHUMAN_TEETH_OBJECT_NAME = "teeth_ORIGINAL"
METAHUMAN_FBX = "Metahuman_Head.fbx"
