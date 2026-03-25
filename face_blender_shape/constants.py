"""全局常量：SRanipal 37 维 blendshape 名称与帧列数、Open3D 默认窗口与渲染参数、MetaHuman 场景对象名与资源文件名。"""

from __future__ import annotations

BLENDSHAPE_NAMES = (
    "Jaw_Left",
    "Jaw_Right",
    "Jaw_Forward",
    "Jaw_Open",
    "Mouth_Ape_Shape",
    "Mouth_Upper_Left",
    "Mouth_Upper_Right",
    "Mouth_Lower_Left",
    "Mouth_Lower_Right",
    "Mouth_Upper_Overturn",
    "Mouth_Lower_Overturn",
    "Mouth_Pout",
    "Mouth_Smile_Left",
    "Mouth_Smile_Right",
    "Mouth_Sad_Left",
    "Mouth_Sad_Right",
    "Cheek_Puff_Left",
    "Cheek_Puff_Right",
    "Cheek_Suck",
    "Mouth_Upper_UpLeft",
    "Mouth_Upper_UpRight",
    "Mouth_Lower_DownLeft",
    "Mouth_Lower_DownRight",
    "Mouth_Upper_Inside",
    "Mouth_Lower_Inside",
    "Mouth_Lower_Overlay",
    "Tongue_LongStep1",
    "Tongue_LongStep2",
    "Tongue_Left",
    "Tongue_Right",
    "Tongue_Up",
    "Tongue_Down",
    "Tongue_Roll",
    "Tongue_UpLeft_Morph",
    "Tongue_UpRight_Morph",
    "Tongue_DownLeft_Morph",
    "Tongue_DownRight_Morph",
)

BLENDSHAPE_INDEX = {name: idx for idx, name in enumerate(BLENDSHAPE_NAMES)}
FRAME_WIDTH = len(BLENDSHAPE_NAMES)
DEFAULT_PLAYBACK_FPS = 30.0
DEFAULT_OPEN3D_WINDOW_NAME = "Face Blender Shape Viewer"
DEFAULT_OPEN3D_WIDTH = 1280
DEFAULT_OPEN3D_HEIGHT = 900
# Soft cool-gray backdrop (pure white reads clinical / “specimen” and worsens uncanny valley).
DEFAULT_OPEN3D_BACKGROUND_RGB = (0.86, 0.89, 0.93)
# >1.0 darkens bright albedo (less “plastic shine” under Open3D’s fixed lighting).
DEFAULT_OPEN3D_VERTEX_MATTE_GAMMA = 1.14
# Bake soft diffuse into vertex colors and turn off scene lights (reduces plastic specular).
DEFAULT_OPEN3D_BAKED_SHADING = False
DEFAULT_OPEN3D_BAKED_AMBIENT = 0.24
DEFAULT_OPEN3D_BAKED_DIFFUSE = 0.78
# 无贴图时用法线变化与空间噪声伪造微褶皱/毛孔（Open3D 无法采样材质法线贴图）。
DEFAULT_OPEN3D_SKIN_DETAIL_ENABLED = True
DEFAULT_OPEN3D_SKIN_CREASE_STRENGTH = 0.45
DEFAULT_OPEN3D_SKIN_MICRO_STRENGTH = 0.085
DEFAULT_OPEN3D_SKIN_MICRO_FREQ = 95.0
# >1 pulls the camera closer (face fills more of the window). See Open3D ViewControl.set_zoom.
DEFAULT_VIEW_SCALE = 2.2
DEFAULT_HEAD_OBJECT_NAME = "Head"
SRANIPAL_EYE_LEFT_OBJECT_NAME = "Eye_Left"
SRANIPAL_EYE_RIGHT_OBJECT_NAME = "Eye_Right"

METAHUMAN_HEAD_OBJECT_NAME = "head_lod0_ORIGINAL"
METAHUMAN_TEETH_OBJECT_NAME = "teeth_ORIGINAL"
METAHUMAN_EYE_LEFT_OBJECT_NAME = "eyeLeft_ORIGINAL"
METAHUMAN_EYE_RIGHT_OBJECT_NAME = "eyeRight_ORIGINAL"
METAHUMAN_FBX = "Metahuman_Head.fbx"
# Open3D 观感预设名，见 visual_presets.py（neutral_bust / detailed）。
DEFAULT_OPEN3D_VISUAL_PRESET = "neutral_bust"
