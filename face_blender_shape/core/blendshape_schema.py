"""SRanipal blendshape 维度定义与索引表。"""

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
