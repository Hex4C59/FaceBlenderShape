"""兼容导出：保留旧路径工具导入路径。"""

from __future__ import annotations

from face_blender_shape.core.paths import (
    DATA_DIR,
    DEFAULT_SAMPLE_CSV_PATH,
    DOCS_ASSETS_DIR,
    METAHUMAN_FBX_PATH,
    MODELS_DIR,
    PROJECT_ROOT,
    resolve_fbx_path,
    resolve_input_csv_path,
)

__all__ = [
    "DATA_DIR",
    "DEFAULT_SAMPLE_CSV_PATH",
    "DOCS_ASSETS_DIR",
    "METAHUMAN_FBX_PATH",
    "MODELS_DIR",
    "PROJECT_ROOT",
    "resolve_fbx_path",
    "resolve_input_csv_path",
]
