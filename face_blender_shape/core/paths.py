"""项目根目录、资源目录与输入路径解析。"""

from __future__ import annotations

from pathlib import Path

from face_blender_shape.core.asset_names import METAHUMAN_FBX

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
DOCS_ASSETS_DIR = PROJECT_ROOT / "docs" / "assets"
METAHUMAN_FBX_PATH = MODELS_DIR / METAHUMAN_FBX
DEFAULT_SAMPLE_CSV_PATH = DATA_DIR / "sample_data.csv"


def _resolve_existing_path(path: str | Path, fallback_dirs: tuple[Path, ...]) -> Path:
    """在候选目录中查找真实存在的文件。

    path: 用户传入的路径；相对路径会按当前工作目录、项目根目录和回退目录依次查找。
    fallback_dirs: 额外搜索目录元组。
    """
    candidate = Path(path).expanduser()
    search_candidates: list[Path] = []

    if candidate.is_absolute():
        search_candidates.append(candidate)
    else:
        search_candidates.append(Path.cwd() / candidate)
        search_candidates.append(PROJECT_ROOT / candidate)
        search_candidates.extend(directory / candidate for directory in fallback_dirs)

    for resolved in search_candidates:
        if resolved.exists():
            return resolved.resolve()

    raise FileNotFoundError(f"Could not find file: {path}")


def resolve_fbx_path(path: str | Path | None = None) -> Path:
    """解析 FBX 文件路径。

    path: 显式传入的 FBX 路径；为 None 时返回项目默认 MetaHuman FBX。
    """
    if path is None:
        return METAHUMAN_FBX_PATH.resolve()
    return _resolve_existing_path(path, (MODELS_DIR,))


def resolve_input_csv_path(path: str | Path) -> Path:
    """解析输入 CSV 文件路径。

    path: 输入 CSV 路径；相对路径会在数据目录等候选位置查找。
    """
    return _resolve_existing_path(path, (DATA_DIR,))
