"""项目根、模型与数据目录等路径常量，以及 FBX 与 CSV 输入路径的解析与回退查找。"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
DOCS_ASSETS_DIR = PROJECT_ROOT / "docs" / "assets"
METAHUMAN_FBX_PATH = MODELS_DIR / "Metahuman_Head.fbx"
DEFAULT_SAMPLE_CSV_PATH = DATA_DIR / "sample_data.csv"


def _resolve_existing_path(path: str | Path, fallback_dirs: tuple[Path, ...]) -> Path:
    """
    在用户给定路径及若干回退目录中查找第一个真实存在的文件，并返回规范绝对路径。

    path: 用户传入的路径字符串或 Path；相对路径时依次在 cwd、项目根、fallback_dirs 下拼接查找。
    fallback_dirs: 相对路径的额外搜索根目录元组（如 models、data）。
    """
    candidate = Path(path).expanduser()
    search_candidates = []

    if candidate.is_absolute():
        search_candidates.append(candidate)
    else:
        search_candidates.extend(
            [
                Path.cwd() / candidate,
                PROJECT_ROOT / candidate,
                *(directory / candidate for directory in fallback_dirs),
            ]
        )

    for resolved in search_candidates:
        if resolved.exists():
            return resolved.resolve()

    raise FileNotFoundError(f"Could not find file: {path}")


def resolve_fbx_path(path: str | Path | None = None) -> Path:
    """
    解析 MetaHuman 等 FBX 模型文件路径；未指定时使用仓库默认 Metahuman_Head.fbx。

    path: 显式 FBX 路径；为 None 时返回 METAHUMAN_FBX_PATH；为相对路径时按 _resolve_existing_path 规则在 MODELS_DIR 等位置查找。
    """
    if path is None:
        return METAHUMAN_FBX_PATH.resolve()
    return _resolve_existing_path(path, (MODELS_DIR,))


def resolve_input_csv_path(path: str | Path) -> Path:
    """
    解析 blendshape CSV 输入路径；相对路径时在 cwd、项目根与 DATA_DIR 下查找存在的文件。

    path: CSV 文件路径字符串或 Path。
    """
    return _resolve_existing_path(path, (DATA_DIR,))
