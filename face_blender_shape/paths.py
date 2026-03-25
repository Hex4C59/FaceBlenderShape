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
    if path is None:
        return METAHUMAN_FBX_PATH.resolve()
    return _resolve_existing_path(path, (MODELS_DIR,))


def resolve_input_csv_path(path: str | Path) -> Path:
    return _resolve_existing_path(path, (DATA_DIR,))
