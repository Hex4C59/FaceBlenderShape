from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = ASSETS_DIR / "models"
TEXTURES_DIR = ASSETS_DIR / "textures"
DATA_DIR = PROJECT_ROOT / "data"
EXAMPLES_DIR = DATA_DIR / "examples"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DOCS_ASSETS_DIR = PROJECT_ROOT / "docs" / "assets"
DEFAULT_FBX_PATH = MODELS_DIR / "sranipal_head.fbx"
METAHUMAN_FBX_PATH = MODELS_DIR / "Metahuman_Head.fbx"
DEFAULT_SAMPLE_CSV_PATH = EXAMPLES_DIR / "sample_data.csv"


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
        return DEFAULT_FBX_PATH.resolve()
    return _resolve_existing_path(path, (MODELS_DIR,))


def resolve_texture_path(path: str | Path | None = None) -> Path | None:
    """Resolve a texture image path, returning *None* when not found.

    Search order: explicit *path* → ``assets/textures/`` directory
    (first ``.png`` / ``.jpg`` / ``.jpeg`` / ``.tga`` found).
    """
    if path is not None:
        try:
            return _resolve_existing_path(path, (TEXTURES_DIR, MODELS_DIR))
        except FileNotFoundError:
            return None

    if not TEXTURES_DIR.is_dir():
        return None
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tga"):
        hits = sorted(TEXTURES_DIR.glob(ext))
        if hits:
            return hits[0].resolve()
    return None


def resolve_input_csv_path(path: str | Path) -> Path:
    return _resolve_existing_path(path, (EXAMPLES_DIR,))


def resolve_output_path(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    *,
    suffix: str,
) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        if input_path is None:
            raise ValueError("input_path is required when output_path is omitted")
        input_candidate = Path(input_path)
        return (OUTPUTS_DIR / f"{input_candidate.stem}{suffix}").resolve()

    candidate = Path(output_path).expanduser()
    if candidate.is_absolute():
        resolved = candidate
    elif candidate.parent == Path("."):
        resolved = OUTPUTS_DIR / candidate
    else:
        resolved = PROJECT_ROOT / candidate

    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved.resolve()
