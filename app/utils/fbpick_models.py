"""Helpers for fbpick model discovery and resolution."""

from __future__ import annotations

from pathlib import Path

from app.services.errors import ConflictError, NotFoundError, UnprocessableError

_MODEL_GLOB = 'fbpick_*.pth'


def fbpick_model_dir() -> Path:
    """Return the repository model directory path."""
    return Path(__file__).resolve().parents[2] / 'model'


def list_fbpick_models() -> list[dict[str, bool | str]]:
    """Return sorted fbpick model candidates from ``model/``."""
    model_dir = fbpick_model_dir()
    models: list[dict[str, bool | str]] = []
    for path in model_dir.glob(_MODEL_GLOB):
        if not path.is_file():
            continue
        model_id = path.name
        models.append({'id': model_id, 'uses_offset': 'offset' in model_id.lower()})
    return sorted(models, key=lambda item: str(item['id']))


def validate_model_id(model_id: str) -> str:
    """Validate a model identifier that is expected to be a file name."""
    if Path(model_id).name != model_id:
        raise UnprocessableError('model_id must be a plain file name')
    if not model_id.startswith('fbpick_') or not model_id.endswith('.pth'):
        raise UnprocessableError(
            "model_id must start with 'fbpick_' and end with '.pth'"
        )
    return model_id


def resolve_model_path(
    model_id: str | None,
    *,
    default_id: str = 'fbpick_edgenext_small.pth',
    require_exists: bool,
) -> Path:
    """Resolve ``model_id`` to a model file path under ``model/``."""
    chosen_id = default_id if model_id is None else model_id
    validate_model_id(chosen_id)
    path = fbpick_model_dir() / chosen_id
    if not require_exists:
        return path
    if path.is_file():
        return path
    if model_id is None:
        raise ConflictError(f'Default fbpick model not found: {chosen_id}')
    raise NotFoundError(f'fbpick model not found: {chosen_id}')


def model_version(path: Path) -> str:
    """Return cache/version key from file name and mtime nanoseconds."""
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return path.name
    return f'{path.name}:{mtime_ns}'
