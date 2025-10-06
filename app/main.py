"""FastAPI application entry point."""

import json
from pathlib import Path
from typing import Any

FILE_REGISTRY: dict[str, dict[str, Any]] = {}


def read_segy_dt_seconds(path: str) -> float | None:
        """Return dt in seconds read from SEG-Y binary header, or None on failure."""
        try:
                with open(path, "rb") as f:
                        f.seek(3200 + 16)
                        raw = f.read(2)
                if len(raw) != 2:
                        return None
                us = int.from_bytes(raw, byteorder="big", signed=False)
                if us <= 0:
                        return None
                return us / 1_000_000.0
        except Exception:  # noqa: BLE001
                return None


def get_dt_for_file(file_id: str) -> float:
        rec = FILE_REGISTRY.get(file_id)
        if not isinstance(rec, dict):
                rec = {}

        dt_val = rec.get("dt")
        if isinstance(dt_val, (int, float)) and dt_val > 0:
                return float(dt_val)

        path = rec.get("path")
        if not path:
                store_path = rec.get("store_path")
                if isinstance(store_path, str):
                        meta_path = Path(store_path) / "meta.json"
                        try:
                                meta = json.loads(meta_path.read_text())
                        except Exception:  # noqa: BLE001
                                meta = None
                        if isinstance(meta, dict):
                                meta_dt = meta.get("dt")
                                if isinstance(meta_dt, (int, float)) and meta_dt > 0:
                                        rec["dt"] = float(meta_dt)
                                        FILE_REGISTRY[file_id] = rec
                                        return float(meta_dt)
                                original_path = meta.get("original_segy_path")
                                if isinstance(original_path, str):
                                        path = original_path
                                        rec["path"] = path

        dt = read_segy_dt_seconds(path) if path else None
        if not dt:
                dt = 0.002
        rec["dt"] = dt
        FILE_REGISTRY[file_id] = rec
        return dt


from api import endpoints
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from utils.picks import store

app = FastAPI()


@app.on_event('startup')
async def load_picks() -> None:
	"""Load picks from disk at startup."""
	store.load()


@app.on_event('shutdown')
async def save_picks() -> None:
	"""Save picks to disk when shutting down."""
	store.save()


# 静的ファイル (HTML, JS)
app.mount(
	'/static',  # ← URLパス
	StaticFiles(directory='/workspace/app/static'),  # ← ローカルパス
	name='static',
)
# エンドポイント登録
app.include_router(endpoints.router)


@app.get('/', response_class=HTMLResponse)
async def index() -> str:
	"""Return the main page."""
	index_path = Path('/workspace/app/static/index.html')
	return index_path.read_text(encoding='utf-8')


@app.get('/upload', response_class=HTMLResponse)
async def upload() -> str:
	"""Return the upload page."""
	upload_path = Path('/workspace/app/static/upload.html')
	return upload_path.read_text(encoding='utf-8')
