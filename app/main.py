"""FastAPI application entry point."""

from pathlib import Path

from api import endpoints
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from utils.picks import store

app = FastAPI()


@app.on_event("startup")
async def load_picks() -> None:
    """Load picks from disk at startup."""
    store.load()


@app.on_event("shutdown")
async def save_picks() -> None:
    """Save picks to disk when shutting down."""
    store.save()


# 静的ファイル (HTML, JS)
app.mount(
    "/static",  # ← URLパス
    StaticFiles(directory="/workspace/app/static"),  # ← ローカルパス
    name="static",
)
# エンドポイント登録
app.include_router(endpoints.router)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Return the main page."""
    index_path = Path("/workspace/app/static/index.html")
    return index_path.read_text(encoding="utf-8")


@app.get("/upload", response_class=HTMLResponse)
async def upload() -> str:
    """Return the upload page."""
    upload_path = Path("/workspace/app/static/upload.html")
    return upload_path.read_text(encoding="utf-8")
