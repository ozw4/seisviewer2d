"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from utils import picks_by_name
from utils.picks import store

from app.api.routers import (
	fbpick_router,
	picks_router,
	pipeline_router,
	section_router,
	upload_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
	# --- startup ---
	store.load()
	picks_by_name.load()
	yield
	# --- shutdown ---
	store.save()
	picks_by_name.save()


app = FastAPI(lifespan=lifespan)

# 静的ファイル (HTML, JS)
app.mount(
	'/static',  # ← URLパス
	StaticFiles(directory='/workspace/app/static'),  # ← ローカルパス
	name='static',
)

# エンドポイント登録
app.include_router(upload_router)
app.include_router(section_router)
app.include_router(fbpick_router)
app.include_router(pipeline_router)
app.include_router(picks_router)


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
