"""FastAPI application entry point."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api.routers import (
	fbpick_router,
	picks_router,
	pipeline_router,
	section_router,
	upload_router,
)

STATIC_DIR = (Path(__file__).parent / 'static').resolve()


app = FastAPI()

# 静的ファイル (HTML, JS)
app.mount(
	'/static',
	StaticFiles(directory=str(STATIC_DIR)),
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
	index_path = STATIC_DIR / 'index.html'
	return index_path.read_text(encoding='utf-8')


@app.get('/upload', response_class=HTMLResponse)
async def upload() -> str:
	"""Return the upload page."""
	upload_path = STATIC_DIR / 'upload.html'
	return upload_path.read_text(encoding='utf-8')
