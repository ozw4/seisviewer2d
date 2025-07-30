"""FastAPI application entrypoint."""

from pathlib import Path

from api import endpoints
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

STATIC_DIR = Path(__file__).parent / 'static'

app.mount('/workspace/app/static', StaticFiles(directory=STATIC_DIR), name='static')

# エンドポイント登録
app.include_router(endpoints.router)


@app.get('/', response_class=HTMLResponse)
async def index() -> str:
	"""Serve the main HTML page."""
	index_path = STATIC_DIR / 'index.html'
	return index_path.read_text(encoding='utf-8')
