from pathlib import Path

from api import endpoints
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# 静的ファイル（HTML, JS）
app.mount(
	'/workspace/app/static',
	StaticFiles(directory='/workspace/app/static'),
	name='static',
)

# エンドポイント登録
app.include_router(endpoints.router)


@app.get('/', response_class=HTMLResponse)
async def index():
	index_path = Path('/workspace/app/static/index.html')
	return index_path.read_text(encoding='utf-8')
