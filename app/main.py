"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routers import (
    batch_apply_router,
    fbpick_predict_router,
    fbpick_router,
    picks_router,
    pipeline_router,
    section_router,
    statics_router,
    upload_router,
)
from app.api.routers.upload import cleanup_staged_uploads
from app.core.state import create_app_state
from app.services.errors import DomainError

STATIC_DIR = (Path(__file__).parent / 'static').resolve()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cleanup_staged_uploads(app.state.sv, force=True)
    yield


app = FastAPI(lifespan=lifespan)
app.state.sv = create_app_state()

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
app.include_router(fbpick_predict_router)
app.include_router(pipeline_router)
app.include_router(batch_apply_router)
app.include_router(picks_router)
app.include_router(statics_router)


@app.exception_handler(DomainError)
async def handle_domain_error(_: Request, exc: DomainError) -> JSONResponse:
    """Convert service-layer domain errors into HTTP responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content={'detail': exc.detail},
    )


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


@app.get('/batch', response_class=HTMLResponse)
async def batch() -> str:
    """Return the batch apply page."""
    batch_path = STATIC_DIR / 'batch_apply.html'
    return batch_path.read_text(encoding='utf-8')
