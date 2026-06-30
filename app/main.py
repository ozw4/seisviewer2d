"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routers.batch_apply import router as batch_apply_router
from app.api.routers.fbpick import router as fbpick_router
from app.api.routers.fbpick_predict import router as fbpick_predict_router
from app.api.routers.picks import router as picks_router
from app.api.routers.pipeline import router as pipeline_router
from app.api.routers.section import router as section_router
from app.api.routers.statics import router as statics_router
from app.api.routers.upload import router as upload_router
from app.core.paths import get_upload_dir
from app.core.state import create_app_state
from app.services.errors import DomainError
from app.services.segy_upload_storage import cleanup_staged_uploads

STATIC_DIR = (Path(__file__).parent / 'static').resolve()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cleanup_staged_uploads(app.state.sv, upload_dir=get_upload_dir(), force=True)
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


@app.get('/static-correction', response_class=HTMLResponse)
async def static_correction() -> str:
    """Return the dedicated static correction page."""
    static_correction_path = STATIC_DIR / 'static_correction.html'
    return static_correction_path.read_text(encoding='utf-8')


@app.get('/refraction-qc', response_class=HTMLResponse)
async def refraction_qc() -> str:
    """Return the dedicated refraction QC page."""
    refraction_qc_path = STATIC_DIR / 'refraction_qc.html'
    return refraction_qc_path.read_text(encoding='utf-8')


@app.get('/batch', response_class=HTMLResponse)
async def batch() -> str:
    """Return the batch apply page."""
    batch_path = STATIC_DIR / 'batch_apply.html'
    return batch_path.read_text(encoding='utf-8')
