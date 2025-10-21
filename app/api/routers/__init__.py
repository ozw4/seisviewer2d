"""Router exports for FastAPI endpoints."""

from app.api.routers.fbpick import router as fbpick_router
from app.api.routers.fbpick_predict import router as fbpick_predict_router
from app.api.routers.picks import router as picks_router
from app.api.routers.pipeline import router as pipeline_router
from app.api.routers.section import router as section_router
from app.api.routers.upload import router as upload_router

__all__ = [
	'fbpick_router',
	'fbpick_predict_router',
	'picks_router',
	'pipeline_router',
	'section_router',
	'upload_router',
]
