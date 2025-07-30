# endpoint.py
"""HTTP endpoints for uploading files and retrieving sections."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from utils.utils import SegySectionReader

router = APIRouter()

UPLOAD_DIR = Path(__file__).parent / 'uploads'
UPLOAD_DIR.mkdir(exist_ok=True)

_cached_readers: dict[str, SegySectionReader] = {}
_segys: dict[str, Path] = {}

UPLOAD_FILE = File(...)


def _get_reader(file_id: str, key1_byte: int, key2_byte: int) -> SegySectionReader:
	cache_key = f'{file_id}_{key1_byte}_{key2_byte}'
	if cache_key not in _cached_readers:
		if file_id not in _segys:
			raise HTTPException(status_code=404, detail='File ID not found')
		_cached_readers[cache_key] = SegySectionReader(
			_segys[file_id],
			key1_byte,
			key2_byte,
		)
	return _cached_readers[cache_key]


@router.post('/upload_segy')
async def upload_segy(file: UploadFile = UPLOAD_FILE) -> dict[str, str]:
	"""Store uploaded SEG-Y file and return its generated ID."""
	if not file.filename:
		raise HTTPException(
			status_code=400,
			detail='Uploaded file must have a filename',
		)

	ext = Path(file.filename).suffix.lower()
	file_id = str(uuid4())
	dest_path = UPLOAD_DIR / f'{file_id}{ext}'
	with dest_path.open('wb') as f:
		f.write(await file.read())

	_segys[file_id] = dest_path
	return {'file_id': file_id}


@router.get('/get_section')
def get_section(
	file_id: Annotated[str, Query()],
	key1_byte: Annotated[int, Query(189)],
	key2_byte: Annotated[int, Query(193)],
	key1_idx: Annotated[int, Query()],
) -> JSONResponse:
	"""Return a seismic section as JSON."""
	try:
		reader = _get_reader(file_id, key1_byte, key2_byte)
		if key1_idx not in reader.unique_key1:
			raise HTTPException(status_code=400, detail='Invalid key1_idx')  # noqa: TRY301
		section = reader.get_section(key1_idx)
		return JSONResponse(content={'section': section})
	except Exception as e:  # pragma: no cover - generic error response
		raise HTTPException(status_code=500, detail=str(e)) from e
