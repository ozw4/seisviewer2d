"""Upload and registration endpoints."""

from __future__ import annotations

import asyncio
import pathlib
import re
import threading
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from app.api._helpers import SEGYS, _update_file_registry, cached_readers
from app.utils.ingest import SegyIngestor
from app.utils.segy_meta import FILE_REGISTRY, read_segy_dt_seconds
from app.utils.utils import TraceStoreSectionReader

router = APIRouter()

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / 'uploads'
UPLOAD_DIR.mkdir(exist_ok=True)

PROCESSED_DIR = UPLOAD_DIR / 'processed'

TRACE_DIR = PROCESSED_DIR / 'traces'
TRACE_DIR.mkdir(parents=True, exist_ok=True)


def _is_trace_store_complete(store_dir: Path, key1_byte: int, key2_byte: int) -> bool:
	"""Return ``True`` when ``store_dir`` has the required TraceStore artifacts."""
	required = [
		store_dir / 'traces.npy',
		store_dir / 'meta.json',
		store_dir / 'index.npz',
		store_dir / f'headers_byte_{key1_byte}.npy',
		store_dir / f'headers_byte_{key2_byte}.npy',
	]
	return all(path.exists() for path in required)


@router.post('/open_segy')
async def open_segy(
	original_name: Annotated[str, Form(...)],
	key1_byte: Annotated[int, Form()] = 189,
	key2_byte: Annotated[int, Form()] = 193,
):
	safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', original_name)
	store_dir = TRACE_DIR / safe_name
	if not _is_trace_store_complete(store_dir, key1_byte, key2_byte):
		raise HTTPException(
			status_code=404,
			detail=f'Trace store not found for {original_name}',
		)
	print(f'Opening existing trace store for {original_name}')
	file_id = str(uuid4())
	reader = TraceStoreSectionReader(store_dir, key1_byte, key2_byte)
	SEGYS[file_id] = str(store_dir)
	cache_key = f'{file_id}_{key1_byte}_{key2_byte}'
	cached_readers[cache_key] = reader
	threading.Thread(target=reader.preload_all_sections, daemon=True).start()
	for b in {key1_byte, key2_byte}:
		threading.Thread(target=reader.ensure_header, args=(b,), daemon=True).start()
	segy_path = (
		reader.meta.get('original_segy_path') if isinstance(reader.meta, dict) else None
	)
	dt_meta = None
	if isinstance(reader.meta, dict):
		dt_meta = reader.meta.get('dt')
	if (
			dt_meta is None or not isinstance(dt_meta, (int, float)) or dt_meta <= 0
	) and isinstance(segy_path, str):
		dt_meta = read_segy_dt_seconds(segy_path)
	_update_file_registry(
		file_id,
		path=segy_path if isinstance(segy_path, str) else None,
		store_path=str(store_dir),
		dt=dt_meta,
	)
	return {'file_id': file_id, 'reused_trace_store': True}

@router.post('/upload_segy')
async def upload_segy(
	file: Annotated[UploadFile, File(...)],
	key1_byte: Annotated[int, Form()] = 189,
	key2_byte: Annotated[int, Form()] = 193,
):
	if not file.filename:
		raise HTTPException(
			status_code=400, detail='Uploaded file must have a filename'
		)
	print(f'Uploading file: {file.filename}')
	safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', file.filename)
	store_dir = TRACE_DIR / safe_name
	file_id = str(uuid4())
	store_dir.mkdir(parents=True, exist_ok=True)
	reused_trace_store = _is_trace_store_complete(store_dir, key1_byte, key2_byte)
	if not reused_trace_store:
		raw_path = UPLOAD_DIR / safe_name
		data = await file.read()
		await asyncio.to_thread(raw_path.write_bytes, data)
		try:
			await asyncio.to_thread(
				SegyIngestor.from_segy,
				str(raw_path),
				str(store_dir),
				key1_byte,
				key2_byte,
			)
		except RuntimeError as exc:
			raise HTTPException(status_code=400, detail=str(exc)) from exc
	else:
		print(f'Reusing trace store for {file.filename}')
	reader = TraceStoreSectionReader(store_dir, key1_byte, key2_byte)
	SEGYS[file_id] = str(store_dir)
	cache_key = f'{file_id}_{key1_byte}_{key2_byte}'
	cached_readers[cache_key] = reader
	threading.Thread(target=reader.preload_all_sections, daemon=True).start()
	for b in {key1_byte, key2_byte}:
		threading.Thread(target=reader.ensure_header, args=(b,), daemon=True).start()
	segy_path = (
		reader.meta.get('original_segy_path') if isinstance(reader.meta, dict) else None
	)
	dt_meta = None
	if isinstance(reader.meta, dict):
		dt_meta = reader.meta.get('dt')
	if (
			dt_meta is None or not isinstance(dt_meta, (int, float)) or dt_meta <= 0
	) and isinstance(segy_path, str):
		dt_meta = read_segy_dt_seconds(segy_path)
	_update_file_registry(
		file_id,
		path=segy_path if isinstance(segy_path, str) else None,
		store_path=str(store_dir),
		dt=dt_meta,
	)
	return {'file_id': file_id, 'reused_trace_store': reused_trace_store}
@router.get('/file_info')
async def file_info(file_id: Annotated[str, Query()]) -> dict[str, str]:
	"""Return basename for a given ``file_id``."""
	rec = FILE_REGISTRY.get(file_id) or {}
	path = rec.get('path') or rec.get('store_path')
	if not path:
		raise HTTPException(status_code=404, detail='Unknown file_id')
	return {'file_name': Path(path).name}
