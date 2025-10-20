"""Upload and registration endpoints."""

from __future__ import annotations

import asyncio
import json
import pathlib
import re
import threading
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from app.api._helpers import SEGYS, cached_readers
from app.utils.ingest import SegyIngestor
from app.utils.segy_meta import FILE_REGISTRY
from app.utils.utils import TraceStoreSectionReader

router = APIRouter()

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / 'uploads'
UPLOAD_DIR.mkdir(exist_ok=True)

PROCESSED_DIR = UPLOAD_DIR / 'processed'

TRACE_DIR = PROCESSED_DIR / 'traces'
TRACE_DIR.mkdir(parents=True, exist_ok=True)


def _trace_store_complete(store_dir: Path, key1_byte: int, key2_byte: int) -> bool:
	required = [
		store_dir / 'traces.npy',
		store_dir / 'index.npz',
		store_dir / 'meta.json',
		store_dir / f'headers_byte_{key1_byte}.npy',
		store_dir / f'headers_byte_{key2_byte}.npy',
	]
	return all(path.exists() for path in required)


def _register_trace_store(
	file_id: str, store_dir: Path, key1_byte: int, key2_byte: int
) -> TraceStoreSectionReader:
	reader = TraceStoreSectionReader(store_dir, key1_byte, key2_byte)
	SEGYS[file_id] = str(store_dir)
	cache_key = f'{file_id}_{key1_byte}_{key2_byte}'
	cached_readers[cache_key] = reader
	threading.Thread(target=reader.preload_all_sections, daemon=True).start()
	for b in {key1_byte, key2_byte}:
		threading.Thread(target=reader.ensure_header, args=(b,), daemon=True).start()
	return reader


@router.post('/open_segy')
async def open_segy(
	original_name: Annotated[str, Form(...)],
	key1_byte: Annotated[int, Form()] = 189,
	key2_byte: Annotated[int, Form()] = 193,
):
	safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', original_name)
	store_dir = TRACE_DIR / safe_name
	meta_path = store_dir / 'meta.json'
	if not meta_path.exists():
		raise HTTPException(
			status_code=404,
			detail=f'Trace store not found for {original_name}',
		)
	print(f'Opening existing trace store for {original_name}')
	file_id = str(uuid4())
	reused = _trace_store_complete(store_dir, key1_byte, key2_byte)
	if reused:
		meta = json.loads(meta_path.read_text())
	else:
		meta = json.loads(meta_path.read_text())
		segy_path = meta.get('original_segy_path') if isinstance(meta, dict) else None
		if not isinstance(segy_path, str):
			raise HTTPException(
				status_code=500,
				detail='Trace store incomplete and SEG-Y path unavailable',
			)
		meta = await asyncio.to_thread(
			SegyIngestor.from_segy,
			segy_path,
			store_dir,
			key1_byte,
			key2_byte,
		)
	_register_trace_store(file_id, store_dir, key1_byte, key2_byte)
	if isinstance(meta, dict):
		FILE_REGISTRY[file_id] = {
			'store_path': str(store_dir),
			'dt': meta.get('dt'),
		}
	return {'file_id': file_id, 'reused_trace_store': reused}


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
	meta_path = store_dir / 'meta.json'
	file_id = str(uuid4())

	if meta_path.exists() and _trace_store_complete(store_dir, key1_byte, key2_byte):
		print(f'Reusing trace store for {file.filename}')
		reader = _register_trace_store(file_id, store_dir, key1_byte, key2_byte)
		meta = reader.meta if isinstance(reader.meta, dict) else {}
		if isinstance(meta, dict):
			FILE_REGISTRY[file_id] = {
				'store_path': str(store_dir),
				'dt': meta.get('dt'),
			}
		return {'file_id': file_id, 'reused_trace_store': True}

	raw_path = UPLOAD_DIR / safe_name
	data = await file.read()
	await asyncio.to_thread(raw_path.write_bytes, data)
	store_dir.mkdir(parents=True, exist_ok=True)
	meta = await asyncio.to_thread(
		SegyIngestor.from_segy,
		str(raw_path),
		store_dir,
		key1_byte,
		key2_byte,
	)
	reader = _register_trace_store(file_id, store_dir, key1_byte, key2_byte)
	if isinstance(meta, dict):
		FILE_REGISTRY[file_id] = {
			'store_path': str(store_dir),
			'dt': meta.get('dt'),
		}
	return {'file_id': file_id, 'reused_trace_store': False}


@router.get('/file_info')
async def file_info(file_id: Annotated[str, Query()]) -> dict[str, str]:
	"""Return basename for a given ``file_id``."""
	rec = FILE_REGISTRY.get(file_id) or {}
	path = rec.get('path') or rec.get('store_path')
	if not path:
		raise HTTPException(status_code=404, detail='Unknown file_id')
	return {'file_name': Path(path).name}
