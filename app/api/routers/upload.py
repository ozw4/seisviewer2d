"""Upload and registration endpoints."""

from __future__ import annotations

import asyncio
import json
import pathlib
import re
import threading
from pathlib import Path
from uuid import uuid4

import numpy as np
import segyio
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from app.api._helpers import SEGYS, _update_file_registry, cached_readers
from app.utils.segy_meta import FILE_REGISTRY, read_segy_dt_seconds
from app.utils.utils import TraceStoreSectionReader

router = APIRouter()

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / 'uploads'
UPLOAD_DIR.mkdir(exist_ok=True)

PROCESSED_DIR = UPLOAD_DIR / 'processed'

TRACE_DIR = PROCESSED_DIR / 'traces'
TRACE_DIR.mkdir(parents=True, exist_ok=True)


@router.post('/open_segy')
async def open_segy(
        original_name: str = Form(...),
        key1_byte: int = Form(189),
        key2_byte: int = Form(193),
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
        file: UploadFile = File(...),
        key1_byte: int = Form(189),
        key2_byte: int = Form(193),
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

        if meta_path.exists():
                print(f'Reusing trace store for {file.filename}')
                reader = TraceStoreSectionReader(store_dir, key1_byte, key2_byte)
                SEGYS[file_id] = str(store_dir)
                cache_key = f'{file_id}_{key1_byte}_{key2_byte}'
                cached_readers[cache_key] = reader
                threading.Thread(target=reader.preload_all_sections, daemon=True).start()
                for b in {key1_byte, key2_byte}:
                        threading.Thread(
                                target=reader.ensure_header, args=(b,), daemon=True
                        ).start()
                segy_path = (
                        reader.meta.get('original_segy_path')
                        if isinstance(reader.meta, dict)
                        else None
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

        raw_path = UPLOAD_DIR / safe_name
        data = await file.read()
        await asyncio.to_thread(raw_path.write_bytes, data)
        store_dir.mkdir(parents=True, exist_ok=True)
        traces_tmp = store_dir / 'traces.npy.tmp'
        dt_seconds = read_segy_dt_seconds(str(raw_path)) or 0.002

        with segyio.open(raw_path, 'r', ignore_geometry=True) as segy:
                segy.mmap()
                n_traces = segy.tracecount
                n_samples = len(segy.trace[0])
        mm = np.lib.format.open_memmap(
                traces_tmp,
                mode='w+',
                dtype=np.float32,
                shape=(n_traces, n_samples),
        )
        for i in range(n_traces):
                tr = segy.trace[i].astype(np.float32)
                mean = tr.mean()
                std = tr.std()
                if std == 0:
                        std = 1.0
                mm[i] = (tr - mean) / std
        del mm
        traces_tmp.replace(store_dir / 'traces.npy')
        meta = {
                'n_traces': int(n_traces),
                'n_samples': int(n_samples),
                'original_segy_path': str(raw_path),
                'version': 1,
                'normalized': True,
                'dt': dt_seconds,
        }
        tmp_meta = store_dir / 'meta.json.tmp'
        tmp_meta.write_text(json.dumps(meta))
        tmp_meta.replace(meta_path)

        reader = TraceStoreSectionReader(store_dir, key1_byte, key2_byte)
        SEGYS[file_id] = str(store_dir)
        cache_key = f'{file_id}_{key1_byte}_{key2_byte}'
        cached_readers[cache_key] = reader
        threading.Thread(target=reader.preload_all_sections, daemon=True).start()
        for b in {key1_byte, key2_byte}:
                threading.Thread(target=reader.ensure_header, args=(b,), daemon=True).start()
        _update_file_registry(
                file_id,
                path=str(raw_path),
                store_path=str(store_dir),
                dt=dt_seconds,
        )
        return {'file_id': file_id, 'reused_trace_store': False}


@router.get('/file_info')
async def file_info(file_id: str = Query(...)) -> dict[str, str]:
        """Return basename for a given ``file_id``."""
        rec = FILE_REGISTRY.get(file_id) or {}
        path = rec.get('path') or rec.get('store_path')
        if not path:
                raise HTTPException(status_code=404, detail='Unknown file_id')
        return {'file_name': Path(path).name}

