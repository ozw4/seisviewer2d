# endpoint.py
import pathlib
from threading import Thread
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from utils.utils import SegySectionReader

router = APIRouter()

UPLOAD_DIR = pathlib.Path(__file__).parent / 'uploads'
UPLOAD_DIR.mkdir(exist_ok=True)

cached_readers: dict[str, SegySectionReader] = {}
SEGYS: dict[str, str] = {}

@router.get('/get_key1_values')
def get_key1_values(
	file_id: str = Query(...),
	key1_byte: int = Query(189),
	key2_byte: int = Query(193),
):
	cache_key = f'{file_id}_{key1_byte}_{key2_byte}'
	if cache_key not in cached_readers:
		if file_id not in SEGYS:
			raise HTTPException(status_code=404, detail='File ID not found')
		path = SEGYS[file_id]
		cached_readers[cache_key] = SegySectionReader(path, key1_byte, key2_byte)
	reader = cached_readers[cache_key]
	return JSONResponse(content={'values': reader.get_key1_values().tolist()})


@router.post('/upload_segy')
async def upload_segy(
        file: UploadFile = File(...),
        key1_byte: int = Query(189),
        key2_byte: int = Query(193),
):
        if not file.filename:
                raise HTTPException(
                        status_code=400, detail='Uploaded file must have a filename'
                )
        print(f'Uploading file: {file.filename}')
        ext = pathlib.Path(file.filename).suffix.lower()
        file_id = str(uuid4())
        dest_path = UPLOAD_DIR / f'{file_id}{ext}'
        with open(dest_path, 'wb') as f:
                f.write(await file.read())

        SEGYS[file_id] = str(dest_path)

        cache_key = f'{file_id}_{key1_byte}_{key2_byte}'

        def preload_reader():
                cached_readers[cache_key] = SegySectionReader(
                        str(dest_path), key1_byte, key2_byte
                )

        Thread(target=preload_reader, daemon=True).start()
        return {'file_id': file_id}


@router.get('/get_section')
def get_section(
	file_id: str = Query(...),
	key1_byte: int = Query(189),  # デフォルト設定
	key2_byte: int = Query(193),
	key1_idx: int = Query(...),
):
	try:
		# 複合キーを作る（file_id, key1_byte, key2_byte）
		cache_key = f'{file_id}_{key1_byte}_{key2_byte}'

		# キャッシュにreaderがなければ初期化
		if cache_key not in cached_readers:
			if file_id not in SEGYS:
				raise HTTPException(status_code=404, detail='File ID not found')
			path = SEGYS[file_id]
			cached_readers[cache_key] = SegySectionReader(path, key1_byte, key2_byte)

		reader = cached_readers[cache_key]
		section = reader.get_section(key1_idx)
		return JSONResponse(content={'section': section})

	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))
