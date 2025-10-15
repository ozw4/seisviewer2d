# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""./data/ 配下の SEG-Y (各ファイル=60秒=1イベント) を 1 本に結合。
- 各トレースに JST の開始時刻（カタログの DATE+TIME）を書き込む
- そのほかの情報を CSV にまとめて出力
依存: segyio, numpy, pandas
"""

from pathlib import Path

import numpy as np
import pandas as pd
import segyio
from preproc import valid_preproc
from segyio import BinField as BF
from segyio import TraceField as TF

# ===== 設定 =====
DATA_DIR = Path('/workspace/data/DAS_extract_sgy_resamp100Hz/')

FFID_START = 1
CATALOG = Path('/workspace/data/EQ_widerange_with_tauP_dist.xlsx')
SCORE_COL = 'borehole_score'
TIME_COL = (
	'origin_time_local'  # 入力SEG-Yのファイル決定には従来通りこの列のスタブを使う
)
SCORE = 1
OUTPUT = Path(
	f'/workspace/data/DAS_extract_sgy_resamp100Hz/merge/merged_events_score{SCORE}.sgy'
)
MANIFEST_CSV = OUTPUT.with_suffix('.csv')  # まとめCSV

# カタログ列名（存在すれば CSV に含める）
OPTIONAL_COLS = [
	'EQseq',
	'LON',
	'LAT',
	'Dep',
	'Mag',
	'dist_deg',
	'tP_s',
	'tS_s',
	'S_minus_P_s',
	'Distance_km',
]

# 抽出するトレース範囲（0始まり・終端排他的）
START_TRACE = 3999
END_TRACE = 4736  # 4000..4749 → 750 traces
# 固定ターゲット長
FORCE_NS: int | None = 2999


def build_event_table(df: pd.DataFrame) -> pd.DataFrame:
	"""カタログからイベント表を作る。
	- フィルタ: SCORE_COL == SCORE
	- origin_time_local → 入力ファイル名スタブ（YYYYmmdd_HHMMSS）
	- DATE + TIME → JST のイベント時刻（ヘッダに書く値）
	"""
	score = pd.to_numeric(df[SCORE_COL], errors='coerce')
	df1 = df.loc[score == SCORE].copy()
	if df1.empty:
		raise RuntimeError(f'No rows with {SCORE_COL}=={SCORE}')

	# 入力ファイル決定用スタブ（従来通り）
	t_origin = pd.to_datetime(df1[TIME_COL], errors='raise')
	df1['stub'] = t_origin.dt.strftime('%Y%m%d_%H%M%S')

	# ヘッダに書く JST 開始時刻（DATE + TIME）
	# DATE が日付/日時、TIME が "HH:MM:SS.sss" 形式想定
	date_str = pd.to_datetime(df1['DATE'], errors='raise').dt.strftime('%Y-%m-%d')
	time_str = df1['TIME'].astype(str)
	jst_dt = pd.to_datetime(date_str + ' ' + time_str, errors='raise')  # naive JST
	df1['jst_dt'] = jst_dt

	return df1


def pick_files_from_events(events: pd.DataFrame, data_dir: Path) -> list[Path]:
	"""イベント表から入力ファイルパス列挙。"""
	stubs = events['stub'].tolist()
	files: list[Path] = []
	missing: list[str] = []
	for s in stubs:
		p1 = data_dir / f'{s}.sgy'
		p2 = data_dir / f'{s}.segy'
		if p1.exists():
			files.append(p1)
		elif p2.exists():
			files.append(p2)
		else:
			missing.append(f'{s}.sgy|.segy')
	if missing:
		print(
			f'Missing {len(missing)} files under {data_dir} (first few): {", ".join(missing[:10])}'
		)

	files = sorted(set(files), key=lambda p: p.stem)
	if not files:
		raise RuntimeError('No target files after filtering.')
	return files


def scan_inputs(paths: list[Path], force_ns: int | None = None):
	"""dt, fmt は一致必須。ns は force_ns があれば '>= force_ns' を要求。"""
	dt = fmt = None
	ns_min = None
	total_traces = 0

	for p in paths:
		with segyio.open(p.as_posix(), 'r', ignore_geometry=True) as f:
			ns_i = int(f.bin[BF.Samples])
			dt_i = int(f.bin[BF.Interval])
			fmt_i = int(f.bin[BF.Format])

			if dt is None:
				dt, fmt = dt_i, fmt_i
			elif (dt_i != dt) or (fmt_i != fmt):
				raise RuntimeError(
					f'Format mismatch: {p.name} (dt={dt_i} vs {dt}, fmt={fmt_i} vs {fmt})'
				)
			ns_min = ns_i if ns_min is None else min(ns_min, ns_i)
			total_traces += END_TRACE - START_TRACE

	if dt is None:
		raise RuntimeError('No inputs')

	ns_out = force_ns if force_ns is not None else ns_min
	if ns_out is None:
		raise RuntimeError('Failed to decide output ns')

	if force_ns is not None:
		for p in paths:
			with segyio.open(p.as_posix(), 'r', ignore_geometry=True) as f:
				ns_i = int(f.bin[BF.Samples])
				if ns_i < ns_out:
					raise RuntimeError(
						f'{p.name}: ns={ns_i} < target ns={ns_out} (cannot pad; abort)'
					)

	return {'ns': ns_out, 'dt': dt, 'fmt': fmt}, total_traces


def merge_inputs_to_single(
	paths: list[Path], events: pd.DataFrame, out_path: Path, ffid_start: int = 1
):
	"""イベント結合 & ヘッダに JST の開始時刻を記録。併せて CSV を出力。"""
	meta, total_traces = scan_inputs(paths)
	if out_path.exists():
		out_path.unlink()

	ns_out = int(meta['ns'])
	n_ch = END_TRACE - START_TRACE

	# TraceField キーの互換フォールバック
	TF_TRACE_SEQ_LINE = getattr(TF, 'TraceSequenceLine', 1)
	TF_TRACE_SEQ_FILE = getattr(TF, 'TraceSequenceFile', 5)
	TF_FIELD_RECORD = getattr(TF, 'FieldRecord', 9)
	TF_TRACE_NUMBER = getattr(TF, 'ChannelNumber', getattr(TF, 'TraceNumber', 13))
	TF_CDP = getattr(TF, 'CDP', 21)
	TF_TRACE_NS = getattr(TF, 'TraceSampleCount', None)
	TF_YEAR = getattr(TF, 'YearDataRecorded', None)
	TF_DOY = getattr(TF, 'DayOfYear', None)
	TF_HOUR = getattr(TF, 'HourOfDay', None)
	TF_MINUTE = getattr(TF, 'MinuteOfHour', None)
	TF_SECOND = getattr(TF, 'SecondOfMinute', None)
	TF_TIMEBASE = getattr(TF, 'TimeBaseCode', None)

	# イベント表を高速参照できるように
	ev_by_stub = events.set_index('stub')

	# 出力メタCSV用
	rows = []

	# 出力ファイル仕様
	spec = segyio.spec()
	spec.format = meta['fmt']
	spec.samples = meta.get('samples', range(ns_out))
	if hasattr(spec, 'tracecount'):
		spec.tracecount = total_traces
	else:
		spec.ilines = [1]
		spec.xlines = list(range(total_traces))

	with segyio.create(str(out_path), spec) as dst:
		dst.bin[BF.Samples] = ns_out
		dst.bin[BF.Interval] = meta['dt']
		dst.bin[BF.Format] = meta['fmt']

		out_idx = 0
		ffid = ffid_start

		# paths は origin_time_local のスタブでユニーク化されている前提
		for p in paths:
			stub = p.stem
			if stub not in ev_by_stub.index:
				raise RuntimeError(f'Catalog row not found for stub={stub}')
			ev = ev_by_stub.loc[stub]

			# JST 開始時刻（DATE+TIME）
			jst_dt: pd.Timestamp = pd.to_datetime(ev['jst_dt'])
			py_dt = jst_dt.to_pydatetime()
			y = py_dt.year
			doy = py_dt.timetuple().tm_yday  # 1..366
			hh, mm, ss = py_dt.hour, py_dt.minute, py_dt.second

			# ギャザー読み込み → 前処理
			with segyio.open(str(p), 'r+', ignore_geometry=True) as src:
				gather = np.empty((n_ch, ns_out), dtype=np.float32)
				for j, k in enumerate(range(START_TRACE, END_TRACE)):
					tr = src.trace[k]
					if len(tr) < ns_out:
						raise RuntimeError(
							f'{p.name}: trace {k} ns={len(tr)} < target {ns_out}'
						)
					gather[j, :] = tr[:ns_out]

				proc = valid_preproc(gather, resample=False)
				if proc.shape != gather.shape:
					raise RuntimeError(
						f'valid_preproc returned {proc.shape}, expected {gather.shape}'
					)

				for j, k in enumerate(range(START_TRACE, END_TRACE)):
					dst.trace[out_idx] = proc[j]
					hdr = src.header[k]

					hdr[TF_TRACE_SEQ_LINE] = j + 1
					hdr[TF_TRACE_SEQ_FILE] = out_idx + 1
					hdr[TF_TRACE_NUMBER] = j + 1
					hdr[TF_FIELD_RECORD] = ffid
					hdr[TF_CDP] = ffid
					if TF_TRACE_NS is not None:
						hdr[TF_TRACE_NS] = ns_out

					# --- JST の開始時刻を書き込む（このビルドのフィールド名に合わせる）---
					missing = [
						n
						for n, v in [
							('YearDataRecorded', TF_YEAR),
							('DayOfYear', TF_DOY),
							('HourOfDay', TF_HOUR),
							('MinuteOfHour', TF_MINUTE),
							('SecondOfMinute', TF_SECOND),
						]
						if v is None
					]
					if missing:
						raise RuntimeError(f'segyio TraceField missing: {missing}')

					hdr[TF_YEAR] = y
					hdr[TF_DOY] = doy
					hdr[TF_HOUR] = hh
					hdr[TF_MINUTE] = mm
					hdr[TF_SECOND] = ss
					if TF_TIMEBASE is not None:
						hdr[TF_TIMEBASE] = 1  # Local time (= JST)

					dst.header[out_idx] = hdr
					out_idx += 1

			# メタCSV行を追加（イベント単位）
			row = {
				'ffid': ffid,
				'stub': stub,
				'input_file': p.name,
				'event_time_jst': jst_dt.isoformat(),
				'n_traces': n_ch,
				'ns': ns_out,
				'dt_us': int(meta['dt']),
			}
			for c in OPTIONAL_COLS:
				if c in ev.index:
					row[c] = ev[c]
			rows.append(row)

			ffid += 1

	# メタCSV出力
	man = pd.DataFrame(rows)
	man = man[
		['ffid', 'stub', 'input_file', 'event_time_jst', 'n_traces', 'ns', 'dt_us']
		+ [c for c in OPTIONAL_COLS if c in man.columns]
	]
	man.sort_values('ffid').to_csv(MANIFEST_CSV, index=False)

	print(f'Done: {out_path}  traces={total_traces}  ns={ns_out}  dt(us)={meta["dt"]}')
	print(f'Wrote manifest: {MANIFEST_CSV}')


if __name__ == '__main__':
	df = pd.read_excel(CATALOG)
	events = build_event_table(df)  # カタログ → イベント表（stub, jst_dt を含む）
	inputs = pick_files_from_events(events, DATA_DIR)
	merge_inputs_to_single(inputs, events, OUTPUT, ffid_start=FFID_START)
