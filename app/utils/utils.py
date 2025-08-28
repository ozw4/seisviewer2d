import os

import numpy as np
import segyio


def quantize_float32(arr: np.ndarray, bits: int = 8, fixed_scale: float | None = None):
	qmax = (1 << (bits - 1)) - 1  # 127
	# 環境変数で既定値を上書き可能。例: FIXED_INT8_SCALE=42.333
	default = float(os.getenv('FIXED_INT8_SCALE', '42.333333'))
	scale = float(fixed_scale) if fixed_scale is not None else default
	q = np.clip(np.round(arr * scale), -qmax, qmax).astype(np.int8)
	return scale, q


class SegySectionReader:
	def __init__(self, path, key1_byte=189, key2_byte=193):
		self.path = path
		self.key1_byte = key1_byte
		self.key2_byte = key2_byte
		self.section_cache = {}  # ← sectionごとのキャッシュ
		self._initialize_metadata()

	def _initialize_metadata(self):
		with segyio.open(self.path, 'r', ignore_geometry=True) as f:
			f.mmap()
			self.key1s = f.attributes(self.key1_byte)[:]
			self.key2s = f.attributes(self.key2_byte)[:]
		self.unique_key1 = np.unique(self.key1s)

	def get_key1_values(self):
		return self.unique_key1

	def get_section(self, key1_val):
		# キャッシュにあれば返す
		if key1_val in self.section_cache:
			return self.section_cache[key1_val]

		# なければSEGYから読み込む
		indices = np.where(self.key1s == key1_val)[0]
		print(len(indices), 'indices found for key1_val:', key1_val)
		if len(indices) == 0:
			raise ValueError(f'Key1 value {key1_val} not found')

		# key2でソート
		key2_vals = self.key2s[indices]
		sorted_indices = indices[np.argsort(key2_vals)]

		with segyio.open(self.path, 'r', ignore_geometry=True) as f:
			f.mmap()
			traces = np.array([f.trace[i] for i in sorted_indices], dtype='float32')
			# --- z-score 正規化（トレース毎）: 平均0・標準偏差1 ---
			mean = traces.mean(axis=1, keepdims=True)
			std = traces.std(axis=1, keepdims=True)
			std[std == 0] = 1.0  # 定常/ゼロトレース対策
			section = ((traces - mean) / std).tolist()

		# キャッシュに保存
		self.section_cache[key1_val] = section
		return section

	def preload_all_sections(self):
		for key1_val in self.unique_key1:
			self.get_section(key1_val)
