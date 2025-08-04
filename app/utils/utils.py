import numpy as np
import segyio


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
			max_abs = np.max(np.abs(traces), axis=1, keepdims=True)
			max_abs[max_abs == 0] = 1.0
			section = (traces / max_abs).tolist()

		# キャッシュに保存
		self.section_cache[key1_val] = section
		return section

	def preload_all_sections(self):
		for key1_val in self.unique_key1:
			self.get_section(key1_val)
