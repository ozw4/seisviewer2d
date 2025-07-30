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

	def get_section(self, key1_val, sample_start=None, sample_end=None):
                # キャッシュにあれば利用
                if key1_val in self.section_cache:
                        section = self.section_cache[key1_val]
                else:
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
                                section = [
                                        f.trace[i].tolist() for i in sorted_indices
                                ]  # list of lists に変換

                        # キャッシュに保存
                        self.section_cache[key1_val] = section

                # サンプル範囲の適用
                if sample_start is not None or sample_end is not None:
                        section = [tr[sample_start:sample_end] for tr in section]
                return section
