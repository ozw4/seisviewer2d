import json
import os
from pathlib import Path

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
                self.path = Path(path)
                self.key1_byte = key1_byte
                self.key2_byte = key2_byte
                self.section_cache: dict[int, list] = {}
                self._load_or_build_cache()

        def _load_or_build_cache(self) -> None:
                cache_dir = self.path.parent
                traces_path = cache_dir / 'traces.npy'
                keys_path = cache_dir / (
			f'keys_k1_{self.key1_byte}_k2_{self.key2_byte}.npz'
		)
                index_path = cache_dir / (
			f'indexmap_k1_{self.key1_byte}_k2_{self.key2_byte}.json'
		)

                if traces_path.exists() and keys_path.exists() and index_path.exists():
                        self.traces_mm = np.load(traces_path, mmap_mode='r')
                        with np.load(keys_path) as npz:
                                self.key1s = npz['key1s']
                                self.key2s = npz['key2s']
                                self.unique_key1 = npz['unique_key1']
                                self.unique_key2 = npz['unique_key2']
                        with open(index_path, encoding='utf-8') as f:
                                self.indexmap = json.load(f)
                        return

                with segyio.open(self.path, 'r', ignore_geometry=True) as f:
                        f.mmap()
                        n_traces = len(f.trace)
                        n_samples = f.trace[0].size
                        arr = np.empty((n_traces, n_samples), dtype=np.float32)
                        for i in range(n_traces):
                                arr[i] = np.asarray(f.trace[i], dtype=np.float32)
                        np.save(traces_path, arr)
                        del arr
                        key1s = f.attributes(self.key1_byte)[:]
                        key2s = f.attributes(self.key2_byte)[:]

                unique_key1 = np.unique(key1s)
                unique_key2 = np.unique(key2s)
                by_key1 = {
                        str(int(v)): np.where(key1s == v)[0].tolist() for v in unique_key1
                }
                with open(index_path, 'w', encoding='utf-8') as fw:
                        json.dump({'by_key1': by_key1}, fw)
                np.savez_compressed(
                        keys_path,
                        key1s=key1s,
                        key2s=key2s,
                        unique_key1=unique_key1,
                        unique_key2=unique_key2,
                )

                self.traces_mm = np.load(traces_path, mmap_mode='r')
                self.key1s = key1s
                self.key2s = key2s
                self.unique_key1 = unique_key1
                self.unique_key2 = unique_key2
                self.indexmap = {'by_key1': by_key1}

        def get_key1_values(self):
                return self.unique_key1

        def get_section(self, key1_val):
                if key1_val in self.section_cache:
                        return self.section_cache[key1_val]

                idxs = self.indexmap['by_key1'].get(str(int(key1_val)))
                if not idxs:
                        raise ValueError(f'Key1 value {key1_val} not found')
                traces = np.take(self.traces_mm, idxs, axis=0)
                k2vals = np.take(self.key2s, idxs)
                order = np.argsort(k2vals)
                traces = traces[order]
                mean = traces.mean(axis=1, keepdims=True)
                std = traces.std(axis=1, keepdims=True)
                std[std == 0] = 1.0
                section = ((traces - mean) / std).tolist()
                self.section_cache[key1_val] = section
                return section

        def preload_all_sections(self):
                for key1_val in self.unique_key1:
                        self.get_section(int(key1_val))
