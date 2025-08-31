import hashlib
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
        """Read SEGY sections with an on‑disk cache.

        The cache avoids repeated disk IO for large files by building a
        ``numpy.memmap`` of all traces and storing header information.
        ``get_section`` simply gathers the required traces from the memmap,
        preserving the previous public API.
        """

        def __init__(self, path, key1_byte: int = 189, key2_byte: int = 193):
                self.path = path
                self.key1_byte = key1_byte
                self.key2_byte = key2_byte
                # keep per‑key1 in‑memory cache of normalised sections
                self.section_cache: dict[int, list[list[float]]] = {}
                self._open_or_build_cache()

        # ------------------------------------------------------------------
        # Cache handling
        # ------------------------------------------------------------------
        def _file_cache_dir(self) -> Path:
                """Return directory for file-level cache (traces/meta)."""
                p = Path(os.path.abspath(self.path))
                digest = hashlib.sha1(p.as_posix().encode('utf-8')).hexdigest()[:16]
                safe = p.stem.replace('/', '_').replace('\\', '_')
                base = (
                        Path(__file__).resolve().parent.parent
                        / 'api'
                        / 'uploads'
                        / 'processed'
                        / 'cache'
                )
                return base / f"{safe}_{digest}"

        def _domain_cache_dir(self) -> Path:
                """Return directory for domain-specific cache (keys/indexmap)."""
                return self._file_cache_dir() / f"k1_{self.key1_byte}_k2_{self.key2_byte}"

        @staticmethod
        def _build_index_maps(key1s: np.ndarray, key2s: np.ndarray):
                """Return unique values and index maps for key1/key2 arrays."""
                def build(keys: np.ndarray):
                        order = np.argsort(keys, kind="mergesort")
                        sorted_keys = keys[order]
                        by = {}
                        unique_vals = []
                        start = 0
                        n = len(sorted_keys)
                        for i in range(1, n + 1):
                                if i == n or sorted_keys[i] != sorted_keys[start]:
                                        val = int(sorted_keys[start])
                                        unique_vals.append(val)
                                        by[str(val)] = order[start:i].tolist()
                                        start = i
                        return np.asarray(unique_vals, dtype=keys.dtype), by

                unique_key1, by_key1 = build(key1s)
                unique_key2, by_key2 = build(key2s)
                return unique_key1, unique_key2, by_key1, by_key2

        def _open_or_build_cache(self):
                file_dir = self._file_cache_dir()
                file_dir.mkdir(parents=True, exist_ok=True)
                dom_dir = self._domain_cache_dir()
                dom_dir.mkdir(parents=True, exist_ok=True)

                meta_path = file_dir / 'meta.json'
                traces_path = file_dir / 'traces.dat'
                keys_path = dom_dir / 'keys.npz'
                indexmap_path = dom_dir / 'indexmap.json'

                stat = os.stat(self.path)
                segy_size = stat.st_size
                segy_mtime = int(stat.st_mtime)

                meta = None
                if meta_path.exists():
                        with open(meta_path, encoding='utf-8') as f:
                                meta = json.load(f)

                file_ok = False
                if meta and traces_path.exists():
                        if meta.get('size') == segy_size and meta.get('mtime') == segy_mtime:
                                itemsize = np.dtype(meta['dtype']).itemsize
                                expected = itemsize * meta['shape'][0] * meta['shape'][1]
                                if traces_path.stat().st_size == expected:
                                        file_ok = True

                if not file_ok:
                        # build file-level cache (traces + meta + domain cache)
                        with segyio.open(self.path, 'r', ignore_geometry=True) as f:
                                f.mmap()
                                n_traces = len(f.trace)
                                n_samples = f.trace[0].size
                                dtype = np.float32

                                tmp_traces = traces_path.with_suffix('.dat.tmp')
                                mm = np.memmap(
                                        tmp_traces,
                                        dtype=dtype,
                                        mode='w+',
                                        shape=(n_traces, n_samples),
                                )
                                for i in range(n_traces):
                                        mm[i] = np.asarray(f.trace[i], dtype=dtype)
                                mm.flush()
                                del mm
                                os.replace(tmp_traces, traces_path)

                                key1s = f.attributes(self.key1_byte)[:]
                                key2s = f.attributes(self.key2_byte)[:]
                                (
                                        unique_key1,
                                        unique_key2,
                                        by_key1,
                                        by_key2,
                                ) = self._build_index_maps(key1s, key2s)
                                if len(unique_key1) <= 1:
                                        print("[warn] unique_key1 has only one value; check key1_byte/TraceField mapping")

                                tmp_keys = keys_path.with_suffix('.npz.tmp')
                                np.savez_compressed(
                                        tmp_keys,
                                        key1s=key1s,
                                        key2s=key2s,
                                        unique_key1=unique_key1,
                                        unique_key2=unique_key2,
                                )
                                os.replace(tmp_keys, keys_path)

                                tmp_index = indexmap_path.with_suffix('.json.tmp')
                                with open(tmp_index, 'w', encoding='utf-8') as f_idx:
                                        json.dump({'by_key1': by_key1, 'by_key2': by_key2}, f_idx)
                                os.replace(tmp_index, indexmap_path)

                                dt = None
                                try:
                                        dt = segyio.dt(f) / 1_000_000
                                except Exception:  # pragma: no cover - dt retrieval is optional
                                        dt = None

                        meta = {
                                'shape': [n_traces, n_samples],
                                'dtype': np.dtype(dtype).name,
                                'dt': dt,
                                'size': segy_size,
                                'mtime': segy_mtime,
                        }
                        tmp_meta = meta_path.with_suffix('.json.tmp')
                        with open(tmp_meta, 'w', encoding='utf-8') as f_meta:
                                json.dump(meta, f_meta)
                        os.replace(tmp_meta, meta_path)

                        self._traces_mm = np.memmap(
                                traces_path, dtype=dtype, mode='r', shape=(n_traces, n_samples)
                        )
                        self.key1s = key1s
                        self.key2s = key2s
                        self.unique_key1 = unique_key1
                        self.unique_key2 = unique_key2
                        self.indexmap = {'by_key1': by_key1, 'by_key2': by_key2}
                        self.dt = dt
                        return

                # file-level cache is valid; open traces
                self._traces_mm = np.memmap(
                        traces_path,
                        dtype=np.dtype(meta['dtype']),
                        mode='r',
                        shape=tuple(meta['shape']),
                )
                self.dt = meta.get('dt')

                # load or rebuild domain cache
                try:
                        with np.load(keys_path, allow_pickle=False) as npz:
                                self.key1s = npz['key1s']
                                self.key2s = npz['key2s']
                                self.unique_key1 = npz['unique_key1']
                                self.unique_key2 = npz['unique_key2']
                        with open(indexmap_path, encoding='utf-8') as f:
                                self.indexmap = json.load(f)
                except Exception:
                        for p in (keys_path, indexmap_path):
                                try:
                                        p.unlink()
                                except Exception:
                                        pass
                        with segyio.open(self.path, 'r', ignore_geometry=True) as f:
                                f.mmap()
                                key1s = f.attributes(self.key1_byte)[:]
                                key2s = f.attributes(self.key2_byte)[:]
                        (
                                unique_key1,
                                unique_key2,
                                by_key1,
                                by_key2,
                        ) = self._build_index_maps(key1s, key2s)
                        if len(unique_key1) <= 1:
                                print("[warn] unique_key1 has only one value; check key1_byte/TraceField mapping")
                        tmp_keys = keys_path.with_suffix('.npz.tmp')
                        np.savez_compressed(
                                tmp_keys,
                                key1s=key1s,
                                key2s=key2s,
                                unique_key1=unique_key1,
                                unique_key2=unique_key2,
                        )
                        os.replace(tmp_keys, keys_path)
                        tmp_index = indexmap_path.with_suffix('.json.tmp')
                        with open(tmp_index, 'w', encoding='utf-8') as f_idx:
                                json.dump({'by_key1': by_key1, 'by_key2': by_key2}, f_idx)
                        os.replace(tmp_index, indexmap_path)

                        self.key1s = key1s
                        self.key2s = key2s
                        self.unique_key1 = unique_key1
                        self.unique_key2 = unique_key2
                        self.indexmap = {'by_key1': by_key1, 'by_key2': by_key2}

                if len(self.unique_key1) <= 1:
                        print("[warn] unique_key1 has only one value; check key1_byte/TraceField mapping")

        # ------------------------------------------------------------------
        # Public API
        # ------------------------------------------------------------------
        def get_key1_values(self):
                return self.unique_key1

        def get_key2_values(self):
                return self.unique_key2

        def get_section(self, key1_val):
                if key1_val in self.section_cache:
                        return self.section_cache[key1_val]

                idxs = self.indexmap['by_key1'].get(str(int(key1_val)))
                if not idxs:
                        raise ValueError(f'Key1 value {key1_val} not found')

                traces = np.take(self._traces_mm, idxs, axis=0)
                key2_vals = np.take(self.key2s, idxs)
                order = np.argsort(key2_vals)
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
