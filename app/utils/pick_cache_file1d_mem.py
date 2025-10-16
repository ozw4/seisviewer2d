import os
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


ROOT = Path(os.getenv("PICKS_NPY_DIR", "/workspace/app_data/picks_npy"))
ROOT.mkdir(parents=True, exist_ok=True)

DTYPE = np.float32
NAN = np.float32(np.nan)


def _safe(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


def _p(file_name: str) -> Path:
    return ROOT / f"{_safe(file_name)}.npy"


def _init_file(p: Path, ntraces: int) -> None:
    mm = open_memmap(p, mode="w+", dtype=DTYPE, shape=(ntraces,))
    mm[:] = NAN
    mm.flush()
    del mm


def _resize_file(p: Path, n_new: int) -> None:
    tmp = p.with_suffix(".tmp.npy")
    old = open_memmap(p, mode="r+", dtype=DTYPE)
    new = open_memmap(tmp, mode="w+", dtype=DTYPE, shape=(n_new,))
    new[:] = NAN
    n_copy = min(old.shape[0], n_new)
    if n_copy:
        new[:n_copy] = old[:n_copy]
    new.flush()
    del old
    del new
    os.replace(tmp, p)


def _ensure_shape(p: Path, ntraces: int) -> None:
    if not p.exists():
        _init_file(p, ntraces)
        return
    mm = open_memmap(p, mode="r", dtype=DTYPE)
    n_cur = int(mm.shape[0])
    del mm
    if n_cur != ntraces:
        _resize_file(p, ntraces)


def _open(file_name: str, ntraces: int, mode: str = "r+"):
    p = _p(file_name)
    _ensure_shape(p, ntraces)
    return open_memmap(p, mode=mode, dtype=DTYPE)


def set_by_traceseq(file_name: str, ntraces: int, trace_seq: int, time_s: float) -> None:
    if not (0 <= trace_seq < ntraces):
        raise RuntimeError(f"trace_seq out of range: {trace_seq}/{ntraces}")
    mm = _open(file_name, ntraces, "r+")
    mm[trace_seq] = np.float32(time_s)
    mm.flush()
    del mm


def clear_by_traceseq(file_name: str, ntraces: int, trace_seq: int) -> None:
    if not (0 <= trace_seq < ntraces):
        return
    mm = _open(file_name, ntraces, "r+")
    mm[trace_seq] = NAN
    mm.flush()
    del mm


def clear_section(file_name: str, ntraces: int, trace_seq_arr) -> None:
    idx = np.asarray(trace_seq_arr, dtype=np.int64)
    if idx.size == 0:
        return
    # Drop indices outside [0, ntraces) instead of raising.
    bad = (idx < 0) | (idx >= ntraces)
    if np.any(bad):
        idx = idx[~bad]
        if idx.size == 0:
            return
    mm = _open(file_name, ntraces, "r+")
    mm[idx] = NAN
    mm.flush()
    del mm


def to_pairs_for_section(file_name: str, ntraces: int, trace_seq_arr):
    idx = np.asarray(trace_seq_arr, dtype=np.int64)
    mm = _open(file_name, ntraces, "r")
    vals = mm[idx]
    del mm
    loc = np.where(~np.isnan(vals))[0]
    return [{"trace": int(i), "time": float(vals[i])} for i in loc]

