import numpy as np
import pytest

from app.utils.utils import SegySectionReader


def make_reader_for_test(key1s: np.ndarray, key2s: np.ndarray) -> SegySectionReader:
	# Create instance without running __init__ (avoids segy I/O)
	r = object.__new__(SegySectionReader)
	r.path = None
	r.key1_byte = 189
	r.key2_byte = 193
	r.section_cache = {}
	r._trace_seq_cache = {}
	r._trace_seq_disp_cache = {}
	r.key1s = np.asarray(key1s, dtype=np.int32)
	r.key2s = np.asarray(key2s, dtype=np.int32)
	r.unique_key1 = np.unique(r.key1s)
	r.ntraces = len(r.key1s)
	return r


def test_trace_seq_display_matches_legacy():
	key1s = np.array([1, 1, 2, 1, 2, 3, 3, 1, 2, 3], dtype=np.int32)
	key2s = np.array([5, 2, 9, 2, 1, 1, 1, 2, 5, 1], dtype=np.int32)
	reader = make_reader_for_test(key1s, key2s)

	for v in np.unique(key1s):
		indices = np.where(reader.key1s == v)[0]
		old_sorted = indices[np.argsort(reader.key2s[indices], kind='stable')]
		new_sorted = reader.get_trace_seq_for_section(int(v), align_to='display')
		assert np.array_equal(old_sorted, new_sorted)


def test_trace_seq_original_matches_where_order():
	key1s = np.array([10, 10, 20, 10, 20, 30, 30, 10, 20, 30], dtype=np.int32)
	key2s = np.array([5, 2, 9, 2, 1, 1, 1, 2, 5, 1], dtype=np.int32)
	reader = make_reader_for_test(key1s, key2s)

	for v in np.unique(key1s):
		indices = np.where(reader.key1s == v)[0]
		orig = reader.get_trace_seq_for_section(int(v), align_to='original')
		assert np.array_equal(indices, orig)


def test_trace_seq_raises_for_missing_key1():
	key1s = np.array([1, 1, 2, 3], dtype=np.int32)
	key2s = np.array([0, 1, 2, 3], dtype=np.int32)
	reader = make_reader_for_test(key1s, key2s)

	with pytest.raises(ValueError):
		reader.get_trace_seq_for_section(999999, align_to='display')
