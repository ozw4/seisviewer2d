"""Utility module for persisting seismic picks."""

from __future__ import annotations

import json
from pathlib import Path


class PickStore:
	"""In-memory store with JSON persistence for pick data."""

	def __init__(self, path: str | Path) -> None:
		"""Initialize the store with a file path."""
		self.path = Path(path)
		self.picks: dict[str, list[dict[str, int | float]]] = {}

	def load(self) -> None:
		"""Load picks from the JSON file if it exists."""
		if self.path.exists():
			self.picks = json.loads(self.path.read_text(encoding='utf-8'))
		else:
			self.picks = {}

	def save(self) -> None:
		"""Write the current picks to the JSON file."""
		self.path.write_text(
			json.dumps(self.picks, ensure_ascii=False, indent=2),
			encoding='utf-8',
		)

	def add_pick(
		self,
		file_id: str,
		trace: int,
		time: float,
		key1_idx: int,
		key1_byte: int,
	) -> None:
		"""Record or update a pick for a trace.

		Picks are uniquely identified by ``trace``, ``key1_idx`` and
		``key1_byte``.
		"""
		picks = self.picks.setdefault(file_id, [])
		for pick in picks:
			if (
				pick.get('trace') == trace
				and pick.get('key1_idx') == key1_idx
				and pick.get('key1_byte') == key1_byte
			):
				pick['time'] = time
				break
		else:
			picks.append(
				{
					'trace': trace,
					'time': time,
					'key1_idx': key1_idx,
					'key1_byte': key1_byte,
				}
			)

	def list_picks(
		self, file_id: str, key1_idx: int | None = None, key1_byte: int | None = None
	) -> list[dict[str, int | float]]:
		"""Return picks for ``file_id`` filtered by key information."""
		picks = self.picks.get(file_id, [])
		if key1_idx is None and key1_byte is None:
			return picks
		return [
			p
			for p in picks
			if (
				(key1_idx is None or p.get('key1_idx') == key1_idx)
				and (key1_byte is None or p.get('key1_byte') == key1_byte)
			)
		]

	def delete_pick(
		self,
		file_id: str,
		trace: int | None = None,
		key1_idx: int | None = None,
		key1_byte: int | None = None,
	) -> None:
		"""Remove picks for ``file_id``.

		If no filters are provided, all picks for ``file_id`` are removed.
		Otherwise, picks matching the provided ``trace``, ``key1_idx`` and
		``key1_byte`` are deleted.
		"""
		picks = self.picks.get(file_id)
		if not picks:
			return

		if trace is None and key1_idx is None and key1_byte is None:
			del self.picks[file_id]
			return

		self.picks[file_id] = [
			p
			for p in picks
			if not (
				(trace is None or p.get('trace') == trace)
				and (key1_idx is None or p.get('key1_idx') == key1_idx)
				and (key1_byte is None or p.get('key1_byte') == key1_byte)
			)
		]
		if not self.picks[file_id]:
			del self.picks[file_id]


store = PickStore(Path(__file__).with_name('picks.json'))


def add_pick(
	file_id: str,
	trace: int,
	time: float,
	key1_idx: int,
	key1_byte: int,
) -> None:
	"""Add a pick to the global store."""
	store.add_pick(file_id, trace, time, key1_idx, key1_byte)


def list_picks(
	file_id: str, key1_idx: int | None = None, key1_byte: int | None = None
) -> list[dict[str, int | float]]:
	"""List picks for ``file_id`` from the global store."""
	return store.list_picks(file_id, key1_idx, key1_byte)



def delete_pick(
	file_id: str,
	trace: int | None = None,
	key1_idx: int | None = None,
	key1_byte: int | None = None,
) -> None:
	"""Delete pick(s) for ``file_id`` from the global store."""
	store.delete_pick(file_id, trace, key1_idx, key1_byte)
