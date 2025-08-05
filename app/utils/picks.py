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

	def add_pick(self, file_id: str, trace: int, time: float) -> None:
		"""Record a pick for a trace."""
		self.picks.setdefault(file_id, []).append({'trace': trace, 'time': time})

	def list_picks(self, file_id: str) -> list[dict[str, int | float]]:
		"""Return all picks for ``file_id``."""
		return self.picks.get(file_id, [])


store = PickStore(Path(__file__).with_name('picks.json'))


def add_pick(file_id: str, trace: int, time: float) -> None:
	"""Add a pick to the global store."""
	store.add_pick(file_id, trace, time)


def list_picks(file_id: str) -> list[dict[str, int | float]]:
	"""List picks for ``file_id`` from the global store."""
	return store.list_picks(file_id)
