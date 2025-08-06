import threading
from collections.abc import Mapping


class PickStore:
    """Thread-safe in-memory storage for picks.

    Picks are stored per file ID with trace indices mapped to pick values.
    """

    def __init__(self):
        self._picks: dict[str, dict[int, float]] = {}
        self._lock = threading.Lock()

    def add_or_update(self, file_id: str, picks: Mapping[int, float]) -> dict[int, float]:
        """Add new picks or update existing ones for a file.

        Args:
            file_id: Identifier of the SEG-Y file.
            picks: Mapping of trace index to pick value.

        Returns:
            The updated picks for the file.

        """
        with self._lock:
            store = self._picks.setdefault(file_id, {})
            for k, v in picks.items():
                store[int(k)] = float(v)
            return store.copy()

    def get(self, file_id: str) -> dict[int, float]:
        """Retrieve picks for a file.

        Args:
            file_id: Identifier of the SEG-Y file.

        Returns:
            Mapping of trace index to pick value. Empty if none.

        """
        with self._lock:
            return self._picks.get(file_id, {}).copy()

    def delete(self, file_id: str, trace_idx: int | None = None) -> None:
        """Delete picks for a file or a specific trace.

        Args:
            file_id: Identifier of the SEG-Y file.
            trace_idx: Optional trace index to remove. If ``None`` all picks for the
                file are removed.

        """
        with self._lock:
            if trace_idx is None:
                self._picks.pop(file_id, None)
            else:
                picks = self._picks.get(file_id)
                if picks is not None:
                    picks.pop(trace_idx, None)
                    if not picks:
                        self._picks.pop(file_id, None)
