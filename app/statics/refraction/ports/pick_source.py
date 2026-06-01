"""First-break pick source ports for refraction workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class RefractionPickSourcePayload(Protocol):
    """Loaded refraction picks normalized to TraceStore sorted order."""

    picks_time_s_sorted: object
    sorted_trace_index: object
    source_kind: str
    metadata: dict[str, Any]


class RefractionPickSourceLoader(Protocol):
    """Load refraction first-break picks from external pick sources."""

    def load_npz(
        self,
        npz_path: Path,
        *,
        n_traces: int,
        n_samples: int,
        dt_s: float,
        sorted_trace_index: object,
        source_kind: str,
        allow_invalid_pick_values: bool = False,
    ) -> RefractionPickSourcePayload:
        """Load an NPZ pick source and normalize it to sorted trace order."""

    def load_manual_memmap(
        self,
        *,
        file_id: str,
        n_traces: int,
        sorted_trace_index: object,
    ) -> RefractionPickSourcePayload:
        """Load manual picks from the viewer's manual-pick memmap."""


__all__ = ['RefractionPickSourceLoader', 'RefractionPickSourcePayload']
