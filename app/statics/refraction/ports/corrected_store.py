"""Corrected TraceStore build and registration port."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class RefractionCorrectedStoreProvider(Protocol):
    """Build, register, verify, and clean up corrected TraceStores."""

    def corrected_store_path(
        self,
        *,
        source_store_path: Path,
        statics_kind: str,
        suffix: str,
        error_type: type[Exception],
        output_name: str | None = None,
    ) -> Path:
        """Return the corrected TraceStore output path."""

    def build_and_register_time_shifted_trace_store(
        self,
        *,
        corrected_file_id: str,
        source_store_path: Path,
        output_store_path: Path,
        trace_shift_s_sorted: np.ndarray,
        fill_value: float,
        output_dtype: str,
        derived_metadata: dict[str, Any],
        from_file_id: str,
        original_segy_path: str | None,
        key1_byte: int,
        key2_byte: int,
        header_bytes_to_materialize: Sequence[int],
        preload_header_bytes: Sequence[int],
    ) -> Any:
        """Build and register a corrected TraceStore."""

    def build_time_shifted_trace_store(
        self,
        *,
        source_store_path: Path,
        output_store_path: Path,
        trace_shift_s_sorted: np.ndarray,
        fill_value: float,
        output_dtype: str,
        derived_metadata: dict[str, Any],
        from_file_id: str,
        original_segy_path: str | None,
        header_bytes_to_materialize: Sequence[int],
    ) -> Any:
        """Build a corrected TraceStore without registering it."""

    def register_trace_store(
        self,
        *,
        file_id: str,
        store_dir: Path,
        key1_byte: int,
        key2_byte: int,
        dt: float,
        update_registry: bool,
        touch_meta: bool,
        preload_header_bytes: Sequence[int],
    ) -> Any:
        """Register an existing TraceStore and return its reader."""

    def verify_registered_trace_store(
        self,
        *,
        file_id: str,
        store_path: Path,
        key1_byte: int,
        key2_byte: int,
        reader: object,
    ) -> None:
        """Verify corrected TraceStore registration."""

    def cleanup_registration(
        self,
        *,
        file_id: str,
        key1_byte: int,
        key2_byte: int,
    ) -> None:
        """Remove a failed corrected TraceStore registration."""

    def cleanup_store(self, output_path: Path) -> None:
        """Remove failed corrected TraceStore directories."""

    def cleanup_artifact(self, path: Path) -> None:
        """Remove a failed corrected TraceStore artifact."""


__all__ = ['RefractionCorrectedStoreProvider']
