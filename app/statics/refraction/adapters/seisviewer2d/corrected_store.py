"""SeisViewer2D corrected TraceStore adapter for refraction workflows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.core.state import AppState
from app.services.trace_store_registration import register_trace_store
from app.statics.common.corrected_store import (
    build_and_register_time_shifted_trace_store,
    build_time_shifted_trace_store,
    cleanup_artifact,
    cleanup_registration,
    cleanup_store,
    corrected_store_path,
    safe_store_name_component,
    verify_registered_trace_store,
)
from app.statics.refraction.ports.corrected_store import (
    RefractionCorrectedStoreProvider,
)


@dataclass(frozen=True)
class SeisViewer2DRefractionCorrectedStoreProvider(
    RefractionCorrectedStoreProvider
):
    """Expose corrected TraceStore operations through a refraction port."""

    state: AppState

    def corrected_store_path(
        self,
        *,
        source_store_path: Path,
        statics_kind: str,
        suffix: str,
        error_type: type[Exception],
        output_name: str | None = None,
    ) -> Path:
        if output_name is not None:
            output_path = source_store_path.parent / safe_store_name_component(
                output_name,
                error_type=error_type,
            )
            if output_path.exists() or output_path.is_symlink():
                raise error_type(f'corrected output path already exists: {output_path}')
            return output_path
        return corrected_store_path(
            source_store_path=source_store_path,
            statics_kind=statics_kind,
            suffix=suffix,
            error_type=error_type,
        )

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
        return build_and_register_time_shifted_trace_store(
            state=self.state,
            corrected_file_id=corrected_file_id,
            source_store_path=source_store_path,
            output_store_path=output_store_path,
            trace_shift_s_sorted=trace_shift_s_sorted,
            fill_value=fill_value,
            output_dtype=output_dtype,
            derived_metadata=derived_metadata,
            from_file_id=from_file_id,
            original_segy_path=original_segy_path,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
            header_bytes_to_materialize=tuple(header_bytes_to_materialize),
            preload_header_bytes=tuple(preload_header_bytes),
            register_fn=register_trace_store,
        )

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
        return build_time_shifted_trace_store(
            source_store_path=source_store_path,
            output_store_path=output_store_path,
            trace_shift_s_sorted=trace_shift_s_sorted,
            fill_value=fill_value,
            output_dtype=output_dtype,
            derived_metadata=derived_metadata,
            from_file_id=from_file_id,
            original_segy_path=original_segy_path,
            header_bytes_to_materialize=tuple(header_bytes_to_materialize),
        )

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
        return register_trace_store(
            state=self.state,
            file_id=file_id,
            store_dir=store_dir,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
            dt=dt,
            update_registry=update_registry,
            touch_meta=touch_meta,
            preload_header_bytes=tuple(preload_header_bytes),
        )

    def verify_registered_trace_store(
        self,
        *,
        file_id: str,
        store_path: Path,
        key1_byte: int,
        key2_byte: int,
        reader: object,
    ) -> None:
        verify_registered_trace_store(
            state=self.state,
            file_id=file_id,
            store_path=store_path,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
            reader=reader,
        )

    def cleanup_registration(
        self,
        *,
        file_id: str,
        key1_byte: int,
        key2_byte: int,
    ) -> None:
        cleanup_registration(
            self.state,
            file_id=file_id,
            key1_byte=key1_byte,
            key2_byte=key2_byte,
        )

    def cleanup_store(self, output_path: Path) -> None:
        cleanup_store(output_path)

    def cleanup_artifact(self, path: Path) -> None:
        cleanup_artifact(path)


__all__ = ['SeisViewer2DRefractionCorrectedStoreProvider']
