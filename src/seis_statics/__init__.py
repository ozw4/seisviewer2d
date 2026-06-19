"""Standalone numerical core for seismic static-correction workflows.

The dependency direction is one-way: application repositories may import
``seis_statics``, but this package must stay independent from FastAPI,
Pydantic, SEG-Y readers, TraceStore, job runtimes, and artifact registries.
"""

__all__: list[str] = []
