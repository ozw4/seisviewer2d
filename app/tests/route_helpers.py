from __future__ import annotations

from collections.abc import Iterable, Iterator

from starlette.routing import BaseRoute


def iter_app_routes(routes: Iterable[BaseRoute]) -> Iterator[BaseRoute]:
    """Yield concrete routes, including FastAPI lazy included-router contents."""
    for route in routes:
        original_router = getattr(route, 'original_router', None)
        nested_routes = getattr(original_router, 'routes', None)
        if nested_routes is not None:
            yield from iter_app_routes(nested_routes)
            continue
        yield route
