"""Service-layer domain exceptions."""

from __future__ import annotations


class DomainError(Exception):
    """Base error for service/domain failures."""

    status_code: int
    detail: str

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = int(status_code)
        self.detail = str(detail)
        super().__init__(self.detail)


class BadRequestError(DomainError):
    """Domain error mapped to HTTP 400."""

    def __init__(self, detail: str) -> None:
        super().__init__(400, detail)


class NotFoundError(DomainError):
    """Domain error mapped to HTTP 404."""

    def __init__(self, detail: str) -> None:
        super().__init__(404, detail)


class ConflictError(DomainError):
    """Domain error mapped to HTTP 409."""

    def __init__(self, detail: str) -> None:
        super().__init__(409, detail)


class UnprocessableError(DomainError):
    """Domain error mapped to HTTP 422."""

    def __init__(self, detail: str) -> None:
        super().__init__(422, detail)


class InternalError(DomainError):
    """Domain error mapped to HTTP 500."""

    def __init__(self, detail: str) -> None:
        super().__init__(500, detail)


__all__ = [
    'BadRequestError',
    'ConflictError',
    'DomainError',
    'InternalError',
    'NotFoundError',
    'UnprocessableError',
]
