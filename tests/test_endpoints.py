"""Tests for the API endpoints."""

from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from app.api import endpoints
from app.main import app


class DummyReader:
    """Stub ``SegySectionReader`` for endpoint tests."""

    unique_key1: ClassVar[list[int]] = [1, 2, 3]

    def get_section(self, idx: int) -> list[list[int]]:  # noqa: ARG002
        """Return dummy section."""
        return [[0]]


def dummy_get_reader(
    _file_id: str, _key1_byte: int, _key2_byte: int
) -> DummyReader:
    """Return a :class:`DummyReader` instance."""
    return DummyReader()


def test_get_section_invalid_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """``get_section`` should return 400 for unknown key values."""
    monkeypatch.setattr(endpoints, "_get_reader", dummy_get_reader)
    client = TestClient(app)
    resp = client.get(
        "/get_section",
        params={"file_id": "id", "key1_byte": 1, "key2_byte": 2, "key1_idx": 99},
    )
    assert resp.status_code == 400  # noqa: S101, PLR2004
    assert resp.json() == {"detail": "Invalid key1_idx"}  # noqa: S101

