# app/tests/test_missing_endpoints_fbpick_job_status.py
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client() -> TestClient:
    state = app.state.sv
    state.jobs.clear()
    state.fbpick_cache.clear()
    with TestClient(app) as c:
        yield c
    state.jobs.clear()
    state.fbpick_cache.clear()


def test_fbpick_job_status_unknown_job_id_returns_404(client: TestClient):
    r = client.get("/fbpick_job_status", params={"job_id": "no-such-id"})
    assert r.status_code == 404
    assert r.json().get("detail") == "Job ID not found"


def test_fbpick_job_status_reports_status_transitions_and_message(client: TestClient):
    state = app.state.sv
    job_id = "job-1"
    state.jobs[job_id] = {"status": "queued"}

    r1 = client.get("/fbpick_job_status", params={"job_id": job_id})
    assert r1.status_code == 200
    assert r1.json() == {"status": "queued", "message": ""}

    state.jobs[job_id]["status"] = "running"
    r2 = client.get("/fbpick_job_status", params={"job_id": job_id})
    assert r2.status_code == 200
    assert r2.json() == {"status": "running", "message": ""}

    state.jobs[job_id]["status"] = "done"
    r3 = client.get("/fbpick_job_status", params={"job_id": job_id})
    assert r3.status_code == 200
    assert r3.json() == {"status": "done", "message": ""}

    state.jobs[job_id]["status"] = "error"
    state.jobs[job_id]["message"] = "boom"
    r4 = client.get("/fbpick_job_status", params={"job_id": job_id})
    assert r4.status_code == 200
    assert r4.json() == {"status": "error", "message": "boom"}


def test_fbpick_job_status_defaults_to_unknown_when_status_missing(client: TestClient):
    state = app.state.sv
    job_id = "job-2"
    state.jobs[job_id] = {}

    r = client.get("/fbpick_job_status", params={"job_id": job_id})
    assert r.status_code == 200
    assert r.json() == {"status": "unknown", "message": ""}
