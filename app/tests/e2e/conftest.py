import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
import pytest
import segyio


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for d in p.parents:
        if (d / "pyproject.toml").exists():
            return d
    raise AssertionError("repo root not found (pyproject.toml is missing in parents)")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _tail(path: Path, max_bytes: int = 8192) -> str:
    if not path.exists():
        return "<log file missing>"
    data = path.read_bytes()[-max_bytes:]
    return data.decode(errors="replace")


def _wait_http_ready(
    url: str, proc: subprocess.Popen, log_path: Path, timeout_s: float = 20.0
) -> None:
    deadline = time.time() + timeout_s
    last = None

    while time.time() < deadline:
        if proc.poll() is not None:
            raise AssertionError(
                "uvicorn exited before it became ready.\n"
                f"---- uvicorn log tail ----\n{_tail(log_path)}"
            )

        try:
            r = httpx.get(url, timeout=1.0, trust_env=False, follow_redirects=False)
            if 200 <= r.status_code < 400:
                return
            last = f"HTTP {r.status_code}"
        except httpx.RequestError as e:
            # 起動直後は connection refused が普通に出るのでリトライ扱い
            last = str(e)

        time.sleep(0.2)

    raise AssertionError(
        f"server not ready: {url} ({last})\n"
        f"---- uvicorn log tail ----\n{_tail(log_path)}"
    )


def _slug(nodeid: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", nodeid)
    return s.strip("_")


def _artifacts_root() -> Path:
    return Path(os.getenv("E2E_ARTIFACTS_DIR", "playwright-artifacts"))


def _contains_any(s: str, parts: tuple[str, ...]) -> bool:
    return any(p in s for p in parts)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


@pytest.fixture(scope="session")
def built_assets() -> None:
    assets = _repo_root() / "app/static/assets/main.js"
    if not assets.exists():
        raise AssertionError(
            f"missing {assets}; run: cd app && npm ci && npm run build"
        )


def _write_tiny_segy(path: Path) -> None:
    n_traces = 12
    n_samples = 200

    spec = segyio.spec()
    spec.format = 5  # IEEE float
    spec.samples = list(range(n_samples))
    spec.tracecount = n_traces

    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_traces, n_samples), dtype=np.float32)

    with segyio.create(str(path), spec) as f:
        f.bin[segyio.BinField.Interval] = 2000  # 2ms
        for i in range(n_traces):
            key1 = 100 + (i // 4)
            key2 = 10 + (i % 4)
            f.header[i] = {
                segyio.TraceField.INLINE_3D: int(key1),
                segyio.TraceField.CROSSLINE_3D: int(key2),
            }
            f.trace[i] = data[i]


@pytest.fixture(scope="session")
def tiny_segy_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("e2e_data") / "tiny.sgy"
    _write_tiny_segy(p)
    return p


@pytest.fixture(scope="session")
def base_url(tmp_path_factory: pytest.TempPathFactory, built_assets: None):
    external = os.getenv("E2E_BASE_URL")
    if external:
        url = external.rstrip("/")
        r = httpx.get(
            f"{url}/upload", timeout=2.0, trust_env=False, follow_redirects=False
        )
        if not (200 <= r.status_code < 400):
            raise AssertionError(f"E2E_BASE_URL returned {r.status_code}: {url}/upload")
        yield url
        return

    port = _free_port()
    data_dir = tmp_path_factory.mktemp("sv_app_data")
    log_dir = tmp_path_factory.mktemp("e2e_logs")
    log_path = log_dir / f"uvicorn_{port}.log"

    env = os.environ.copy()
    env["SV_APP_DATA_DIR"] = str(data_dir)
    env["PYTHONPATH"] = str(_repo_root())

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]

    log_fp = log_path.open("wb")
    proc = subprocess.Popen(cmd, env=env, stdout=log_fp, stderr=subprocess.STDOUT)

    try:
        _wait_http_ready(f"http://127.0.0.1:{port}/upload", proc, log_path)
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        log_fp.close()


@pytest.fixture(scope="session")
def browser_type_launch_args():
    return {"args": ["--no-proxy-server"]}


@pytest.fixture
def context(browser, request):
    ctx = browser.new_context()
    is_e2e = request.node.get_closest_marker("e2e") is not None
    art_dir = _artifacts_root() / _slug(request.node.nodeid)
    art_dir.mkdir(parents=True, exist_ok=True)

    if is_e2e:
        ctx.tracing.start(screenshots=True, snapshots=True, sources=True)

    yield ctx

    if is_e2e:
        ctx.tracing.stop(path=str(art_dir / "trace.zip"))
    ctx.close()


@dataclass
class E2EDebug:
    artifact_dir: Path
    not_found: list[str] = field(default_factory=list)  # "404 METHOD URL"
    request_failed: list[str] = field(
        default_factory=list
    )  # "REQ_FAILED METHOD URL (msg)"
    console_error: list[str] = field(default_factory=list)
    page_error: list[str] = field(default_factory=list)

    def unexpected_404(
        self,
        allow_404: tuple[str, ...] = ("favicon.ico", ".map"),
        allow_open_segy_404: bool = True,
    ) -> list[str]:
        out: list[str] = []
        for x in self.not_found:
            if _contains_any(x, allow_404):
                continue
            if allow_open_segy_404 and x.startswith("404 POST") and "/open_segy" in x:
                continue
            out.append(x)
        return out

    def unexpected_request_failed(
        self,
        allow_open_segy_aborted: bool = True,
        allow_favicon_aborted: bool = True,
    ) -> list[str]:
        out: list[str] = []
        for x in self.request_failed:
            if (
                allow_open_segy_aborted
                and "/open_segy" in x
                and "net::ERR_ABORTED" in x
            ):
                continue
            if allow_favicon_aborted and "favicon.ico" in x and "ERR_ABORTED" in x:
                continue
            out.append(x)
        return out

    def assert_clean(self) -> None:
        bad_404 = self.unexpected_404()
        bad_req = self.unexpected_request_failed()

        parts: list[str] = []
        if bad_404:
            parts.append("Unexpected 404 responses:\n" + "\n".join(bad_404))
        if self.page_error:
            parts.append("Page errors:\n" + "\n".join(self.page_error))
        if bad_req:
            parts.append("Request failed:\n" + "\n".join(bad_req))

        if parts:
            raise AssertionError("\n\n".join(parts))


@pytest.fixture
def e2e_debug(page, request) -> E2EDebug:
    art_dir = _artifacts_root() / _slug(request.node.nodeid)
    art_dir.mkdir(parents=True, exist_ok=True)

    dbg = E2EDebug(artifact_dir=art_dir)

    def on_response(resp):
        if resp.status == 404:
            dbg.not_found.append(f"404 {resp.request.method} {resp.url}")

    def on_requestfailed(req):
        fail = req.failure
        if fail is None:
            msg = "<no failure>"
        elif isinstance(fail, str):
            msg = fail
        elif isinstance(fail, dict):
            msg = fail.get("errorText") or fail.get("error_text") or str(fail)
        else:
            msg = getattr(fail, "error_text", str(fail))
        dbg.request_failed.append(f"REQ_FAILED {req.method} {req.url} ({msg})")

    def on_console(msg):
        if msg.type == "error":
            dbg.console_error.append(msg.text)

    def on_pageerror(err):
        dbg.page_error.append(str(err))

    page.on("response", on_response)
    page.on("requestfailed", on_requestfailed)
    page.on("console", on_console)
    page.on("pageerror", on_pageerror)

    yield dbg

    (art_dir / "not_found.txt").write_text(
        "\n".join(dbg.not_found) + ("\n" if dbg.not_found else ""), encoding="utf-8"
    )
    (art_dir / "request_failed.txt").write_text(
        "\n".join(dbg.request_failed) + ("\n" if dbg.request_failed else ""),
        encoding="utf-8",
    )
    (art_dir / "console_error.txt").write_text(
        "\n".join(dbg.console_error) + ("\n" if dbg.console_error else ""),
        encoding="utf-8",
    )
    (art_dir / "page_error.txt").write_text(
        "\n".join(dbg.page_error) + ("\n" if dbg.page_error else ""), encoding="utf-8"
    )

    rep_call = getattr(request.node, "rep_call", None)
    failed = bool(rep_call and rep_call.failed)

    if failed:
        page.screenshot(path=str(art_dir / "failure.png"), full_page=True)
        (art_dir / "failure.html").write_text(page.content(), encoding="utf-8")
