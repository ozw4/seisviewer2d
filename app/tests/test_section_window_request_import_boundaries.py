from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap


_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_section_window_request_import_does_not_load_http_or_segy_runtime() -> None:
    code = textwrap.dedent(
        """
        from __future__ import annotations

        import sys
        import app.services.section_window_request

        assert 'app.main' not in sys.modules
        assert 'app.api.routers.section' not in sys.modules
        assert 'segyio' not in sys.modules
        """
    )

    subprocess.run(
        [sys.executable, '-c', code],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
