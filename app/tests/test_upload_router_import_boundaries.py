from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap


_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_upload_router_and_services_import_without_segyio() -> None:
    code = textwrap.dedent(
        """
        from __future__ import annotations

        import importlib
        import sys

        upload_modules = {
            'app.api.routers.upload',
            'app.services.segy_upload_service',
            'app.services.segy_ingest_service',
            'app.services.segy_open_service',
            'app.services.compare_raw_import_service',
            'app.services.staged_segy_upload_service',
        }
        forbidden_modules = {'app.utils.ingest', 'app.utils.header_qc', 'segyio'}

        for name in list(sys.modules):
            if name in upload_modules or name in forbidden_modules:
                sys.modules.pop(name, None)

        sys.modules['segyio'] = None
        for module_name in sorted(upload_modules):
            importlib.import_module(module_name)

        assert 'app.utils.ingest' not in sys.modules
        assert 'app.utils.header_qc' not in sys.modules
        assert sys.modules['segyio'] is None
        """
    )

    subprocess.run(
        [sys.executable, '-c', code],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
