"""Smoke tests for the standalone seis_statics package boundary."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'


def test_seis_statics_package_imports_with_app_runtime_blocked() -> None:
    script = textwrap.dedent(
        """
        from __future__ import annotations

        import importlib
        import importlib.abc
        import os
        from pathlib import Path
        import sys

        BLOCKED = ('app', 'fastapi', 'pydantic', 'segyio')
        SRC_ROOT = Path(os.environ['SEIS_STATICS_SRC_ROOT']).resolve()

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == name or fullname.startswith(f'{name}.') for name in BLOCKED):
                    raise ImportError(f'blocked runtime import: {fullname}')
                return None

        sys.meta_path.insert(0, Blocker())

        seis_statics = importlib.import_module('seis_statics')
        datum = importlib.import_module('seis_statics.datum')
        residual = importlib.import_module('seis_statics.residual')
        validation = importlib.import_module('seis_statics.validation')

        assert seis_statics.__name__ == 'seis_statics'
        assert Path(seis_statics.__file__).resolve().is_relative_to(SRC_ROOT)
        assert datum.__name__ == 'seis_statics.datum'
        assert residual.__name__ == 'seis_statics.residual'
        assert validation.__name__ == 'seis_statics.validation'
        assert hasattr(validation, 'coerce_1d_finite_float64')

        imported_blocked = [
            name
            for name in sys.modules
            if any(name == blocked or name.startswith(f'{blocked}.') for blocked in BLOCKED)
        ]
        assert imported_blocked == []
        """
    )
    env = os.environ.copy()
    env['SEIS_STATICS_SRC_ROOT'] = str(SRC_ROOT)
    env['PYTHONPATH'] = os.pathsep.join(
        path
        for path in [str(SRC_ROOT), env.get('PYTHONPATH', '')]
        if path
    )

    subprocess.run(
        [sys.executable, '-c', script],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )


def test_public_validation_exports_private_validation_helpers() -> None:
    from seis_statics import _validation
    from seis_statics.validation import coerce_1d_finite_float64
    import seis_statics.validation as validation

    assert validation.__all__ == _validation.__all__
    assert coerce_1d_finite_float64 is _validation.coerce_1d_finite_float64
