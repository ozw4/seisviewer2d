from __future__ import annotations

import ast
import importlib
from pathlib import Path
import subprocess
import sys
import textwrap


_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPARE_MODULES = {
    'app.api.routers.compare',
    'app.services.raw_compare_validation',
}
_FORBIDDEN_MODULES = {
    'app.api.routers.section',
    'app.utils.ingest',
    'app.utils.header_qc',
    'segyio',
}


def _imports_from(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    return imports


def test_compare_router_import_does_not_import_section_router() -> None:
    for name in list(sys.modules):
        if name in {'app.api.routers.compare', 'app.api.routers.section'}:
            sys.modules.pop(name, None)

    importlib.import_module('app.api.routers.compare')

    assert 'app.api.routers.section' not in sys.modules


def test_compare_router_and_service_import_without_segyio_or_ingest_modules() -> None:
    code = textwrap.dedent(
        """
        from __future__ import annotations

        import importlib
        import sys

        compare_modules = {
            'app.api.routers.compare',
            'app.services.raw_compare_validation',
        }
        forbidden_modules = {
            'app.api.routers.section',
            'app.utils.ingest',
            'app.utils.header_qc',
            'segyio',
        }

        for name in list(sys.modules):
            if name in compare_modules or name in forbidden_modules:
                sys.modules.pop(name, None)

        sys.modules['segyio'] = None
        for module_name in sorted(compare_modules):
            importlib.import_module(module_name)

        assert 'app.api.routers.section' not in sys.modules
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


def test_compare_modules_do_not_directly_import_forbidden_boundaries() -> None:
    module_paths = {
        'app.api.routers.compare': _REPO_ROOT / 'app/api/routers/compare.py',
        'app.services.raw_compare_validation': _REPO_ROOT
        / 'app/services/raw_compare_validation.py',
    }
    forbidden_imports = _FORBIDDEN_MODULES - {'segyio'}

    for module_name, path in module_paths.items():
        imports = _imports_from(path)
        assert imports.isdisjoint(forbidden_imports), module_name
