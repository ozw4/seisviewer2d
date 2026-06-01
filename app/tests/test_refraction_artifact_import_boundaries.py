from __future__ import annotations

from pathlib import Path

ARTIFACTS_ROOT = (
    Path(__file__).resolve().parents[1] / 'statics' / 'refraction' / 'artifacts'
)


def test_refraction_artifacts_do_not_import_application_layer() -> None:
    for source_path in sorted(ARTIFACTS_ROOT.glob('*.py')):
        source = source_path.read_text(encoding='utf-8')
        assert 'app.statics.refraction.application' not in source, (
            f'{source_path.relative_to(ARTIFACTS_ROOT)} imports application layer'
        )
