import pytest
from pydantic import ValidationError

from app.api.schemas import PipelineOp


def test_denoise_params_allow_pair_overlap_when_tile_is_set() -> None:
    op = PipelineOp(
        kind='transform',
        name='denoise',
        params={
            'chunk_h': 128,
            'tile': [128, 6016],
            'overlap': [16, 1024],
        },
    )

    assert tuple(op.params['tile']) == (128, 6016)
    assert tuple(op.params['overlap']) == (16, 1024)


def test_denoise_params_reject_pair_overlap_without_tile() -> None:
    with pytest.raises(ValidationError, match='overlap must be an int'):
        PipelineOp(
            kind='transform',
            name='denoise',
            params={
                'chunk_h': 128,
                'overlap': [16, 1024],
            },
        )
