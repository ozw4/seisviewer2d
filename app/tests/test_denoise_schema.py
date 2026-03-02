import pytest
from pydantic import ValidationError

from app.api.schemas import PipelineOp, PipelineSpec
from app.utils.pipeline import pipeline_key


def test_denoise_params_reject_overlap_ge_chunk_h() -> None:
    with pytest.raises(ValidationError, match='overlap must be less than chunk_h'):
        PipelineOp(
            kind='transform',
            name='denoise',
            params={'chunk_h': 128, 'overlap': 128},
        )


def test_denoise_pipeline_key_is_canonicalized_for_direct_mode() -> None:
    spec_a = PipelineSpec(
        steps=[
            {
                'kind': 'transform',
                'name': 'denoise',
                'params': {
                    'chunk_h': 128,
                    'overlap': 32,
                    'mask_ratio': 0.0,
                    'noise_std': 0.2,
                    'mask_noise_mode': 'add',
                    'seed': 7,
                    'passes_batch': 9,
                },
            }
        ]
    )
    spec_b = PipelineSpec(
        steps=[
            {
                'kind': 'transform',
                'name': 'denoise',
                'params': {
                    'chunk_h': 128,
                    'overlap': 32,
                    'mask_ratio': 0.0,
                    'noise_std': 2.0,
                    'mask_noise_mode': 'replace',
                    'seed': 12345,
                    'passes_batch': 4,
                },
            }
        ]
    )

    params_a = spec_a.steps[0].params
    params_b = spec_b.steps[0].params

    assert params_a == params_b
    assert params_a['noise_std'] == pytest.approx(1.0)
    assert params_a['mask_noise_mode'] == 'replace'
    assert params_a['seed'] == 12345
    assert params_a['passes_batch'] == 4
    assert pipeline_key(spec_a) == pipeline_key(spec_b)
