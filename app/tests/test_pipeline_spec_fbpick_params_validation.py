import pytest
from pydantic import ValidationError

from app.api.schemas import PipelineOp


def test_fbpick_params_empty_dict_is_allowed() -> None:
    op = PipelineOp(kind='analyzer', name='fbpick', params={})
    assert op.params == {}


@pytest.mark.parametrize(
    'params',
    [
        {'overlap': 0},
        {'overlap': [32]},
        {'tile': [128]},
        {'tile': 128},
    ],
)
def test_fbpick_rejects_invalid_overlap_and_tile(params: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        PipelineOp(kind='analyzer', name='fbpick', params=params)


@pytest.mark.parametrize(
    'model_id',
    [
        'fbpick_xxx.pth',
        'dir/fbpick_x.pt',
    ],
)
def test_fbpick_rejects_invalid_model_id(model_id: str) -> None:
    with pytest.raises(ValidationError):
        PipelineOp(
            kind='analyzer',
            name='fbpick',
            params={'model_id': model_id},
        )


def test_fbpick_accepts_valid_model_id_format() -> None:
    op = PipelineOp(
        kind='analyzer',
        name='fbpick',
        params={'model_id': 'fbpick_ok.pt'},
    )
    assert op.params == {'model_id': 'fbpick_ok.pt'}


@pytest.mark.parametrize(
    'params',
    [
        {'overlap': (32, 32)},
        {'tile': [128, 6016]},
    ],
)
def test_fbpick_accepts_valid_overlap_and_tile_shapes(
    params: dict[str, object],
) -> None:
    op = PipelineOp(kind='analyzer', name='fbpick', params=params)
    assert op.params == params


@pytest.mark.parametrize(
    'params',
    [
        {'chunk_h': 128},
        {'tile_w': 6016},
    ],
)
def test_fbpick_rejects_legacy_chunk_h_and_tile_w(params: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        PipelineOp(kind='analyzer', name='fbpick', params=params)
