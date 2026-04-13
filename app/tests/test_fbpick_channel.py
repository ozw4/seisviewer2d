from __future__ import annotations

from pathlib import Path

import numpy as np

from app.api.routers import fbpick as fbpick_router_mod
from app.api.routers import fbpick_predict as fbpick_predict_mod
from app.utils import fbpick as fbpick_mod
from app.utils import ops as ops_mod


def test_fbpick_cache_key_isolated_by_channel() -> None:
    common = {
        'file_id': 'fid',
        'key1': 1,
        'key1_byte': 189,
        'key2_byte': 193,
        'offset_byte': None,
        'tile_h': 128,
        'tile_w': 6016,
        'overlap': 32,
        'amp': True,
        'pipeline_key': None,
        'tap_label': None,
        'model_id': 'fbpick_edgenext_small.pt',
        'model_ver': 'fbpick_edgenext_small.pt:1',
    }
    key_default = fbpick_router_mod._build_fbpick_cache_key(**common, channel=None)
    key_p = fbpick_router_mod._build_fbpick_cache_key(**common, channel='P')
    key_s = fbpick_router_mod._build_fbpick_cache_key(**common, channel='S')
    assert key_default == key_p
    assert key_p != key_s


def test_op_fbpick_passes_channel_to_infer_prob_map(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_infer_prob_map(section, **kwargs):
        captured['shape'] = tuple(section.shape)
        captured.update(kwargs)
        return np.zeros(section.shape, dtype=np.float32)

    monkeypatch.setattr(ops_mod, 'infer_prob_map', _fake_infer_prob_map, raising=True)
    section = np.ones((4, 6), dtype=np.float32)
    out = ops_mod.op_fbpick(
        section,
        params={'channel': 'S', 'amp': True},
        meta={},
    )
    assert tuple(out['prob'].shape) == (4, 6)
    assert captured.get('channel') == 'S'


def test_fbpick_predict_last_prob_cache_isolated_by_channel(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_resolve_model_selection(_model_id):
        return fbpick_predict_mod._ModelSelection(
            model_id='fbpick_edgenext_small.pt',
            model_path=Path('model/fbpick_edgenext_small.pt'),
            model_ver='fbpick_edgenext_small.pt:1',
            uses_offset=False,
        )

    def _fake_compute_probability_map(req, *, state, model_sel):
        _ = (state, model_sel)
        calls.append('P' if req.channel is None else req.channel)
        return fbpick_predict_mod._ProbabilityPayload(
            prob=np.zeros((3, 5), dtype=np.float32),
            dt=0.002,
            source='raw',
        )

    monkeypatch.setattr(
        fbpick_predict_mod,
        '_resolve_model_selection',
        _fake_resolve_model_selection,
        raising=True,
    )
    monkeypatch.setattr(
        fbpick_predict_mod,
        '_compute_probability_map',
        _fake_compute_probability_map,
        raising=True,
    )
    monkeypatch.setattr(
        fbpick_predict_mod,
        '_last_prob_state',
        fbpick_predict_mod._LastProbabilityState(),
        raising=True,
    )

    req_s = fbpick_predict_mod.FbpickPredictRequest(
        file_id='fid',
        key1=1,
        sigma_ms_max=10.0,
        channel='S',
    )
    req_p = fbpick_predict_mod.FbpickPredictRequest(
        file_id='fid',
        key1=1,
        sigma_ms_max=10.0,
        channel='P',
    )
    _ = fbpick_predict_mod._load_probability_map(req_s, state=object())
    _ = fbpick_predict_mod._load_probability_map(req_s, state=object())
    _ = fbpick_predict_mod._load_probability_map(req_p, state=object())
    assert calls == ['S', 'P']


def test_infer_prob_map_passes_channel_to_infer_prob_hw(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_infer_prob_hw(section_hw, **kwargs):
        captured['shape'] = tuple(section_hw.shape)
        captured.update(kwargs)
        return np.zeros(section_hw.shape, dtype=np.float32)

    monkeypatch.setattr(
        fbpick_mod,
        '_viewer_api',
        lambda: (_fake_infer_prob_hw, lambda *args, **kwargs: None),
        raising=True,
    )

    section = np.ones((6, 8), dtype=np.float32)
    out = fbpick_mod.infer_prob_map(
        section,
        model_path=Path('/tmp/fbpick_edgenext_small.pt'),
        channel='ch2',
    )
    assert tuple(out.shape) == (6, 8)
    assert captured.get('channel') == 'ch2'
