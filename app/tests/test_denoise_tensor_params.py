from pathlib import Path

import numpy as np
import pytest
import torch

from app.utils import denoise as denoise_mod


def test_denoise_tensor_passes_kwargs_with_default_tile_overlap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_infer(section_hw: np.ndarray, **kwargs: object) -> np.ndarray:
        captured['kwargs'] = kwargs
        return np.asarray(section_hw, dtype=np.float32)

    monkeypatch.setattr(denoise_mod, '_viewer_api', lambda: fake_infer)
    ckpt = tmp_path / 'dummy.pt'
    ckpt.write_bytes(b'')

    x = torch.arange(64 * 32, dtype=torch.float32).reshape(1, 1, 64, 32)
    _ = denoise_mod.denoise_tensor(
        x,
        chunk_h=64,
        overlap=16,
        mask_ratio=0.25,
        noise_std=0.75,
        mask_noise_mode='add',
        seed=777,
        passes_batch=3,
        use_amp=False,
        ckpt_path=ckpt,
    )

    kwargs = captured['kwargs']
    assert isinstance(kwargs, dict)
    assert kwargs['tile'] == (64, 6016)
    assert kwargs['overlap'] == (16, 1024)
    assert kwargs['mask_ratio'] == pytest.approx(0.25)
    assert kwargs['noise_std'] == pytest.approx(0.75)
    assert kwargs['mask_noise_mode'] == 'add'
    assert kwargs['seed'] == 777
    assert kwargs['passes_batch'] == 3
    assert kwargs['amp'] is False


def test_denoise_tensor_rejects_overlap_ge_chunk_h_without_tile() -> None:
    x = torch.zeros((1, 1, 16, 8), dtype=torch.float32)
    with pytest.raises(ValueError, match='overlap must satisfy overlap < chunk_h'):
        _ = denoise_mod.denoise_tensor(x, chunk_h=16, overlap=16)
