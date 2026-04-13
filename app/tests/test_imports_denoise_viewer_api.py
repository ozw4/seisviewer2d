def test_import_infer_denoise_hw(monkeypatch):
    monkeypatch.setenv('NUMBA_CACHE_DIR', '/tmp/numba')
    from seisai_engine.viewer import infer_denoise_hw

    assert callable(infer_denoise_hw)
