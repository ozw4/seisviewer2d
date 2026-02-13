import importlib


def test_import_only_has_no_root_mkdir_side_effect(monkeypatch):
    monkeypatch.setenv('PICKS_NPY_DIR', '/proc/forbidden')
    modname = 'app.utils.pick_cache_file1d_mem'
    mod = importlib.import_module(modname)
    importlib.reload(mod)
