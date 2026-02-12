import importlib
import sys

import torch


def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
	raise AssertionError('torch.load must not be called during import/reload')


def _import_and_reload(modname: str) -> None:
	module = sys.modules.get(modname)
	if module is None:
		module = importlib.import_module(modname)
	importlib.reload(module)


def test_import_only_does_not_load_models(monkeypatch):
	monkeypatch.setattr(torch, 'load', _boom, raising=True)
	_import_and_reload('app.utils.denoise')
	_import_and_reload('app.utils.fbpick')
