import importlib.util
import types
from typing import Iterable, List

REQUIRED_STRATEGY_FUNCS = ["add_final_features", "make_env"]

def _load_module_with_required(py_path: str, required_funcs: Iterable[str], mod_name: str = "plugin_mod") -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(mod_name, py_path)
    assert spec and spec.loader, f"Cannot import module at {py_path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    missing: List[str] = [fn for fn in required_funcs if not hasattr(mod, fn)]
    if missing:
        raise ValueError(
            f"Module '{py_path}' missing required functions: {missing}. "
            f"Expected at least: {list(required_funcs)}"
        )
    return mod

def load_strategy_module(py_path: str) -> types.ModuleType:
    """Load a strategy file that provides: add_final_features, make_env"""
    return _load_module_with_required(py_path, REQUIRED_STRATEGY_FUNCS, mod_name="strategy_mod")

def load_backtest_module(py_path: str, fn_name: str = "backtest") -> types.ModuleType:
    """Load a backtest file that must expose a callable `backtest` (or custom name)."""
    return _load_module_with_required(py_path, [fn_name], mod_name="backtest_mod")