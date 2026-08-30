from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "analyze_gate_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("analyze_gate_sensitivity", MODULE_PATH)
assert SPEC and SPEC.loader
SENSITIVITY = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC.loader.exec_module(SENSITIVITY)


def test_model_factory() -> None:
    assert SENSITIVITY.make_model("extra_trees", 7).random_state == 7
    assert SENSITIVITY.make_model("ridge_100", 7).steps[-1][1].alpha == 100.0
