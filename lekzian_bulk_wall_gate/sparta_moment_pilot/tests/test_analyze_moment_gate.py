from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "analyze_moment_gate.py"
SPEC = importlib.util.spec_from_file_location("analyze_moment_gate", MODULE_PATH)
assert SPEC and SPEC.loader
ANALYSIS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYSIS
SPEC.loader.exec_module(ANALYSIS)


def test_nrmse_is_scale_normalized() -> None:
    y = np.array([1.0, -2.0, 3.0])
    prediction = y + np.array([0.1, -0.2, 0.3])
    assert np.isclose(ANALYSIS.nrmse(y, prediction), 0.1)


def test_local_moment_projection() -> None:
    columns = {"momxx": 0, "momyy": 1, "momxy": 2, "momzz": 3}
    rows = np.array([[4.0, 2.0, 1.0, 3.0]])
    tangent = np.array([[1.0, 0.0]])
    normal = np.array([[0.0, 1.0]])
    pnn, pnt, ptt, pzz = ANALYSIS._project_moments(rows, columns, tangent, normal)
    assert np.allclose([pnn[0], pnt[0], ptt[0], pzz[0]], [2.0, 1.0, 4.0, 3.0])


def test_duplicate_centers_keep_populated_subcell() -> None:
    columns = {"xc": 0, "yc": 1, "n": 2}
    grid = np.array([[0.5, 0.5, 2.0], [0.5, 0.5, 20.0], [1.5, 0.5, 5.0]])
    collapsed, _ = ANALYSIS._collapse_duplicate_centers(grid, columns)
    assert len(collapsed) == 2
    assert 20.0 in collapsed[:, 2]
    assert 2.0 not in collapsed[:, 2]
