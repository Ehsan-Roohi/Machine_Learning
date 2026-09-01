from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "analyze_half_range_gate.py"
SPEC = importlib.util.spec_from_file_location("analyze_half_range_gate", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_tally_reader_handles_multiple_snapshots(tmp_path: Path) -> None:
    path = tmp_path / "tally.dat"
    path.write_text(
        "ITEM: TIMESTEP\n1\nITEM: NUMBER OF TALLIES\n1\nITEM: BOX BOUNDS oo oo pp\n"
        "0 1\n0 1\n0 1\nITEM: TALLIES a b\n2 3\n"
        "ITEM: TIMESTEP\n2\nITEM: NUMBER OF TALLIES\n0\nITEM: BOX BOUNDS oo oo pp\n"
        "0 1\n0 1\n0 1\nITEM: TALLIES a b\n"
    )
    snapshots = MODULE.read_tally_snapshots(path, 2)
    assert [snapshot.timestep for snapshot in snapshots] == [1, 2]
    assert snapshots[0].values.shape == (1, 2)
    assert snapshots[1].values.shape == (0, 2)


def test_nrmse_is_relative_rms() -> None:
    target = np.asarray([1.0, -2.0, 3.0])
    assert np.isclose(MODULE.nrmse(target, 0.9 * target), 0.1)
