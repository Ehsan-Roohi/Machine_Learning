#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_reference_benchmark import _comparison, run


def test_reference_comparison_catches_factor_two_normalization() -> None:
    correct = _comparison({"D": 0.1585, "G": 0.1893}, 0.010)
    assert correct["within_diagnostic_tolerance"]
    wrong = _comparison({"D": 0.07925, "G": 0.1893}, 0.010)
    assert not wrong["within_diagnostic_tolerance"]
    assert abs(wrong["error_percent"]["D"] + 50.0) < 1.0e-12


def test_small_linear_driver_writes_auditable_artifacts() -> None:
    with tempfile.TemporaryDirectory() as directory:
        args = argparse.Namespace(
            nx=4,
            ny=4,
            kn=0.01,
            lid_velocity=0.05,
            output_dir=Path(directory),
            direct_linear=True,
            nonlinear=False,
            qmr_rtol=1.0e-10,
            qmr_maxiter=2000,
            nonlinear_rtol=1.0e-8,
            nonlinear_update_tol=1.0e-7,
            nonlinear_maxiter=10,
            minimum_line_step=1.0 / 128.0,
        )
        result = run(args)
        assert (Path(directory) / "linear_state.npy").exists()
        assert (Path(directory) / "benchmark_report.json").exists()
        assert result["linear"]["state_sha256"]
        assert result["gates"]["algebraic_convergence"]
        assert result["publication_grade"] is False


if __name__ == "__main__":
    test_reference_comparison_catches_factor_two_normalization()
    test_small_linear_driver_writes_auditable_artifacts()
    print("Rana 2013 reference-benchmark tests: PASS")
