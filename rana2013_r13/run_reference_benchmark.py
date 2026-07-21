#!/usr/bin/env python3
"""Reproducible gated driver for the independent Rana-2013 R13 solver.

The driver deliberately keeps verification, paper comparison, and production
claims separate.  A finite solve is not sufficient: the algebraic residual,
mass constraint, thermodynamic positivity, and D/G comparison are all emitted
as independent gates.  Nonlinear runs start from the converged linear R13
state, matching Section 4.4 of Rana et al. rather than a pseudo-time restart.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    from .linear_reference_solver import (
        IDX,
        LinearR13Config,
        cavity_metrics,
        solve_linear_r13,
    )
    from .nonlinear_reference_solver import (
        NonlinearR13Config,
        nonlinear_cavity_metrics,
        solve_nonlinear_r13,
    )
except ImportError:
    from linear_reference_solver import (
        IDX,
        LinearR13Config,
        cavity_metrics,
        solve_linear_r13,
    )
    from nonlinear_reference_solver import (
        NonlinearR13Config,
        nonlinear_cavity_metrics,
        solve_nonlinear_r13,
    )


PAPER_REFERENCE = {
    0.010: {"D": 0.1585, "G": 0.1893},
    0.071: {"D": 0.4271, "G": 0.1428},
    0.141: {"D": 0.5084, "G": 0.1216},
    0.354: {"D": 0.5644, "G": 0.1044},
    0.707: {"D": 0.5722, "G": 0.1003},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _comparison(metrics: dict[str, float], kn: float) -> dict[str, object]:
    reference = PAPER_REFERENCE.get(round(kn, 3))
    if reference is None:
        return {"available": False}
    errors = {
        key: 100.0 * (float(metrics[key]) - reference[key]) / reference[key]
        for key in ("D", "G")
    }
    return {
        "available": True,
        "paper": reference,
        "error_percent": errors,
        "within_diagnostic_tolerance": bool(
            abs(errors["D"]) <= 2.0 and abs(errors["G"]) <= 7.0
        ),
        "tolerance_percent": {"D": 2.0, "G": 7.0},
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    linear_config = LinearR13Config(
        nx=args.nx,
        ny=args.ny,
        kn=args.kn,
        lid_velocity=args.lid_velocity,
        qmr_rtol=args.qmr_rtol,
        qmr_maxiter=args.qmr_maxiter,
    )
    started = time.monotonic()
    linear_state, linear_solver = solve_linear_r13(
        linear_config, direct=args.direct_linear
    )
    linear_metrics = cavity_metrics(linear_state, linear_config)
    linear_path = args.output_dir / "linear_state.npy"
    np.save(linear_path, linear_state)
    result: dict[str, object] = {
        "reference": (
            "Rana, Torrilhon & Struchtrup, Journal of Computational Physics "
            "236 (2013) 169-186"
        ),
        "git_revision": _git_revision(),
        "configuration": linear_config.__dict__,
        "linear": {
            "solver": linear_solver,
            "metrics": linear_metrics,
            "paper_comparison": _comparison(linear_metrics, args.kn),
            "state": str(linear_path),
            "state_sha256": _sha256(linear_path),
        },
    }
    if args.nonlinear:
        nonlinear_config = NonlinearR13Config(
            **linear_config.__dict__,
            nonlinear_rtol=args.nonlinear_rtol,
            nonlinear_update_tol=args.nonlinear_update_tol,
            nonlinear_maxiter=args.nonlinear_maxiter,
            minimum_line_step=args.minimum_line_step,
        )
        nonlinear_state, nonlinear_solver = solve_nonlinear_r13(
            nonlinear_config,
            linear_state,
            callback=lambda item: print(json.dumps(item, sort_keys=True), flush=True),
        )
        nonlinear_metrics = nonlinear_cavity_metrics(
            nonlinear_state, nonlinear_config
        )
        nonlinear_path = args.output_dir / "nonlinear_state.npy"
        np.save(nonlinear_path, nonlinear_state)
        result["nonlinear"] = {
            "solver": nonlinear_solver,
            "metrics": nonlinear_metrics,
            "paper_comparison": _comparison(nonlinear_metrics, args.kn),
            "state": str(nonlinear_path),
            "state_sha256": _sha256(nonlinear_path),
        }
    selected = result["nonlinear"] if args.nonlinear else result["linear"]
    solver = selected["solver"]
    metrics = selected["metrics"]
    comparison = selected["paper_comparison"]
    gates = {
        "finite": bool(solver["finite"]),
        "mass": abs(float(solver["mass_perturbation"])) <= 1.0e-8,
        "rho_positive": float(
            solver["rho_min"] if "rho_min" in solver else metrics["rho_min"]
        ) > 0.0,
        "theta_positive": float(
            solver["theta_min"] if "theta_min" in solver else metrics["theta_min"]
        ) > 0.0,
        "algebraic_convergence": bool(
            solver.get("converged", True)
            and float(solver.get("relative_residual", solver.get("relative_linear_residual")))
            <= 1.0e-7
        ),
        "paper_diagnostic": bool(
            comparison.get("within_diagnostic_tolerance", False)
        ),
    }
    required_gates = [
        "finite",
        "mass",
        "rho_positive",
        "theta_positive",
        "algebraic_convergence",
    ]
    if getattr(args, "require_paper_comparison", False):
        required_gates.append("paper_diagnostic")
    result["gates"] = gates
    result["required_gates"] = required_gates
    result["passed"] = all(gates[name] for name in required_gates)
    result["publication_grade"] = False
    result["scientific_status"] = (
        "paper-equation verification solve; publication requires an independent "
        "N=75 nonlinear reproduction plus a grid study"
    )
    result["elapsed_seconds"] = time.monotonic() - started
    report_path = args.output_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--kn", type=float, default=0.01)
    parser.add_argument("--lid-velocity", type=float, default=0.2096)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--direct-linear", action="store_true")
    parser.add_argument("--nonlinear", action="store_true")
    parser.add_argument("--qmr-rtol", type=float, default=1.0e-10)
    parser.add_argument("--qmr-maxiter", type=int, default=20000)
    parser.add_argument("--nonlinear-rtol", type=float, default=1.0e-8)
    parser.add_argument("--nonlinear-update-tol", type=float, default=1.0e-7)
    parser.add_argument("--nonlinear-maxiter", type=int, default=20)
    parser.add_argument("--minimum-line-step", type=float, default=1.0 / 1024.0)
    parser.add_argument("--require-paper-comparison", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
