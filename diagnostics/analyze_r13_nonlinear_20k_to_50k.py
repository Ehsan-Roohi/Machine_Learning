#!/usr/bin/env python3
"""Validate the exact-20k-seed nonlinear-R13 continuation to pseudo-time 0.5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

from analyze_r13_nonlinear_diagnostic import (
    CONVERGENCE_FIELDS,
    anti_fourier_metrics,
    field2d,
    recursive_h5_report,
    rms_relative_percent,
    runtime_report,
)


def convergence_report(paths: list[Path], threshold_percent: float = 0.1) -> dict:
    expected = [48000, 49000, 50000]
    steps, snapshots = [], []
    for path in paths:
        with h5py.File(path, "r") as h5:
            steps.append(int(h5["nstep"][0]))
            snapshots.append({key: field2d(h5, key) for key in CONVERGENCE_FIELDS})
    if steps != expected:
        raise ValueError(f"checkpoint sequence is {steps}, expected {expected}")
    intervals = []
    for left, right, old, new in zip(steps[:-1], steps[1:], snapshots[:-1], snapshots[1:]):
        intervals.append({
            "from_step": left,
            "to_step": right,
            "relative_RMS_change_percent": {
                key: rms_relative_percent(new[key], old[key])
                for key in CONVERGENCE_FIELDS
            },
        })
    first = intervals[0]["relative_RMS_change_percent"]
    second = intervals[1]["relative_RMS_change_percent"]
    below = all(first[k] <= threshold_percent and second[k] <= threshold_percent
                for k in CONVERGENCE_FIELDS)
    non_growing = all(second[k] <= max(1.10 * first[k], 1.0e-6)
                      for k in CONVERGENCE_FIELDS)
    return {
        "criterion": (
            f"all {len(CONVERGENCE_FIELDS)} fields below {threshold_percent}% relative RMS "
            "on both 48k->49k and 49k->50k intervals, with the latter not more than "
            "10% above the former"
        ),
        "threshold_percent": threshold_percent,
        "steps": steps,
        "intervals": intervals,
        "all_intervals_below_threshold": below,
        "non_growing_last_interval": non_growing,
        "steady_converged_three_checkpoint_gate": bool(below and non_growing),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--snapshots-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    current, backup = args.case / "outdat", args.case / "bakup"
    h5_reports = {
        "current_flow": recursive_h5_report(current / "flowfield.h5"),
        "current_auxiliary": recursive_h5_report(current / "auxiliary.h5"),
        "backup_flow": recursive_h5_report(backup / "flowfield.h5"),
        "backup_auxiliary": recursive_h5_report(backup / "auxiliary.h5"),
    }
    required_h5_clean = all(
        report["exists"] and report.get("recursive_finite", False)
        for report in h5_reports.values()
    )
    thermo_clean = all(
        h5_reports[key].get("positive_rho_p_t", False)
        for key in ("current_flow", "backup_flow")
    )
    paths = [args.snapshots_dir / f"flowfield_{step}.h5"
             for step in (48000, 49000, 50000)]
    convergence = convergence_report(paths)
    af = anti_fourier_metrics(current / "flowfield.h5", args.case / "datin/grid.h5")
    runtime = runtime_report([args.case / "logs/astr.log", args.case / "logs/astr_time.log"])
    status = json.loads((args.case / "analysis/run_status.json").read_text())
    metadata = json.loads((args.case / "rana2013_case_metadata.json").read_text())
    seed = json.loads(Path("diagnostics/seed20k_validation.json").read_text())
    execution_clean = (
        seed.get("clean_stability_restart_checkpoint") is True
        and status.get("status") == "completed"
        and int(status.get("return_code", -1)) == 0
        and int(status.get("last_checkpoint_step", -1)) == 50000
        and abs(float(status.get("simulation_time", -1.0)) - 0.5) < 1.0e-10
        and h5_reports["current_flow"].get("nstep") == 50000
        and h5_reports["backup_flow"].get("nstep") == 49000
        and required_h5_clean and thermo_clean and runtime["marker_count"] == 0
        and metadata.get("continuation_from_step") == 20000
        and metadata.get("seed_artifact_id") == 8443346726
    )
    report = {
        "case_label": "nonlinear Rana-R13 exact-20k restart to matched pseudo-time 0.5; diagnostic only",
        "seed_provenance": {
            "run": 29689596458,
            "artifact_id": 8443346726,
            "artifact_digest": "sha256:2e3dbc26aa6277203d4021582269e685a01e6669f301255bd2ef5e5948c2f5f9",
            "seed_step": 20000,
            "seed_time": 0.20000000000005924,
        },
        "case_metadata": metadata,
        "run_status": status,
        "hdf5": h5_reports,
        "runtime": runtime,
        "three_checkpoint_convergence": convergence,
        "anti_fourier_metrics": af,
        "clean_stability_restart_checkpoint": bool(execution_clean),
        "scientific_status": (
            "three-checkpoint steady criterion passed at time 0.5; dt/2 and grid checks still required"
            if execution_clean and convergence["steady_converged_three_checkpoint_gate"]
            else "matched-time stability diagnostic only; three-checkpoint steady criterion not passed"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not execution_clean:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
