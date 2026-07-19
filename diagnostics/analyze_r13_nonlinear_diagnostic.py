#!/usr/bin/env python3
"""Validate the matched nonlinear-R13 diagnostic and compute convergence/AF metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter


CONVERGENCE_FIELDS = (
    "ro", "p", "t", "u1", "u2", "qx", "qy", "sigmaxx", "sigmaxy",
    "sigmayy", "Rxx", "Rxy", "Ryy", "Delta",
)
AF_FIELDS = ("t", "qx", "qy", "Rxx", "Rxy", "Ryy", "Delta")
RUNTIME_MARKERS = (
    "IEEE_INVALID_FLAG", "SIGFPE", "Floating-point exception",
    "RANA_P_FAILURE", "COMPUTATION CRASHED", "UPDATEFVAR_DENSITY_GATE",
)


def scalar(h5: h5py.File, key: str) -> float:
    return float(np.asarray(h5[key]).reshape(-1)[0])


def field2d(h5: h5py.File, key: str) -> np.ndarray:
    value = np.asarray(h5[key], dtype=float)
    if value.ndim == 3:
        value = value[0]
    if value.ndim != 2:
        raise ValueError(f"{key}: expected 2-D physical plane, got {value.shape}")
    return value


def recursive_h5_report(path: Path) -> dict:
    report = {"path": str(path), "exists": path.is_file(), "bad_floating": [],
              "floating_datasets": 0, "nstep": None, "time": None,
              "positive_ranges": {}}
    if not path.is_file():
        return report
    with h5py.File(path, "r") as h5:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "fc":
                report["floating_datasets"] += 1
                data = np.asarray(obj[...])
                if not np.isfinite(data).all():
                    report["bad_floating"].append(name)
        h5.visititems(visit)
        if "nstep" in h5:
            report["nstep"] = int(scalar(h5, "nstep"))
        if "time" in h5:
            report["time"] = scalar(h5, "time")
        for key in ("ro", "p", "t"):
            if key in h5:
                data = np.asarray(h5[key], dtype=float)
                report["positive_ranges"][key] = {
                    "min": float(np.min(data)), "max": float(np.max(data)),
                    "positive": bool(np.min(data) > 0.0),
                }
    report["recursive_finite"] = not report["bad_floating"]
    report["positive_rho_p_t"] = bool(
        len(report["positive_ranges"]) == 3
        and all(v["positive"] for v in report["positive_ranges"].values())
    )
    return report


def rms_relative_percent(new: np.ndarray, old: np.ndarray) -> float:
    nr = float(np.sqrt(np.mean(np.square(new))))
    no = float(np.sqrt(np.mean(np.square(old))))
    if max(nr, no) < 1.0e-14:
        return 0.0
    return float(100.0 * np.sqrt(np.mean(np.square(new - old))) /
                 (0.5 * (nr + no) + 1.0e-300))


def convergence_report(paths: list[Path], threshold_percent: float = 0.1) -> dict:
    steps = []
    snapshots = []
    for path in paths:
        with h5py.File(path, "r") as h5:
            steps.append(int(scalar(h5, "nstep")))
            snapshots.append({key: field2d(h5, key) for key in CONVERGENCE_FIELDS})
    if steps != [18000, 19000, 20000]:
        raise ValueError(f"checkpoint sequence is {steps}, expected [18000, 19000, 20000]")
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
    sustained = all(second[k] <= max(1.10 * first[k], 1.0e-6)
                    for k in CONVERGENCE_FIELDS)
    return {
        "criterion": (
            f"all {len(CONVERGENCE_FIELDS)} fields below {threshold_percent}% relative RMS "
            "on both 18k->19k and 19k->20k intervals, with the latter not more than "
            "10% above the former"
        ),
        "threshold_percent": threshold_percent,
        "steps": steps,
        "intervals": intervals,
        "all_intervals_below_threshold": below,
        "non_growing_last_interval": sustained,
        "steady_converged_three_checkpoint_gate": bool(below and sustained),
    }


def interpolate(fields: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray,
                n: int = 160) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    xc = (np.arange(n) + 0.5) / n
    yc = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(xc, yc)
    points = np.column_stack((yy.ravel(), xx.ravel()))
    result = {}
    for key, values in fields.items():
        fn = RegularGridInterpolator((y, x), values, bounds_error=False, fill_value=None)
        result[key] = fn(points).reshape(n, n)
    return result, xc, yc


def anti_fourier_metrics(flow: Path, grid: Path, threshold: float = 0.05,
                         smooth_size: int = 7) -> dict:
    with h5py.File(flow, "r") as h5:
        fields = {key: field2d(h5, key) for key in AF_FIELDS}
    with h5py.File(grid, "r") as h5:
        xx = np.asarray(h5["x"], dtype=float)
        yy = np.asarray(h5["y"], dtype=float)
        if xx.ndim == 3:
            xx = xx[0]
            yy = yy[0]
        x = xx[0, :]
        y = yy[:, 0]
    interp, xc, yc = interpolate(fields, x, y)
    smooth = lambda a: uniform_filter(a, size=smooth_size, mode="nearest")
    temp = smooth(interp["t"])
    qx, qy = smooth(interp["qx"]), smooth(interp["qy"])
    dtdy, dtdx = np.gradient(temp, yc, xc, edge_order=2)
    qmag, gmag = np.hypot(qx, qy), np.hypot(dtdx, dtdy)
    denom = qmag * gmag
    iaf = np.full_like(temp, np.nan)
    valid = np.isfinite(denom) & (denom > 1.0e-14)
    iaf[valid] = (qx[valid] * dtdx[valid] + qy[valid] * dtdy[valid]) / denom[valid]
    active = (valid & (qmag >= threshold * np.nanmax(qmag))
              & (gmag >= threshold * np.nanmax(gmag)))
    af = active & (iaf > 0.0)
    if not np.any(af):
        raise ValueError("anti-Fourier set is empty")

    rxx, rxy, ryy = (smooth(interp[k]) for k in ("Rxx", "Rxy", "Ryy"))
    delta = smooth(interp["Delta"])
    drxxdy, drxxdx = np.gradient(rxx, yc, xc, edge_order=2)
    drxydy, drxydx = np.gradient(rxy, yc, xc, edge_order=2)
    dryydy, _ = np.gradient(ryy, yc, xc, edge_order=2)
    dddy, dddx = np.gradient(delta, yc, xc, edge_order=2)
    divrx, divry = drxxdx + drxydy, drxydx + dryydy
    pr = np.full_like(temp, np.nan)
    pd = np.full_like(temp, np.nan)
    qvalid = qmag > 1.0e-14
    pr[qvalid] = (qx[qvalid] * divrx[qvalid] + qy[qvalid] * divry[qvalid]) / qmag[qvalid]
    pd[qvalid] = (qx[qvalid] * dddx[qvalid] + qy[qvalid] * dddy[qvalid]) / (3.0 * qmag[qvalid])
    mask = af & np.isfinite(pr) & np.isfinite(pd)
    rms_pr = float(np.sqrt(np.mean(pr[mask] ** 2)))
    rms_pd = float(np.sqrt(np.mean(pd[mask] ** 2)))
    chi = np.abs(pd[mask]) / (np.abs(pr[mask]) + np.abs(pd[mask]) + 1.0e-300)
    return {
        "definition": "160x160 interpolation; one 7-point uniform smoothing pass; eta=0.05 active gate",
        "f_AF_active": float(af.sum() / active.sum()),
        "mean_IAF_AF": float(np.mean(iaf[af])),
        "PDelta_over_PR": float(rms_pd / rms_pr),
        "mean_chiDelta": float(np.mean(chi)),
        "active_cells": int(active.sum()),
        "anti_fourier_cells": int(af.sum()),
    }


def runtime_report(paths: list[Path]) -> dict:
    lines = []
    for path in paths:
        if path.exists():
            lines.extend(path.read_text(errors="replace").splitlines())
    matches = []
    for line_no, line in enumerate(lines, 1):
        if any(marker in line for marker in RUNTIME_MARKERS):
            matches.append({"line": line_no, "text": line[:500]})
        elif re.search(r"(^|[^A-Za-z])(NaN|Inf)([^A-Za-z]|$)", line):
            matches.append({"line": line_no, "text": line[:500]})
    return {"configured_markers": list(RUNTIME_MARKERS) + ["NaN/Inf token"],
            "marker_lines": matches, "marker_count": len(matches)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--snapshots-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    current = args.case / "outdat"
    backup = args.case / "bakup"
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

    snapshots = [args.snapshots_dir / f"flowfield_{step}.h5"
                 for step in (18000, 19000, 20000)]
    convergence = convergence_report(snapshots)
    af = anti_fourier_metrics(current / "flowfield.h5", args.case / "datin/grid.h5")
    runtime = runtime_report([args.case / "logs/astr.log", args.case / "logs/astr_time.log"])
    status = json.loads((args.case / "analysis/run_status.json").read_text())
    execution_clean = (
        status.get("status") == "completed"
        and int(status.get("return_code", -1)) == 0
        and int(status.get("last_checkpoint_step", -1)) == 20000
        and h5_reports["current_flow"].get("nstep") == 20000
        and h5_reports["backup_flow"].get("nstep") == 19000
        and required_h5_clean and thermo_clean and runtime["marker_count"] == 0
    )
    report = {
        "case_label": "fresh nonlinear Rana-R13 matched diagnostic; not publication-grade",
        "run_status": status,
        "hdf5": h5_reports,
        "runtime": runtime,
        "three_checkpoint_convergence": convergence,
        "anti_fourier_metrics": af,
        "clean_stability_restart_checkpoint": bool(execution_clean),
        "scientific_status": (
            "three-checkpoint steady criterion passed; further matched dt/2 and grid checks remain"
            if execution_clean and convergence["steady_converged_three_checkpoint_gate"]
            else "stable diagnostic only; three-checkpoint steady-convergence criterion not passed"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not execution_clean:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
