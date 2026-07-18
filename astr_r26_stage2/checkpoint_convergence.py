#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

FIELDS = ["ro", "p", "t", "u1", "u2", "qx", "qy", "Rxx", "Rxy", "Ryy", "Delta"]
MACRO = ["ro", "p", "t"]
VELOCITY = ["u1", "u2"]
HIGHER = ["qx", "qy", "Rxx", "Rxy", "Ryy", "Delta"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two ASTR R26 checkpoints.")
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--macro-threshold", type=float, default=1.0)
    parser.add_argument("--velocity-threshold", type=float, default=2.0)
    parser.add_argument("--higher-threshold", type=float, default=5.0)
    args = parser.parse_args()

    report: dict[str, object] = {"relative_RMS_change_percent": {}}
    with h5py.File(args.previous, "r") as old, h5py.File(args.current, "r") as new:
        previous_step = int(old["nstep"][0])
        current_step = int(new["nstep"][0])
        if current_step <= previous_step:
            raise RuntimeError("Checkpoint order is invalid")
        report["previous_step"] = previous_step
        report["current_step"] = current_step
        for key in FIELDS:
            a = np.asarray(old[key], dtype=float)
            b = np.asarray(new[key], dtype=float)
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise FloatingPointError(f"Non-finite checkpoint data in {key}")
            value = float(
                100.0 * np.sqrt(np.mean((b - a) ** 2))
                / (np.sqrt(np.mean(b * b)) + 1.0e-30)
            )
            report["relative_RMS_change_percent"][key] = value

    changes = report["relative_RMS_change_percent"]
    max_macro = max(float(changes[key]) for key in MACRO)
    max_velocity = max(float(changes[key]) for key in VELOCITY)
    max_higher = max(float(changes[key]) for key in HIGHER)
    passed = (
        max_macro <= args.macro_threshold
        and max_velocity <= args.velocity_threshold
        and max_higher <= args.higher_threshold
    )
    report["criteria"] = {
        "max_macro_percent": max_macro,
        "max_velocity_percent": max_velocity,
        "max_higher_percent": max_higher,
        "macro_threshold_percent": args.macro_threshold,
        "velocity_threshold_percent": args.velocity_threshold,
        "higher_threshold_percent": args.higher_threshold,
        "passed": passed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
