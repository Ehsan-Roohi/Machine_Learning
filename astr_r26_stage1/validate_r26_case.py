#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

FIELDS = [
    "ro", "p", "t", "u1", "u2", "u3",
    "sigmaxx", "sigmaxy", "sigmaxz", "sigmayy", "sigmayz", "sigmazz",
    "qx", "qy", "qz",
    "mxxx", "mxxy", "mxxz", "mxyy", "mxyz", "myyy", "myyz",
    "Rxx", "Rxy", "Rxz", "Ryy", "Ryz", "Delta",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an ASTR R26 HDF5 result.")
    parser.add_argument("case", type=Path)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--mode", choices=["equilibrium", "moving"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    flow = args.case / "outdat" / "flowfield.h5"
    if not flow.is_file() or flow.stat().st_size == 0:
        raise FileNotFoundError(flow)

    result: dict[str, object] = {
        "case": str(args.case),
        "target_step": args.target_step,
        "mode": args.mode,
        "fields": {},
    }

    with h5py.File(flow, "r") as h:
        nstep = int(h["nstep"][0])
        time = float(h["time"][0])
        if nstep < args.target_step:
            raise RuntimeError(f"nstep={nstep} < target={args.target_step}")
        result["nstep"] = nstep
        result["time"] = time

        arrays: dict[str, np.ndarray] = {}
        for key in FIELDS:
            if key not in h:
                raise KeyError(f"Missing R26 output field: {key}")
            array = np.asarray(h[key], dtype=float)
            if not np.isfinite(array).all():
                raise FloatingPointError(f"Non-finite values in {key}")
            arrays[key] = array
            result["fields"][key] = {
                "min": float(array.min()),
                "max": float(array.max()),
                "mean": float(array.mean()),
                "rms": float(np.sqrt(np.mean(array * array))),
            }

    for positive_key in ["ro", "p", "t"]:
        if float(arrays[positive_key].min()) <= 0.0:
            raise FloatingPointError(f"Non-positive {positive_key}")

    density_mean = float(arrays["ro"].mean())
    max_speed = float(np.sqrt(arrays["u1"]**2 + arrays["u2"]**2 + arrays["u3"]**2).max())
    result["conservation_and_bounds"] = {
        "mean_density": density_mean,
        "mean_density_error_from_initial": density_mean - 1.0,
        "max_dimensionless_speed": max_speed,
        "temperature_min": float(arrays["t"].min()),
        "temperature_max": float(arrays["t"].max()),
    }

    if not (0.5 < density_mean < 1.5):
        raise RuntimeError(f"Unphysical mean density: {density_mean}")
    if arrays["t"].max() > 5.0 or arrays["t"].min() < 0.1:
        raise RuntimeError("Temperature left conservative Stage-1 bounds")
    if max_speed > 5.0:
        raise RuntimeError(f"Velocity blow-up detected: {max_speed}")

    if args.mode == "equilibrium":
        equilibrium = {
            "max_abs_velocity": max_speed,
            "max_abs_density_minus_one": float(np.max(np.abs(arrays["ro"] - 1.0))),
            "max_abs_temperature_minus_one": float(np.max(np.abs(arrays["t"] - 1.0))),
            "max_abs_stress": float(max(np.max(np.abs(arrays[k])) for k in ["sigmaxx", "sigmaxy", "sigmayy"])),
            "max_abs_heat_flux": float(max(np.max(np.abs(arrays[k])) for k in ["qx", "qy", "qz"])),
            "max_abs_higher_moment": float(max(np.max(np.abs(arrays[k])) for k in ["mxxx", "mxxy", "mxyy", "myyy", "Rxx", "Rxy", "Ryy", "Delta"])),
        }
        result["equilibrium_residual"] = equilibrium
        if equilibrium["max_abs_velocity"] > 1.0e-7:
            raise RuntimeError("Equilibrium velocity residual too large")
        if equilibrium["max_abs_density_minus_one"] > 1.0e-7:
            raise RuntimeError("Equilibrium density residual too large")
        if equilibrium["max_abs_temperature_minus_one"] > 1.0e-7:
            raise RuntimeError("Equilibrium temperature residual too large")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
