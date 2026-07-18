#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

BASE_MACH_U100 = 0.3099878865960759
BASE_RE_U100 = 10.031341886555307
GAMMA = 5.0 / 3.0


def replace_value_after_comment(lines: list[str], prefix: str, value: str) -> None:
    for index, line in enumerate(lines):
        if line.strip().startswith(prefix):
            if index + 1 >= len(lines):
                raise RuntimeError(f"Missing value after {prefix}")
            lines[index + 1] = value
            return
    raise RuntimeError(f"Comment prefix not found: {prefix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an ASTR R26 lid-driven cavity case.")
    parser.add_argument("--template-case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--lid-speed-ms", type=float, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--maxstep", type=int, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    parser.add_argument("--equilibrium", action="store_true")
    args = parser.parse_args()

    if args.cells < 8:
        raise ValueError("cells must be at least 8")
    if args.dt <= 0.0:
        raise ValueError("dt must be positive")
    if args.maxstep <= 0:
        raise ValueError("maxstep must be positive")

    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(args.template_case, output)

    for name in ["outdat", "bakup", "testout", "monitor", "islice", "jslice", "kslice"]:
        shutil.rmtree(output / name, ignore_errors=True)
    for path in output.glob("*.log"):
        path.unlink()
    for name in ["flowstate.dat", "errnode.log"]:
        path = output / name
        if path.exists():
            path.unlink()
    (output / "outdat").mkdir(parents=True, exist_ok=True)
    (output / "bakup").mkdir(parents=True, exist_ok=True)

    datin = output / "datin"
    input_candidates = list(datin.glob("input.astr*"))
    if not input_candidates:
        raise FileNotFoundError("No input.astr* template found")
    input_path = input_candidates[0]
    controller_path = datin / "controller"

    input_lines = input_path.read_text(encoding="utf-8").splitlines()
    replace_value_after_comment(input_lines, "# im,jm,km", f"{args.cells},{args.cells},0")
    replace_value_after_comment(input_lines, "# lrestar", "f")

    speed = 0.0 if args.equilibrium else args.lid_speed_ms
    mach = BASE_MACH_U100 * speed / 100.0
    reynolds = BASE_RE_U100 * speed / 100.0
    if args.equilibrium:
        scale = 1.0e-12
        mach = BASE_MACH_U100 * scale
        reynolds = BASE_RE_U100 * scale

    replace_value_after_comment(
        input_lines,
        "# ref_t,reynolds,mach,gamma",
        f"300.0d0, {reynolds:.16e}, {mach:.16e}, {GAMMA:.16e}",
    )
    replace_value_after_comment(
        input_lines,
        "# turbmode,iomode, moment",
        "none,h,r26",
    )
    replace_value_after_comment(input_lines, "# ninit", "0")
    input_path.write_text("\n".join(input_lines) + "\n", encoding="utf-8")

    controller_lines = controller_path.read_text(encoding="utf-8").splitlines()
    replace_value_after_comment(
        controller_lines,
        "# lwsequ,lwslic,lavg,lcracon",
        "       f,     f,   f,      f",
    )
    replace_value_after_comment(
        controller_lines,
        "# maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg",
        (
            f"{args.maxstep:9d}, {args.checkpoint_every:7d}, "
            f"{args.checkpoint_every:7d},    1000,    1000,  1000"
        ),
    )
    replace_value_after_comment(controller_lines, "# deltat", f"{args.dt:.16e}")
    controller_path.write_text("\n".join(controller_lines) + "\n", encoding="utf-8")

    coordinates = np.linspace(0.0, 1.0, args.cells + 1, dtype=np.float64)
    X, Y = np.meshgrid(coordinates, coordinates, indexing="xy")
    X = X[np.newaxis, :, :]
    Y = Y[np.newaxis, :, :]
    Z = np.zeros_like(X)
    with h5py.File(datin / "grid.h5", "w") as h:
        h.create_dataset("x", data=X)
        h.create_dataset("y", data=Y)
        h.create_dataset("z", data=Z)

    if args.cells % 2 != 0:
        raise ValueError("The Stage-1 workflow uses a 2x2 MPI decomposition; cells must be even")
    local = args.cells // 2
    parallel_lines = [
        "    isize     jsize     ksize",
        "        2         2         1",
        "     Rank       Irk       Jrk       Krk        IM        JM        KM        I0        J0        K0",
    ]
    rank = 0
    for jrank in range(2):
        for irank in range(2):
            parallel_lines.append(
                f"{rank:9d}{irank:10d}{jrank:10d}{0:10d}{local:10d}{local:10d}{0:10d}"
                f"{irank * local:10d}{jrank * local:10d}{0:10d}"
            )
            rank += 1
    (datin / "parallel.info").write_text("\n".join(parallel_lines) + "\n", encoding="utf-8")

    metadata = {
        "cells": args.cells,
        "lid_speed_ms": speed,
        "target_mach": mach,
        "target_reynolds": reynolds,
        "knudsen": 0.05,
        "wall_temperature_K": 300.0,
        "dt": args.dt,
        "maxstep": args.maxstep,
        "nominal_full_lid_step_from_existing_ramp": int(round(1.0 / args.dt)),
        "moment_model": "r26",
        "collision_coefficients": "Maxwell molecules, Stage-1 Apsi1 correction",
        "nonlinear_R26_blocks": "retained disabled",
    }
    (output / "case_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
