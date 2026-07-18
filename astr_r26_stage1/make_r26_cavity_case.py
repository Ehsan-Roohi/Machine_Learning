#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import h5py
import numpy as np

BASE_MACH_U100 = 0.3099878865960759
BASE_RE_U100 = 10.031341886555307
GAMMA = 5.0 / 3.0


def fortran_double(value: float) -> str:
    """Return a real literal compatible with ASTR's legacy string parser."""
    return f"{value:.16e}".replace("e", "d")


def replace_value_after_comment(lines: list[str], prefix: str, value: str) -> None:
    for index, line in enumerate(lines):
        if line.strip().startswith(prefix):
            if index + 1 >= len(lines):
                raise RuntimeError(f"Missing value after {prefix}")
            lines[index + 1] = value
            return
    raise RuntimeError(f"Comment prefix not found: {prefix}")


def assert_no_e_exponents(text: str, path: Path) -> None:
    bad = re.findall(r"(?i)(?<![A-Za-z])[-+]?\d+(?:\.\d*)?[e][+-]?\d+", text)
    if bad:
        raise RuntimeError(f"E-notation is unsafe for ASTR in {path}: {bad}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an audited ASTR R26 lid-driven cavity case.")
    parser.add_argument("--template-case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--lid-speed-ms", type=float, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--maxstep", type=int, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    args = parser.parse_args()

    if args.cells < 8 or args.cells % 2 != 0:
        raise ValueError("cells must be even and at least 8 for the 2x2 MPI decomposition")
    if args.lid_speed_ms <= 0.0:
        raise ValueError("lid-speed-ms must be positive")
    if args.dt <= 0.0:
        raise ValueError("dt must be positive")
    if args.maxstep <= 0:
        raise ValueError("maxstep must be positive")
    if args.checkpoint_every <= 0 or args.checkpoint_every > args.maxstep:
        raise ValueError("checkpoint-every must be in [1,maxstep]")

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
    input_candidates = sorted(datin.glob("input.astr*"))
    if not input_candidates:
        raise FileNotFoundError("No input.astr* template found")
    input_path = input_candidates[0]
    controller_path = datin / "controller"

    speed_scale = args.lid_speed_ms / 100.0
    mach = BASE_MACH_U100 * speed_scale
    reynolds = BASE_RE_U100 * speed_scale
    computed_knudsen = math.sqrt(math.pi * GAMMA / 2.0) * mach / reynolds

    input_lines = input_path.read_text(encoding="utf-8").splitlines()
    replace_value_after_comment(input_lines, "# im,jm,km", f"{args.cells},{args.cells},0")
    replace_value_after_comment(input_lines, "# lrestar", "f")
    replace_value_after_comment(
        input_lines,
        "# ref_t,reynolds,mach,gamma",
        ", ".join(["300.0d0", fortran_double(reynolds), fortran_double(mach), fortran_double(GAMMA)]),
    )
    replace_value_after_comment(input_lines, "# turbmode,iomode, moment", "none,h,r26")
    replace_value_after_comment(input_lines, "# ninit", "0")
    input_text = "\n".join(input_lines) + "\n"
    assert_no_e_exponents(input_text, input_path)
    input_path.write_text(input_text, encoding="utf-8")

    controller_lines = controller_path.read_text(encoding="utf-8").splitlines()
    replace_value_after_comment(controller_lines, "# lwsequ,lwslic,lavg,lcracon", "       f,     f,   f,      f")
    replace_value_after_comment(
        controller_lines,
        "# maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg",
        f"{args.maxstep:9d}, {args.checkpoint_every:7d}, {args.checkpoint_every:7d},     500,     100,   100",
    )
    replace_value_after_comment(controller_lines, "# deltat", fortran_double(args.dt))
    controller_text = "\n".join(controller_lines) + "\n"
    assert_no_e_exponents(controller_text, controller_path)
    controller_path.write_text(controller_text, encoding="utf-8")

    coordinates = np.linspace(0.0, 1.0, args.cells + 1, dtype=np.float64)
    X, Y = np.meshgrid(coordinates, coordinates, indexing="xy")
    X = X[np.newaxis, :, :]
    Y = Y[np.newaxis, :, :]
    Z = np.zeros_like(X)
    with h5py.File(datin / "grid.h5", "w") as h:
        h.create_dataset("x", data=X)
        h.create_dataset("y", data=Y)
        h.create_dataset("z", data=Z)

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
        "lid_speed_ms": args.lid_speed_ms,
        "target_mach": mach,
        "target_reynolds": reynolds,
        "computed_knudsen": computed_knudsen,
        "wall_temperature_K": 300.0,
        "dt": args.dt,
        "maxstep": args.maxstep,
        "moment_model": "r26",
        "wall_start": "full lid speed from step 0 (Stage-1 source patch)",
        "number_format": "Fortran D exponent required by ASTR parser",
        "collision_coefficients": "Maxwell molecules, Stage-1 Apsi1 correction",
        "nonlinear_R26_blocks": "retained disabled",
    }
    if abs(computed_knudsen - 0.05) > 1.0e-12:
        raise RuntimeError(f"Generated Re/Ma do not preserve Kn=0.05: {computed_knudsen}")
    (output / "case_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
