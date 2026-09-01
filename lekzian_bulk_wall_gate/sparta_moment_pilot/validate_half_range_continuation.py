#!/usr/bin/env python3
"""Validate every-timestep half-range continuation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_half_range_gate import read_tally_snapshots


def read_wall_dump(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("ITEM: SURFS "))
    return np.loadtxt(lines[header + 1 :], ndmin=2)


def validate_case(run_root: Path, case_id: str, expected_steps: int,
                  expected_blocks: int, label: str = "half_range_long") -> dict[str, object]:
    case_dir = run_root / case_id
    metadata = json.loads((case_dir / "metadata.json").read_text())
    output = case_dir / "output" / label
    collision = output / "collisions.dat"
    if not collision.is_file() or collision.stat().st_size == 0:
        raise ValueError(f"{case_id}: missing collisions.dat")
    ncolumns = len(metadata["column_schema"]["collision"])
    snapshots = read_tally_snapshots(collision, ncolumns)
    # Dump frequency 1 writes an initial snapshot at the current reset
    # timestep.  It precedes particle motion and is empty, so remove only that
    # unambiguous leading snapshot and retain all evolved timesteps.
    if snapshots and len(snapshots[0].values) == 0:
        snapshots = snapshots[1:]
    evolved = snapshots
    if len(evolved) != expected_steps:
        raise ValueError(f"{case_id}: {len(evolved)} evolved snapshots, expected {expected_steps}")
    timesteps = np.asarray([snapshot.timestep for snapshot in evolved])
    if len(timesteps) > 1 and not np.all(np.diff(timesteps) == 1):
        raise ValueError(f"{case_id}: collision timesteps are not contiguous")
    records = sum(len(snapshot.values) for snapshot in evolved)
    if records < 10000:
        raise ValueError(f"{case_id}: only {records} collision records")

    wall_files = sorted(output.glob("wall.*.dat"))
    if len(wall_files) != expected_blocks:
        raise ValueError(f"{case_id}: {len(wall_files)} wall blocks, expected {expected_blocks}")
    for path in wall_files:
        values = read_wall_dump(path)
        if values.shape != (60, 15) or not np.isfinite(values).all():
            raise ValueError(f"{case_id}: invalid wall dump {path.name} shape={values.shape}")
    return {
        "case_id": case_id,
            "collision_snapshots": len(evolved),
            "first_timestep": int(timesteps[0]),
            "last_timestep": int(timesteps[-1]),
        "collision_records": records,
        "wall_blocks": len(wall_files),
        "collision_bytes": collision.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--label", default="half_range_long")
    args = parser.parse_args()
    manifest = json.loads((args.run_root / f"{args.label}_manifest.json").read_text())
    selected = [item for item in manifest if args.case is None or item["case_id"] == args.case]
    if not selected:
        raise ValueError(f"unknown case {args.case}")
    summaries = [
        validate_case(
            args.run_root,
            item["case_id"],
            int(item["sampled_steps"]),
            int(item["sampled_steps"]) // int(item["block_steps"]),
            args.label,
        )
        for item in selected
    ]
    for summary in summaries:
        print(
            f"PASS {summary['case_id']} snapshots={summary['collision_snapshots']} "
            f"records={summary['collision_records']} wall_blocks={summary['wall_blocks']}"
        )
    if args.case is None:
        (args.run_root / f"{args.label}_validation.json").write_text(
            json.dumps(summaries, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
