#!/usr/bin/env python3
"""Validate SPARTA pilot completion and the expected four-block output schema."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


ITEM_RE = re.compile(r"^ITEM: ")


def read_last_snapshot(path: Path) -> tuple[list[str], list[list[float]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    atom_headers = [i for i, line in enumerate(lines) if line.startswith("ITEM: CELLS") or line.startswith("ITEM: SURFS")]
    if not atom_headers:
        raise ValueError(f"no SPARTA CELLS/SURFS header in {path}")
    start = atom_headers[-1]
    columns = lines[start].split()[2:]
    rows: list[list[float]] = []
    for line in lines[start + 1 :]:
        if ITEM_RE.match(line):
            break
        if line.strip():
            rows.append([float(token) for token in line.split()])
    return columns, rows


def validate_case(case_dir: Path) -> list[str]:
    meta = json.loads((case_dir / "metadata.json").read_text())
    out = case_dir / "output"
    first_step = (
        meta["settings"]["equil_steps"] // meta["settings"]["block_steps"] + 1
    ) * meta["settings"]["block_steps"]
    expected_steps = [
        first_step + meta["settings"]["block_steps"] * i
        for i in range(meta["settings"]["nblocks"])
    ]
    problems: list[str] = []
    log_path = case_dir / "log.sparta"
    if not log_path.exists() or "MOMENT_PILOT_COMPLETE" not in log_path.read_text(errors="replace"):
        problems.append("missing completion marker in log.sparta")

    grid_files = sorted(out.glob("grid.*.dat"))
    wall_files = sorted(out.glob("wall.*.dat"))
    if len(grid_files) != 4:
        problems.append(f"expected 4 grid blocks, found {len(grid_files)}")
    if len(wall_files) != 4:
        problems.append(f"expected 4 wall blocks, found {len(wall_files)}")
    for step in expected_steps:
        if not (out / f"grid.{step:08d}.dat").exists():
            problems.append(f"missing grid block at step {step}")
        if not (out / f"wall.{step:08d}.dat").exists():
            problems.append(f"missing wall block at step {step}")

    for path, schema_key in ((grid_files[-1] if grid_files else None, "grid"), (wall_files[-1] if wall_files else None, "wall")):
        if path is None:
            continue
        try:
            columns, rows = read_last_snapshot(path)
        except Exception as exc:  # noqa: BLE001
            problems.append(str(exc))
            continue
        expected = meta["column_schema"][schema_key]
        if len(columns) != len(expected):
            problems.append(f"{path.name}: expected {len(expected)} columns, found {len(columns)}")
        if not rows:
            problems.append(f"{path.name}: no data rows")
        elif any(len(row) != len(columns) or not all(math.isfinite(value) for value in row) for row in rows):
            problems.append(f"{path.name}: malformed or non-finite row")

    if meta["collision_tally"] and not list(out.glob("collisions.*.dat")):
        problems.append("collision tally was enabled but no collision file was written")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--case", help="Validate one case ID instead of the entire manifest")
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text())
    if args.case:
        manifest = [entry for entry in manifest if entry["case_id"] == args.case]
        if not manifest:
            raise SystemExit(f"case {args.case!r} is absent from the manifest")
    failed = False
    for entry in manifest:
        case_dir = args.root / entry["case_id"]
        problems = validate_case(case_dir)
        if problems:
            failed = True
            print(f"FAIL {entry['case_id']}")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print(f"PASS {entry['case_id']}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
