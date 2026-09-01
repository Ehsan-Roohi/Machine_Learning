#!/usr/bin/env python3
"""Pack four SPARTA text blocks per case into aligned compressed NPZ files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_dump(path: Path, item: str) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    prefix = f"ITEM: {item} "
    header_idx = next((i for i, line in enumerate(lines) if line.startswith(prefix)), None)
    if header_idx is None:
        raise ValueError(f"{path}: missing {item} header")
    columns = lines[header_idx].split()[2:]
    data = np.loadtxt(lines[header_idx + 1 :], dtype=np.float64, ndmin=2)
    if data.shape[1] != len(columns):
        raise ValueError(f"{path}: {data.shape[1]} data columns but {len(columns)} labels")
    return columns, data


def pack_case(case_dir: Path) -> Path:
    metadata = json.loads((case_dir / "metadata.json").read_text())
    out = case_dir / "output"
    grid_files = sorted(out.glob("grid.*.dat"))
    wall_files = sorted(out.glob("wall.*.dat"))
    if len(grid_files) != 4 or len(wall_files) != 4:
        raise ValueError(f"{case_dir.name}: exactly four grid and wall blocks are required")

    grid_blocks: list[np.ndarray] = []
    wall_blocks: list[np.ndarray] = []
    grid_columns: list[str] | None = None
    wall_columns: list[str] | None = None
    for path in grid_files:
        columns, data = read_dump(path, "CELLS")
        order = np.lexsort((data[:, 1], data[:, 0]))  # stable key: id, split
        data = data[order]
        if grid_columns is None:
            grid_columns = columns
        elif columns != grid_columns:
            raise ValueError(f"{path}: grid schema changed between blocks")
        if grid_blocks and not np.allclose(data[:, :9], grid_blocks[0][:, :9], rtol=0.0, atol=1e-12):
            raise ValueError(f"{path}: grid id/split/geometry changed between blocks")
        grid_blocks.append(data)

    for path in wall_files:
        columns, data = read_dump(path, "SURFS")
        data = data[np.argsort(data[:, 0], kind="stable")]
        if wall_columns is None:
            wall_columns = columns
        elif columns != wall_columns:
            raise ValueError(f"{path}: wall schema changed between blocks")
        if wall_blocks and not np.allclose(data[:, :6], wall_blocks[0][:, :6], rtol=0.0, atol=1e-12):
            raise ValueError(f"{path}: wall id/geometry changed between blocks")
        wall_blocks.append(data)

    assert grid_columns is not None and wall_columns is not None
    expected_grid = metadata["column_schema"]["grid"]
    expected_wall = metadata["column_schema"]["wall"]
    if len(grid_columns) != len(expected_grid) or len(wall_columns) != len(expected_wall):
        raise ValueError(f"{case_dir.name}: metadata and dump column counts disagree")

    packed = out / "moment_blocks.npz"
    np.savez_compressed(
        packed,
        grid=np.stack(grid_blocks),
        wall=np.stack(wall_blocks),
        grid_columns=np.asarray(expected_grid),
        wall_columns=np.asarray(expected_wall),
        grid_steps=np.asarray([int(path.name.split(".")[1]) for path in grid_files]),
        wall_steps=np.asarray([int(path.name.split(".")[1]) for path in wall_files]),
        case_id=np.asarray(metadata["case_id"]),
    )
    return packed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text())
    packed_manifest = []
    for entry in manifest:
        packed = pack_case(args.root / entry["case_id"])
        packed_manifest.append({"case_id": entry["case_id"], "npz": str(packed.relative_to(args.root))})
        print(f"PACKED {entry['case_id']} -> {packed}")
    (args.root / "packed_manifest.json").write_text(json.dumps(packed_manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
