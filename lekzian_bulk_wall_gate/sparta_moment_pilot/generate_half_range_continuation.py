#!/usr/bin/env python3
"""Generate every-timestep half-range continuation inputs from final restarts."""

from __future__ import annotations

import argparse
import json
import re
import zlib
from pathlib import Path


CASE_IDS = ("ISO_Ma6_Kn0p1", "ISO_Ma6_Kn0p8")


def validated_label(label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("label must contain only letters, digits, underscores, and hyphens")
    return label


def render_input(metadata: dict[str, object], steps: int, block_steps: int,
                 label: str = "half_range_long") -> str:
    case_id = str(metadata["case_id"])
    temperature = float(metadata["wall_temperature_K"])
    seed = 10_000 + zlib.crc32(case_id.encode("utf-8")) % 800_000
    label = validated_label(label)
    return f"""# Every-timestep incident half-range continuation
# Case={case_id}; start from the statistically steady final particle restart.

seed {seed}
read_restart output/final.restart

# Computes, fixes, gas/surface collision models, inflow regions, and output
# commands are not stored in a SPARTA restart and must be re-specified.
surf_collide diffuseWallLong diffuse {temperature:.10g} 1.0
surf_modify all collide diffuseWallLong
collide vss gas ar.vss

region bottomIn block -0.1 0.0 INF INF INF INF
fix inflowMain emit/face gas xlo yhi
fix inflowBottom emit/face gas ylo region bottomIn

reset_timestep 1
stats {block_steps}
stats_style step cpu np nattempt ncoll nscoll nscheck

compute cWallLong surf protrusion gas n press shx shy shz ke erot evib etot
fix fWallLong ave/surf protrusion 1 {block_steps} {block_steps} c_cWallLong[*]
dump dWallLong surf protrusion {block_steps} output/{label}/wall.*.dat &
  id v1x v1y v2x v2y area f_fWallLong[*]
dump_modify dWallLong pad 8

# The tally compute is instantaneous, so dump it every timestep.  Omitting '*'
# appends all snapshots to one file instead of creating thousands of files.
compute cCollisionLong surf/collision/tally protrusion gas id/surf id type time xc yc zc &
  vx/pre vy/pre vz/pre vx/post vy/post vz/post
dump dCollisionLong tally all 1 output/{label}/collisions.dat c_cCollisionLong[*]

run {steps}

print "HALF_RANGE_LONG_COMPLETE case={case_id} sampled_steps={steps}"
"""


def generate(
    run_root: Path, steps: int, block_steps: int,
    case_ids: tuple[str, ...] = CASE_IDS,
    label: str = "half_range_long",
) -> list[Path]:
    label = validated_label(label)
    if steps <= 0 or block_steps <= 0 or steps % block_steps:
        raise ValueError("steps must be positive and divisible by block_steps")
    # Preflight every target before creating any directory, so a partially
    # existing prior attempt cannot leave the other case half-generated.
    for case_id in case_ids:
        case_dir = run_root / case_id
        metadata_path = case_dir / "metadata.json"
        restart = case_dir / "output" / "final.restart"
        output = case_dir / "output" / label
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        if not restart.is_file():
            raise FileNotFoundError(restart)
        if output.exists():
            raise FileExistsError(
                f"{output} already exists; existing continuation data were not overwritten"
            )

    generated = []
    manifest = []
    for case_id in case_ids:
        case_dir = run_root / case_id
        metadata_path = case_dir / "metadata.json"
        restart = case_dir / "output" / "final.restart"
        output = case_dir / "output" / label
        output.mkdir(parents=True)
        metadata = json.loads(metadata_path.read_text())
        input_path = case_dir / f"in.{label}"
        input_path.write_text(render_input(metadata, steps, block_steps, label))
        generated.append(input_path)
        manifest.append(
            {
                "case_id": case_id,
                "input": input_path.name,
                "restart": str(restart.relative_to(case_dir)),
                "output": str(output.relative_to(case_dir)),
                "sampled_steps": steps,
                "block_steps": block_steps,
            }
        )
    (run_root / f"{label}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (run_root / f"{label}_case_list.txt").write_text("\n".join(case_ids) + "\n")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--block-steps", type=int, default=1000)
    parser.add_argument("--case-ids", nargs="+", default=list(CASE_IDS))
    parser.add_argument("--label", default="half_range_long")
    args = parser.parse_args()
    for path in generate(args.run_root, args.steps, args.block_steps,
                         tuple(args.case_ids), args.label):
        print(path)


if __name__ == "__main__":
    main()
