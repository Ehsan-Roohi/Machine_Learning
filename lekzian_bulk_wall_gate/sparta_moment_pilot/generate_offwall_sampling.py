#!/usr/bin/env python3
"""Generate restart continuations for finite-distance incoming sampling."""

from __future__ import annotations

import argparse
import json
import re
import zlib
from pathlib import Path


def read_protrusion_vertices(path: Path) -> list[tuple[float, float]]:
    lines = [line.strip() for line in path.read_text().splitlines()]
    points: dict[int, tuple[float, float]] = {}
    line_rows: list[tuple[int, int, int]] = []
    section = None
    for line in lines:
        if line == "Points":
            section = "points"; continue
        if line == "Lines":
            section = "lines"; continue
        fields = line.split()
        if section == "points" and len(fields) == 3:
            try: points[int(fields[0])] = (float(fields[1]), float(fields[2]))
            except ValueError: pass
        elif section == "lines" and len(fields) >= 4:
            try:
                _, surf_type, p1, p2 = map(int, fields[:4])
                if surf_type == 2: line_rows.append((surf_type, p1, p2))
            except ValueError: pass
    used = sorted({p for _, p1, p2 in line_rows for p in (p1, p2)})
    if not used:
        raise ValueError(f"no type-2 protrusion elements in {path}")
    return [points[p] for p in used]


def render(metadata: dict, bbox: tuple[float, float, float, float], steps: int,
           block_steps: int, dump_every: int, label: str) -> str:
    case_id = metadata["case_id"]
    temperature = float(metadata["wall_temperature_K"])
    seed = 10000 + zlib.crc32((case_id + label).encode()) % 800000
    xlo, xhi, ylo, yhi = bbox
    return f"""# Finite-distance incoming sampling; no collision event is an input.
# Case={case_id}; particle snapshots are restricted to an off-wall bounding box.
seed {seed}
read_restart output/final.restart

surf_collide diffuseWallOff diffuse {temperature:.10g} 1.0
surf_modify all collide diffuseWallOff
collide vss gas ar.vss
region bottomIn block -0.1 0.0 INF INF INF INF
fix inflowMain emit/face gas xlo yhi
fix inflowBottom emit/face gas ylo region bottomIn

region offwallBox block {xlo:.12g} {xhi:.12g} {ylo:.12g} {yhi:.12g} INF INF
reset_timestep 0
stats {block_steps}
stats_style step cpu np nattempt ncoll nscoll nscheck

# Raw molecular states are sampled before any association with a wall element.
dump dOff particle gas {dump_every} output/{label}/particles.gz id type cellID x y vx vy vz
dump_modify dOff region offwallBox format float %.16g flush yes

# Targets are kept in a separate file and are never read by the feature reducer.
compute cWallOff surf protrusion gas n press shx shy shz ke erot evib etot
fix fWallOff ave/surf protrusion 1 {block_steps} {block_steps} c_cWallOff[*]
dump dWallOff surf protrusion {block_steps} output/{label}/wall_targets.*.dat &
  id v1x v1y v2x v2y area f_fWallOff[*]
dump_modify dWallOff pad 8

run {steps}
undump dOff
print "OFFWALL_SAMPLING_COMPLETE case={case_id} steps={steps} dump_every={dump_every}"
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_root", type=Path)
    p.add_argument("--case-ids", nargs="+", required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--block-steps", type=int, default=1000)
    p.add_argument("--dump-every", type=int, default=50)
    p.add_argument("--max-depth-lambda", type=float, default=2.0)
    p.add_argument("--label", default="offwall_half_range")
    args = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label): raise ValueError("invalid label")
    if args.steps <= 0 or args.steps % args.block_steps: raise ValueError("invalid block partition")
    if args.block_steps % args.dump_every: raise ValueError("dump frequency must divide block_steps")
    manifest = []
    for case_id in args.case_ids:
        case = args.run_root / case_id
        metadata = json.loads((case / "metadata.json").read_text())
        if not (case / "output/final.restart").is_file(): raise FileNotFoundError(case / "output/final.restart")
        output = case / "output" / args.label
        if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
        output.mkdir(parents=True)
        vertices = read_protrusion_vertices(case / "wall.surf")
        lam = float(metadata["physics"]["mean_free_path_m"])
        margin = (args.max_depth_lambda + 0.75) * lam
        xs, ys = zip(*vertices)
        bbox = (min(xs)-margin, max(xs)+margin, min(ys)-margin, max(ys)+margin)
        input_path = case / f"in.{args.label}"
        input_path.write_text(render(metadata, bbox, args.steps, args.block_steps,
                                     args.dump_every, args.label))
        manifest.append({"case_id": case_id, "bbox": bbox, "steps": args.steps,
                         "block_steps": args.block_steps, "dump_every": args.dump_every,
                         "depths_lambda": [0.25, 0.5, 1.0, 2.0]})
        print(input_path)
    (args.run_root / f"{args.label}_case_list.txt").write_text("\n".join(args.case_ids)+"\n")
    (args.run_root / f"{args.label}_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__": main()
