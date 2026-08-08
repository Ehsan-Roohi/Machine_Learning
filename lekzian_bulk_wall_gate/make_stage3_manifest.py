#!/usr/bin/env python3
"""Create the 108 deterministic multi-target Stage-3 Slurm tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="101,202,303")
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as data:
        case_id = np.asarray(data["case_id"]).astype(str)
        ma = np.asarray(data["Ma"], dtype=float)
        kn = np.asarray(data["Kn"], dtype=float)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    loco = sorted(set(case_id))
    pairout = sorted({f"Ma{value_ma:g}_Kn{value_kn:g}" for value_ma, value_kn in zip(ma, kn)})
    if len(loco) != 27 or len(pairout) != 9:
        raise RuntimeError(
            "Spatial Stage 3 requires 27 Phase-1 cases and 9 (Ma,Kn) groups; "
            f"found {len(loco)} and {len(pairout)}."
        )
    lines = [
        f"{scheme}|{group}|{seed}"
        for scheme, groups in (("loco", loco), ("pairout", pairout))
        for group in groups
        for seed in seeds
    ]
    expected = (27 + 9) * len(seeds)
    if len(lines) != expected:
        raise AssertionError("Stage-3 task accounting failed.")
    path = Path(args.out).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] wrote {len(lines)} Stage-3 tasks to {path}")


if __name__ == "__main__":
    main()
