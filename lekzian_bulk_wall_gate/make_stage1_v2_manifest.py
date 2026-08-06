#!/usr/bin/env python3
"""Create deterministic target-specific Stage-1 V2 Slurm tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TARGETS = ("Cp", "Cq", "tau_abs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-table", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="101,202,303,404,505")
    args = parser.parse_args()

    frame = pd.read_csv(args.feature_table, usecols=["case_id", "Ma", "Kn"])
    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    loco = sorted(frame["case_id"].astype(str).unique())
    pairout = sorted(
        {
            f"Ma{float(ma):g}_Kn{float(kn):g}"
            for ma, kn in frame[["Ma", "Kn"]].drop_duplicates().itertuples(index=False, name=None)
        }
    )
    if len(loco) != 27 or len(pairout) != 9:
        raise RuntimeError(
            f"Stage-1 confirmatory design requires 27 Phase-1 cases and 9 (Ma,Kn) groups; "
            f"found {len(loco)} and {len(pairout)}."
        )
    lines = []
    for scheme, groups in (("loco", loco), ("pairout", pairout)):
        for group in groups:
            for seed in seeds:
                for target in TARGETS:
                    lines.append(f"{scheme}|{group}|{seed}|{target}")
    expected = (len(loco) + len(pairout)) * len(seeds) * len(TARGETS)
    if len(lines) != expected:
        raise AssertionError("Stage-1 task accounting failed.")
    path = Path(args.out).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] wrote {len(lines)} Stage-1 V2 tasks to {path}")


if __name__ == "__main__":
    main()
