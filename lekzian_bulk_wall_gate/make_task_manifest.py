#!/usr/bin/env python3
"""Create deterministic Slurm-array tasks for both grouped validations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-table", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="101,202,303,404,505")
    args = parser.parse_args()

    frame = pd.read_csv(args.feature_table)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    loco = sorted(frame["case_id"].astype(str).unique())
    pairout = sorted(
        {
            f"Ma{float(ma):g}_Kn{float(kn):g}"
            for ma, kn in frame[["Ma", "Kn"]].drop_duplicates().itertuples(index=False, name=None)
        }
    )
    if len(loco) != 27 or len(pairout) != 9:
        raise RuntimeError(f"Expected 27 LOCO and 9 pair-out groups; found {len(loco)} and {len(pairout)}")

    lines = []
    for scheme, groups in [("loco", loco), ("pairout", pairout)]:
        for group in groups:
            for seed in seeds:
                lines.append(f"{scheme}|{group}|{seed}")
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] wrote {len(lines)} array tasks to {out}")


if __name__ == "__main__":
    main()
