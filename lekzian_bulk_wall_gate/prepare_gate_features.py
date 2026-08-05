#!/usr/bin/env python3
"""Build the full fixed-width annular descriptor table from the legacy DSMC audit.

This deliberately reuses the original field reader, surface reader, geometry
logic, and descriptor extractor.  It extracts only the six-block ``full``
representation needed by ``gate_test.py``; finite-radius inputs are generated
later by masking outer annuli, so every candidate has identical dimensionality.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--nonlocal-script", required=True)
    parser.add_argument("--field-base-script", required=True)
    parser.add_argument("--surface-base-script", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-gas-points", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    nonlocal_base = import_module(Path(args.nonlocal_script).expanduser().resolve(), "lekzian_nonlocal_base")
    field_base = import_module(Path(args.field_base_script).expanduser().resolve(), "lekzian_field_base")
    surface_base = import_module(Path(args.surface_base_script).expanduser().resolve(), "lekzian_surface_base")

    manifest = field_base.load_manifest(Path(args.audit_dir).expanduser().resolve())
    manifest = nonlocal_base.get_phase1_cases(manifest)
    if manifest.empty:
        raise RuntimeError("No Phase-1 cases found in the selected audit directory.")
    print(f"[INFO] selected {len(manifest)} Phase-1 physical cases", flush=True)

    rng = np.random.default_rng(args.seed)
    legacy_args = SimpleNamespace(max_gas_points=args.max_gas_points)
    tables = []
    case_rows = []
    for _, row in manifest.iterrows():
        case = nonlocal_base.read_case_data(field_base, surface_base, row, legacy_args, rng)
        if case is None:
            continue
        table = nonlocal_base.make_features_for_case(case, [("full", None)])["full"]
        tables.append(table)
        case_rows.append(
            {
                "case_id": case.case_id,
                "Ma": case.Ma,
                "Kn": case.Kn,
                "geom": case.geom,
                "hphs": case.hphs,
                "TwTinf": case.TwTinf,
                "hs": case.hs,
                "hp": case.hp,
                "n_gas": len(case.gas_xy),
                "n_surface": len(case.surface),
            }
        )
        print(f"[INFO] {case.case_id}: gas={len(case.gas_xy)} surface={len(case.surface)}", flush=True)

    if not tables:
        raise RuntimeError("No feature tables were produced.")
    full = pd.concat(tables, ignore_index=True)
    feature_path = out_dir / "surface_patch_dataset_full.csv"
    full.to_csv(feature_path, index=False)
    pd.DataFrame(case_rows).to_csv(out_dir / "case_table_phase1.csv", index=False)
    record = vars(args).copy()
    record.update({"n_cases": len(case_rows), "n_surface_samples": len(full), "feature_table": str(feature_path)})
    (out_dir / "feature_extraction_config.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"[DONE] feature table: {feature_path}", flush=True)


if __name__ == "__main__":
    main()
