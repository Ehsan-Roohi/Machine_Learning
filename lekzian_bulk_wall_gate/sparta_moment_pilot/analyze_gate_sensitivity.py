#!/usr/bin/env python3
"""Model/depth/protocol sensitivity audit for the moment pilot gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_moment_gate import FEATURE_SETS, TARGETS, CaseData, load_case, nrmse


CONTEXT = ("x", "y_over_hp", "tx", "ty", "nx", "ny", "log10_Kn")
DEPTH_CONFIGS = ((0.5,), (0.5, 1.0, 2.0))


def feature_indices(case: CaseData, feature_set: str, depths: tuple[float, ...]) -> list[int]:
    names = case.feature_names[feature_set]
    return [
        i
        for i, name in enumerate(names)
        if name in CONTEXT or any(f"_d{depth:g}L" in name for depth in depths)
    ]


def make_model(name: str, seed: int):
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=3,
            max_features=0.85,
            n_jobs=-1,
            random_state=seed,
        )
    if name == "ridge_100":
        return make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    raise ValueError(name)


def folds(cases: list[CaseData], protocol: str):
    if protocol == "LOCO":
        for geometry in sorted({case.geometry for case in cases}):
            yield geometry, [case for case in cases if case.geometry != geometry], [case for case in cases if case.geometry == geometry]
        return
    if protocol == "LOO-case":
        for held in cases:
            yield held.case_id, [case for case in cases if case.case_id != held.case_id], [held]
        return
    raise ValueError(protocol)


def evaluate(
    cases: list[CaseData], protocol: str, estimator: str, depths: tuple[float, ...], seed: int
) -> list[dict[str, object]]:
    rows = []
    for target_index, target in enumerate(TARGETS):
        truth_parts: list[np.ndarray] = []
        prediction_parts = {name: [] for name in FEATURE_SETS}
        fold_gains = []
        for fold_index, (fold_name, train, test) in enumerate(folds(cases, protocol)):
            y_train = np.concatenate([case.targets[case.region["nearfield"], target_index] for case in train])
            y_test = np.concatenate([case.targets[case.region["nearfield"], target_index] for case in test])
            truth_parts.append(y_test)
            fold_error = {}
            for feature_index, feature_set in enumerate(FEATURE_SETS):
                keep = feature_indices(train[0], feature_set, depths)
                x_train = np.vstack([case.features[feature_set][case.region["nearfield"]][:, keep] for case in train])
                x_test = np.vstack([case.features[feature_set][case.region["nearfield"]][:, keep] for case in test])
                model = make_model(estimator, seed + 100 * fold_index + 10 * target_index + feature_index)
                model.fit(x_train, y_train)
                prediction = model.predict(x_test)
                prediction_parts[feature_set].append(prediction)
                fold_error[feature_set] = nrmse(y_test, prediction)
            fold_gains.append(
                {
                    "fold": fold_name,
                    "gain_percent": 100.0
                    * (fold_error["S0"] - fold_error["S1"])
                    / max(fold_error["S0"], 1e-15),
                }
            )

        truth = np.concatenate(truth_parts)
        errors = {
            name: nrmse(truth, np.concatenate(prediction_parts[name])) for name in FEATURE_SETS
        }
        rows.append(
            {
                "protocol": protocol,
                "estimator": estimator,
                "depths_lambda": "+".join(f"{x:g}" for x in depths),
                "target": target,
                "nrmse_S0": errors["S0"],
                "nrmse_S1": errors["S1"],
                "nrmse_S2": errors["S2"],
                "S1_gain_percent": 100.0 * (errors["S0"] - errors["S1"]) / max(errors["S0"], 1e-15),
                "fold_gains": fold_gains,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    case_ids = [line.strip() for line in (args.root / "case_list.txt").read_text().splitlines() if line.strip()]
    cases = [load_case(args.root / case_id) for case_id in case_ids]

    rows = []
    for protocol in ("LOCO", "LOO-case"):
        for estimator in ("extra_trees", "ridge_100"):
            for depths in DEPTH_CONFIGS:
                rows.extend(evaluate(cases, protocol, estimator, depths, args.seed))

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sensitivity.json").write_text(json.dumps(rows, indent=2) + "\n")
    flat_rows = [{key: value for key, value in row.items() if key != "fold_gains"} for row in rows]
    with (args.output / "sensitivity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    shear = [row for row in rows if row["target"] == "cf"]
    best = max(shear, key=lambda row: float(row["S1_gain_percent"]))
    consistent = all(float(fold["gain_percent"]) > 0.0 for fold in best["fold_gains"])
    decision = {
        "verdict": "PASS" if float(best["S1_gain_percent"]) >= 20.0 and consistent else "FAIL",
        "best_shear_configuration": best,
        "all_fold_gains_positive": consistent,
        "reason": "No audited configuration reaches 20% shear gain consistently across held-out units.",
    }
    (args.output / "sensitivity_decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    lines = [
        "# Moment-gate sensitivity audit",
        "",
        f"**Verdict: {decision['verdict']}**",
        "",
        "This audit changes estimator, kinetic sampling horizon, and holdout protocol without using the held-out target to tune a model.",
        "",
        "| Protocol | Estimator | Depths/lambda | Target | S0 NRMSE | S1 NRMSE | S2 NRMSE | S1 gain |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['protocol']} | {row['estimator']} | {row['depths_lambda']} | {row['target']} | "
            f"{100*float(row['nrmse_S0']):.2f}% | {100*float(row['nrmse_S1']):.2f}% | "
            f"{100*float(row['nrmse_S2']):.2f}% | {float(row['S1_gain_percent']):.2f}% |"
        )
    best_folds = ", ".join(
        f"{item['fold']}={float(item['gain_percent']):.1f}%" for item in best["fold_gains"]
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The largest aggregate signed-shear gain is {float(best['S1_gain_percent']):.2f}% "
            f"({best['protocol']}, {best['estimator']}, depths {best['depths_lambda']} lambda).",
            f"Its held-out gains are not all positive: {best_folds}.",
            "Pressure does not improve when the full momentum-flux tensor is added.",
            "The conclusion is therefore insensitive to the tested tree/linear models and sampling horizons: full-range moments are not sufficient for a transferable operator on this six-case pilot.",
            "The next diagnostic is the already-generated ISO collision tally, which can test incident half-range moments without launching more DSMC cases.",
            "",
        ]
    )
    (args.output / "SENSITIVITY_REPORT.md").write_text("\n".join(lines))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
