#!/usr/bin/env python3
"""Train the finite-distance bulk-to-wall surrogate with target-balanced fits.

Version 1 used one multi-output ExtraTrees fit for Cp and Cf.  The impurity
reduction then mixed targets with very different numerical variances, so Cp
dominated the split decisions.  This version fits Cp and Cf independently.
No DSMC targets are added to the inputs and the held-out Kn cases remain
unseen during fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from analyze_moment_gate import load_case, nrmse

SEEDS = (11, 29, 47, 71, 101)
FEATURES = ("S0", "S1", "S2", "Soff")
TARGETS = ("Cp", "Cf")


def case_ids(root: Path) -> list[str]:
    return [
        x.strip()
        for x in (root / "case_list.txt").read_text().splitlines()
        if x.strip() and (x.startswith("BWD_") or x.startswith("FWD_"))
    ]


def load(root: Path, case_id: str, label: str):
    case = load_case(root / case_id)
    mask = case.region["protrusion"]
    sid = case.surface_id[mask]
    path = root / case_id / "output" / label / "offwall_incoming_moments.npz"
    with np.load(path, allow_pickle=False) as z:
        order = {int(x): i for i, x in enumerate(z["surface_id"])}
        idx = np.array([order[int(x)] for x in sid])
        # Average the five independent DSMC blocks, retain all four distances,
        # then concatenate the ten incoming descriptors at each distance.
        off = z["features"].mean(axis=0)[:, idx, :]
        depths = np.asarray(z["depths_lambda"], dtype=float)
    off = np.moveaxis(off, 0, 1).reshape(len(sid), -1)
    if not np.isfinite(off).all():
        raise ValueError(f"non-finite off-wall features in {case_id}")
    return case, mask, {
        "S0": case.features["S0"][mask],
        "S1": case.features["S1"][mask],
        "S2": case.features["S2"][mask],
        "Soff": np.column_stack((case.features["S2"][mask], off)),
    }, depths


def fit_one_target(x: np.ndarray, y: np.ndarray, seed: int):
    model = ExtraTreesRegressor(
        n_estimators=800,
        min_samples_leaf=2,
        max_features=0.85,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(x, y)
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-root", type=Path, required=True)
    p.add_argument("--test-root", type=Path, required=True)
    p.add_argument("--label", default="offwall_v1")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    train = [load(a.train_root, c, a.label) for c in case_ids(a.train_root)]
    test = [load(a.test_root, c, a.label) for c in case_ids(a.test_root)]
    depth_ref = train[0][3]
    if any(not np.array_equal(depth_ref, item[3]) for item in train + test):
        raise ValueError("off-wall depth grids are inconsistent")

    rows: list[dict[str, object]] = []
    saved: dict[tuple[str, str], list[np.ndarray]] = {}
    for seed in SEEDS:
        for feature in FEATURES:
            x_train = np.vstack([data[feature] for _, _, data, _ in train])
            y_train = np.vstack([case.targets[mask] for case, mask, _, _ in train])

            # Independent target fits are essential: the numerical variance of
            # Cp must not determine the tree partitions used to predict Cf.
            models = [fit_one_target(x_train, y_train[:, j], seed) for j in range(2)]
            for case, mask, data, _ in test:
                pred = np.column_stack([model.predict(data[feature]) for model in models])
                saved.setdefault((feature, case.case_id), []).append(pred)
                for j, target in enumerate(TARGETS):
                    rows.append(
                        {
                            "seed": seed,
                            "feature": feature,
                            "case_id": case.case_id,
                            "target": target,
                            "nrmse": nrmse(case.targets[mask, j], pred[:, j]),
                        }
                    )

    with (a.out / "offwall_surrogate_v2_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    arrays: dict[str, np.ndarray] = {}
    summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for (feature, case_id), values in saved.items():
        arrays[f"{feature}__{case_id}"] = np.stack(values)
        summary.setdefault(case_id, {})[feature] = {}
        truth = next(
            case.targets[mask]
            for case, mask, _, _ in test
            if case.case_id == case_id
        )
        for j, target in enumerate(TARGETS):
            errors = [nrmse(truth[:, j], value[:, j]) for value in values]
            summary[case_id][feature][target] = {
                "mean_nrmse": float(np.mean(errors)),
                "std_nrmse": float(np.std(errors, ddof=1)),
            }

    np.savez_compressed(
        a.out / "offwall_surrogate_v2_predictions.npz",
        depths_lambda=depth_ref,
        **arrays,
    )
    (a.out / "offwall_surrogate_v2_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (a.out / "METHOD.txt").write_text(
        "Cp and Cf were fitted by independent ExtraTrees regressors.\n"
        "Training cases: BWD/FWD at Kn=0.1 and 0.8.\n"
        "Blind interpolation cases: BWD/FWD at Kn=0.2 and 0.4.\n"
        "Soff contains S2 plus incoming descriptors at d/lambda="
        + ",".join(f"{x:g}" for x in depth_ref)
        + ".\n"
    )
    print(f"OFFWALL_SURROGATE_V2_COMPLETE={a.out}")


if __name__ == "__main__":
    main()
