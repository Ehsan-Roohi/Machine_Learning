#!/usr/bin/env python3
"""Blind intermediate-Kn gate with equal-capacity feature ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from analyze_moment_gate import CaseData, load_case, nrmse

FEATURES = ("P", "S0", "S1", "S2")
TARGETS = ("cp", "cf")
COLORS = {"P": "#8c8c8c", "S0": "#e07a1f", "S1": "#2878b5", "S2": "#2a9d62"}


def read_cases(root: Path) -> list[CaseData]:
    ids = [x.strip() for x in (root / "case_list.txt").read_text().splitlines() if x.strip()]
    return [load_case(root / cid) for cid in ids]


def padded(case: CaseData, feature: str, common_mean: np.ndarray) -> np.ndarray:
    full = np.tile(common_mean, (len(case.targets), 1))
    if feature == "P":
        # Position, local surface frame, and Kn only: parameter-conditioned baseline.
        full[:, :7] = case.features["S2"][:, :7]
    else:
        values = case.features[feature]
        full[:, :values.shape[1]] = values
    return full


def model(seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=800, min_samples_leaf=2, max_features=0.85,
        n_jobs=-1, random_state=seed,
    )


def fit_predict(train: list[CaseData], test: list[CaseData], seed: int):
    region = "nearfield"
    x2 = np.vstack([c.features["S2"][c.region[region]] for c in train])
    common_mean = x2.mean(0)
    y_train = np.vstack([c.targets[c.region[region]] for c in train])
    predictions: dict[str, dict[str, np.ndarray]] = {f: {} for f in FEATURES}
    for feature_index, feature in enumerate(FEATURES):
        x_train = np.vstack([padded(c, feature, common_mean)[c.region[region]] for c in train])
        fitted = model(seed + feature_index)
        fitted.fit(x_train, y_train)
        for c in test:
            predictions[feature][c.case_id] = fitted.predict(padded(c, feature, common_mean))
    return predictions


def case_rows(test: list[CaseData], predictions, seed: int):
    rows = []
    for c in test:
        mask = c.region["protrusion"]
        for j, target in enumerate(TARGETS):
            row = {"seed": seed, "case_id": c.case_id, "geometry": c.geometry,
                   "knudsen": c.knudsen, "target": target, "samples": int(mask.sum())}
            for feature in FEATURES:
                row[f"nrmse_{feature}"] = nrmse(c.targets[mask, j], predictions[feature][c.case_id][mask, j])
            row["S1_gain_vs_S0_percent"] = 100*(row["nrmse_S0"]-row["nrmse_S1"])/max(row["nrmse_S0"], 1e-15)
            row["S2_gain_vs_S1_percent"] = 100*(row["nrmse_S1"]-row["nrmse_S2"])/max(row["nrmse_S1"], 1e-15)
            rows.append(row)
    return rows


def aggregate(test: list[CaseData], predictions_by_seed, seeds):
    result = {}
    truth = np.vstack([c.targets[c.region["protrusion"]] for c in test])
    case_id = np.concatenate([np.full(c.region["protrusion"].sum(), c.case_id) for c in test])
    for j, target in enumerate(TARGETS):
        result[target] = {}
        for feature in FEATURES:
            errs = []
            for predictions in predictions_by_seed:
                pred = np.vstack([predictions[feature][c.case_id][c.region["protrusion"]] for c in test])
                errs.append(nrmse(truth[:, j], pred[:, j]))
            result[target][feature] = {"nrmse_mean": float(np.mean(errs)), "nrmse_std": float(np.std(errs, ddof=1))}
        s0 = result[target]["S0"]["nrmse_mean"]
        s1 = result[target]["S1"]["nrmse_mean"]
        result[target]["S1_gain_vs_S0_percent"] = 100*(s0-s1)/max(s0, 1e-15)

        # Case bootstrap uses seed-averaged predictions and preserves complete wall profiles.
        avg = {}
        for feature in ("S0", "S1"):
            avg[feature] = np.mean([
                np.vstack([p[feature][c.case_id][c.region["protrusion"]] for c in test])
                for p in predictions_by_seed], axis=0)
        unique = np.unique(case_id); rng = np.random.default_rng(20260830 + j)
        gains = []
        indices = {cid: np.flatnonzero(case_id == cid) for cid in unique}
        for _ in range(5000):
            sampled = rng.choice(unique, len(unique), replace=True)
            take = np.concatenate([indices[cid] for cid in sampled])
            e0 = nrmse(truth[take, j], avg["S0"][take, j]); e1 = nrmse(truth[take, j], avg["S1"][take, j])
            gains.append(100*(e0-e1)/max(e0, 1e-15))
        result[target]["S1_gain_ci95_percent"] = [float(x) for x in np.percentile(gains, [2.5, 97.5])]
    cp, cf = result["cp"], result["cf"]
    checks = {
        "S1_cp_nrmse_below_5_percent": cp["S1"]["nrmse_mean"] < 0.05,
        "S1_cf_nrmse_below_15_percent": cf["S1"]["nrmse_mean"] < 0.15,
        "S1_gain_at_least_20_percent_for_cf": cf["S1_gain_vs_S0_percent"] >= 20,
        "cf_gain_ci_excludes_zero": cf["S1_gain_ci95_percent"][0] > 0,
        "S1_beats_parameter_baseline": all(result[t]["S1"]["nrmse_mean"] < result[t]["P"]["nrmse_mean"] for t in TARGETS),
    }
    return result, checks, "PASS" if all(checks.values()) else "FAIL"


def plot_profiles(test, predictions_by_seed, out):
    for target_index, target in enumerate(TARGETS):
        fig, axes = plt.subplots(3, 2, figsize=(10.8, 11.0), sharex=True)
        for row, geometry in enumerate(("BWD", "FWD", "ISO")):
            for col, kn in enumerate((0.2, 0.4)):
                c = next(x for x in test if x.geometry == geometry and abs(x.knudsen-kn) < 1e-12)
                mask = c.region["protrusion"]; wall = c.midpoint[mask]
                ds = np.linalg.norm(np.diff(wall, axis=0), axis=1); s = np.r_[0, np.cumsum(ds)]; s /= s[-1]
                ax = axes[row, col]
                y = c.targets[mask, target_index]
                sem = c.target_blocks[:, mask, target_index].std(0, ddof=1)/2
                ax.fill_between(s, y-2.776*sem, y+2.776*sem, color="0.83", alpha=.6, linewidth=0)
                ax.plot(s, y, "o-", color="black", ms=2.7, lw=1.2, label="DSMC")
                for feature in ("P", "S0", "S1", "S2"):
                    pred = np.mean([p[feature][c.case_id][mask, target_index] for p in predictions_by_seed], axis=0)
                    ax.plot(s, pred, "--" if feature == "P" else "-", color=COLORS[feature], lw=1.25, label=("parameter baseline" if feature == "P" else feature))
                ax.axvline(.5, color="0.5", ls=":", lw=1); ax.axhline(0, color="0.6", lw=.7)
                ax.set_title(fr"{geometry}, $Kn={kn:g}$", weight="bold")
                ax.grid(alpha=.2)
                if col == 0: ax.set_ylabel(r"$C_p$" if target == "cp" else r"signed $C_f$")
                if row == 2: ax.set_xlabel(r"protrusion arclength, $s/L_w$")
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(.5, .985), frameon=False)
        fig.suptitle(("Blind intermediate-Kn wall-pressure profiles" if target=="cp" else "Blind intermediate-Kn signed-shear profiles"), y=.999, fontsize=14, weight="bold")
        fig.tight_layout(rect=(0,0,1,.945))
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out/f"interpolation_{target}_physical_profiles.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint-root", type=Path, required=True)
    ap.add_argument("--intermediate-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47, 71, 101])
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    train = read_cases(a.endpoint_root); test = read_cases(a.intermediate_root)
    if sorted({c.knudsen for c in train}) != [0.1, 0.8] or sorted({c.knudsen for c in test}) != [0.2, 0.4]:
        raise SystemExit("Unexpected train/test Kn support")
    predictions = [fit_predict(train, test, seed) for seed in a.seeds]
    rows = sum((case_rows(test, p, seed) for p,seed in zip(predictions,a.seeds)), [])
    result, checks, verdict = aggregate(test, predictions, a.seeds)
    with (a.output/"interpolation_case_metrics.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    decision={"verdict":verdict,"protocol":{"train_Kn":[0.1,0.8],"blind_test_Kn":[0.2,0.4],"geometries":["BWD","FWD","ISO"],"seeds":a.seeds,"equal_capacity_padding":True},"aggregate":result,"checks":checks}
    (a.output/"interpolation_gate.json").write_text(json.dumps(decision,indent=2)+"\n")
    lines=["# Blind intermediate-Kn moment gate", "", f"**Verdict: {verdict}**", "", "Training: Kn=0.1 and 0.8. Blind testing: Kn=0.2 and 0.4 for BWD/FWD/ISO.", "", "| Target | P | S0 | S1 | S2 | S1 gain | 95% CI |", "|---|---:|---:|---:|---:|---:|---:|"]
    for t in TARGETS:
        q=result[t]; lines.append(f"| {t.upper()} | {100*q['P']['nrmse_mean']:.2f}% | {100*q['S0']['nrmse_mean']:.2f}% | {100*q['S1']['nrmse_mean']:.2f}% | {100*q['S2']['nrmse_mean']:.2f}% | {q['S1_gain_vs_S0_percent']:.2f}% | [{q['S1_gain_ci95_percent'][0]:.2f}, {q['S1_gain_ci95_percent'][1]:.2f}]% |")
    lines += ["", "## Checks", ""]+[f"- {'PASS' if v else 'FAIL'}: `{k}`" for k,v in checks.items()]
    (a.output/"REPORT.md").write_text("\n".join(lines)+"\n")
    plot_profiles(test,predictions,a.output)
    print(json.dumps(decision,indent=2))

if __name__ == "__main__": main()
