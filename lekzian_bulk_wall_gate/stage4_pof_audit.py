#!/usr/bin/env python3
"""Post-Stage-3 representation audit for a redesigned Physics of Fluids paper.

This is an explicitly exploratory follow-up.  It does not revise the locked
Stage-3 verdict or its one-percentage-point prospective gate.  The same compact
operator, folds, seeds, preprocessing, and training schedule are repeated while
inference controls determine whether any residual field gain requires the
two-dimensional wall-aligned arrangement, only patch statistics, or merely
case-level information.  No DSMC/SPARTA calculation is launched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import stage3_spatial as s3


TARGETS = s3.TARGETS
PHYSICAL_CHANNELS = ("u", "v", "T", "logP")
AUDIT_CONFIGS = (
    "A0", "A_full", "A_patchmean", "A_cellperm", "A_casepool",
    "A_surfaceperm", "A_no_u", "A_no_v", "A_no_T", "A_no_logP",
)
COMPARISONS = (
    ("A0", "A_full"),
    ("A_patchmean", "A_full"),
    ("A_cellperm", "A_full"),
    ("A_casepool", "A_full"),
    ("A_surfaceperm", "A_full"),
    ("A_no_u", "A_full"),
    ("A_no_v", "A_full"),
    ("A_no_T", "A_full"),
    ("A_no_logP", "A_full"),
)


def valid_weighted_patch_mean(raw_patches: np.ndarray, standardized: np.ndarray) -> np.ndarray:
    """Destroy 2-D field structure while retaining each patch's field means.

    Only the four physical channels are replaced.  Nearest-distance and valid
    channels remain unchanged so the control does not confound representation
    loss with sampling geometry.
    """
    result = standardized.copy()
    valid = (raw_patches[:, 5:6] > 0.5).astype(np.float32)
    denominator = np.maximum(valid.sum(axis=(2, 3), keepdims=True), 1.0)
    means = (standardized[:, :4] * valid).sum(axis=(2, 3), keepdims=True) / denominator
    result[:, :4] = means
    return result


def cell_permutation_variant(patches: np.ndarray) -> np.ndarray:
    """Jointly permute physical cells, preserving every channel marginal."""
    result = patches.copy()
    height, width = patches.shape[2:]
    rng = np.random.default_rng(s3.stable_seed("stage4", "cell_permutation", height, width))
    permutation = rng.permutation(height * width)
    physical = patches[:, :4].reshape(len(patches), 4, -1)
    result[:, :4] = physical[:, :, permutation].reshape(len(patches), 4, height, width)
    return result


def case_pool_variant(data: s3.SpatialData, patches: np.ndarray) -> np.ndarray:
    """Broadcast each case's surface-averaged local field to all its wall points."""
    result = patches.copy()
    for case_id in sorted(set(data.case_id)):
        location = np.flatnonzero(data.case_id == case_id)
        result[location, :4] = patches[location, :4].mean(axis=0, keepdims=True)
    return result


def surface_permutation_variant(data: s3.SpatialData, patches: np.ndarray) -> np.ndarray:
    """Destroy wall-point alignment but preserve each case's patch multiset."""
    result = patches.copy()
    for case_id in sorted(set(data.case_id)):
        location = np.flatnonzero(data.case_id == case_id)
        location = location[np.argsort(data.s01[location])]
        rng = np.random.default_rng(s3.stable_seed("stage4", "surface_permutation", case_id))
        permutation = rng.permutation(len(location))
        if len(location) > 1 and np.any(permutation == np.arange(len(location))):
            permutation = np.roll(permutation, 1)
        result[location] = patches[location[permutation]]
    return result


def channel_ablation_variant(patches: np.ndarray, channel: int) -> np.ndarray:
    """Mean-fill one train-standardised physical channel."""
    if channel not in range(4):
        raise ValueError("Only physical channels 0..3 may be ablated.")
    result = patches.copy()
    result[:, channel] = 0.0
    return result


def build_variants(data: s3.SpatialData, prep: s3.Preprocessor) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    _, patches, _, _ = prep.arrays(data)
    variants = {
        "A_full": patches,
        "A_patchmean": valid_weighted_patch_mean(data.patches, patches),
        "A_cellperm": cell_permutation_variant(patches),
        "A_casepool": case_pool_variant(data, patches),
        "A_surfaceperm": surface_permutation_variant(data, patches),
    }
    for channel, name in enumerate(PHYSICAL_CHANNELS):
        variants[f"A_no_{name}"] = channel_ablation_variant(patches, channel)
    return patches, variants


def predict_audit_configs(
    model: s3.SpatialOperator,
    data: s3.SpatialData,
    prep: s3.Preprocessor,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    context, _, _, _ = prep.arrays(data)
    patches, variants = build_variants(data, prep)
    masks = s3.make_spatial_masks(
        data.t_grid, data.n_grid, data.tangent, data.normal, args.near_radius_hs
    )
    t_coord, n_coord = s3.coordinate_tensors(data, args.device_obj)
    result: Dict[str, np.ndarray] = {}
    model.eval()
    step = max(args.batch_size, 1024)
    for config in AUDIT_CONFIGS:
        source = patches if config == "A0" else variants[config]
        mask_name = "P0" if config == "A0" else "P_full"
        pieces = []
        with torch.no_grad():
            for start in range(0, len(context), step):
                stop = min(len(context), start + step)
                c = torch.from_numpy(context[start:stop]).to(args.device_obj)
                p = torch.from_numpy(source[start:stop]).to(args.device_obj)
                m = torch.from_numpy(masks[mask_name][start:stop]).to(args.device_obj)
                pieces.append(model(c, p, m, t_coord, n_coord).cpu().numpy())
        normalized = np.concatenate(pieces, axis=0)
        result[config] = prep.invert_targets(normalized, data.Ma).astype(np.float32)
    return result


def write_run_record(dataset_path: Path, data: s3.SpatialData, args: argparse.Namespace) -> None:
    out_dir = Path(args.out).expanduser().resolve()
    stage3_decision = json.loads(Path(args.stage3_decision).read_text(encoding="utf-8"))
    if stage3_decision.get("verdict") != "NO_ACTIONABLE_SPATIAL_BULK_SIGNAL":
        raise RuntimeError("Stage 4 requires the locked completed Stage-3 negative verdict.")
    payload = {
        "dataset_sha256": s3.sha256_file(dataset_path),
        "stage3_decision_sha256": s3.sha256_file(Path(args.stage3_decision)),
        "stage3_summary_sha256": s3.sha256_file(Path(args.stage3_summary)),
        "code_sha256": s3.sha256_file(Path(__file__).resolve()),
        "cv": args.cv_list, "seeds": args.seeds_list,
        "only_scheme": args.only_scheme, "only_outer_group": args.only_outer_group,
        "epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "hidden": args.hidden, "latent": args.latent, "depth": args.depth,
        "dropout": args.dropout, "residual_scale": args.residual_scale,
        "near_radius_hs": args.near_radius_hs, "max_parameters": args.max_parameters,
        "bootstrap": args.bootstrap, "loco_tolerances": args.loco_tolerance_map,
        "pairout_tolerances": args.pairout_tolerance_map,
        "replication_tolerance_pp": args.replication_tolerance_pp,
        "audit_configs": AUDIT_CONFIGS,
        "venue": "Physics of Fluids",
        "stage3_verdict_locked": True,
    }
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    path = out_dir / "stage4_run_config.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("signature") != signature:
            raise RuntimeError("Output directory contains a different Stage-4 run signature.")
    record = {
        "signature": signature, "payload": payload, "dataset": str(dataset_path),
        "stage3_decision": str(Path(args.stage3_decision).resolve()),
        "stage3_summary": str(Path(args.stage3_summary).resolve()),
        "n_cases": len(set(data.case_id)), "n_surface_samples": len(data.targets),
        "patch_shape": list(data.patches.shape[1:]), "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "hostname": os.uname().nodename,
        "started_unix": time.time(), "new_dsmc_runs": 0,
        "higher_order_moments": False, "exploratory_followup": True,
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def run_one_task(data: s3.SpatialData, args: argparse.Namespace) -> None:
    if not args.only_scheme or not args.only_outer_group or len(args.seeds_list) != 1:
        raise ValueError("Worker mode requires one scheme, one outer group, and one seed.")
    out_dir = Path(args.out).expanduser().resolve()
    marker = out_dir / "TASK_COMPLETE"
    if args.resume and marker.exists():
        print(f"[SKIP] completed task {out_dir}", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    scheme, outer_group, seed = args.only_scheme, args.only_outer_group, args.seeds_list[0]
    groups = s3.outer_groups(data, scheme)
    if outer_group not in set(groups):
        raise ValueError(f"Unknown {scheme} group {outer_group!r}.")
    test_index = np.flatnonzero(groups == outer_group)
    outer_train_index = np.flatnonzero(groups != outer_group)
    train_groups = groups[outer_train_index]
    inner_group = s3.choose_inner_group(
        train_groups, outer_group, s3.stable_seed("stage3_inner", scheme, outer_group)
    )
    validation_local = train_groups == inner_group
    train_index = outer_train_index[~validation_local]
    validation_index = outer_train_index[validation_local]
    train_data = data.take(train_index)
    validation_data = data.take(validation_index)
    test_data = data.take(test_index)
    prep = s3.Preprocessor.fit(train_data)
    # Reuse the exact Stage-3 training seed to make A0/A_full a direct audit
    # replication instead of an opportunistically retuned model.
    model, diagnostics = s3.train_model(
        train_data, validation_data, prep, args,
        s3.stable_seed("stage3_model", scheme, outer_group, seed),
    )
    validation_predictions = predict_audit_configs(model, validation_data, prep, args)
    test_predictions = predict_audit_configs(model, test_data, prep, args)
    validation_frame = s3.prediction_frame(
        validation_data, validation_predictions, scheme, outer_group, inner_group, seed
    )
    test_frame = s3.prediction_frame(
        test_data, test_predictions, scheme, outer_group, inner_group, seed
    )
    validation_frame.to_csv(out_dir / "validation_predictions.csv", index=False)
    test_frame.to_csv(out_dir / "surface_predictions.csv", index=False)
    s3.case_metrics(test_frame).to_csv(out_dir / "case_metrics.csv", index=False)
    diagnostics.update({
        "scheme": scheme, "outer_group": outer_group, "inner_group": inner_group,
        "seed": seed, "n_train": len(train_data.targets),
        "n_validation": len(validation_data.targets), "n_test": len(test_data.targets),
        "audit_configs": list(AUDIT_CONFIGS), "stage3_training_seed_reused": True,
    })
    (out_dir / "training_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    marker.write_text(json.dumps(diagnostics, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[DONE] Stage4 {scheme}|{outer_group}|{seed}; epoch={diagnostics['best_epoch']}", flush=True)


def add_case_metadata(case_table: pd.DataFrame, data: s3.SpatialData) -> pd.DataFrame:
    metadata = pd.DataFrame({
        "case_id": data.case_id, "Ma": data.Ma, "Kn": data.Kn, "geom": data.geom,
    }).drop_duplicates("case_id")
    return case_table.drop(columns=["Ma", "Kn", "geom"], errors="ignore").merge(
        metadata, on="case_id", how="left", validate="many_to_one"
    )


def paired_gain_table(units: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for (scheme, target), group in units.groupby(["scheme", "target"], sort=False):
        pivot = group.pivot(index="unit_id", columns="config", values="relL2")
        for baseline, candidate in COMPARISONS:
            difference = (pivot[baseline] - pivot[candidate]).dropna().to_numpy(float)
            mean, lower, upper = s3.bootstrap_mean(
                difference, args.bootstrap,
                s3.stable_seed("stage4_gain", scheme, target, baseline, candidate),
            )
            rows.append({
                "scheme": scheme, "target": target,
                "comparison": f"{baseline}-{candidate}",
                "n_bootstrap_units": len(difference), "mean_gain": mean,
                "mean_gain_pp": 100.0 * mean, "bootstrap_CI2p5": lower,
                "bootstrap_CI97p5": upper, "CI2p5_pp": 100.0 * lower,
                "CI97p5_pp": 100.0 * upper,
            })
    return pd.DataFrame(rows)


def summarize(case_table: pd.DataFrame, args: argparse.Namespace, out_dir: Path):
    table = case_table.copy()
    table["unit_id"] = np.where(table["scheme"] == "loco", table["case_id"], table["outer_group"])
    units = table.groupby(["scheme", "target", "config", "unit_id"], as_index=False).agg(
        relL2=("relL2", "mean"), rangeMAE_pct=("rangeMAE_pct", "mean")
    )
    rows = []
    for (scheme, target, config), group in units.groupby(["scheme", "target", "config"]):
        mean, lower, upper = s3.bootstrap_mean(
            group["relL2"].to_numpy(float), args.bootstrap,
            s3.stable_seed("stage4_summary", scheme, target, config),
        )
        rows.append({
            "scheme": scheme, "target": target, "config": config,
            "n_bootstrap_units": group["unit_id"].nunique(), "mean_relL2": mean,
            "median_relL2": float(group["relL2"].median()),
            "bootstrap_CI2p5": lower, "bootstrap_CI97p5": upper,
            "mean_rangeMAE_pct": float(group["rangeMAE_pct"].mean()),
        })
    summary = pd.DataFrame(rows)
    gains = paired_gain_table(units, args)
    summary.to_csv(out_dir / "stage4_ensemble_summary.csv", index=False)
    gains.to_csv(out_dir / "stage4_paired_gains.csv", index=False)
    return summary, gains


def regime_gains(case_table: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    # LOCO contains exactly one held-out prediction per physical case and is
    # therefore the non-duplicated basis for exploratory regime localization.
    table = case_table[case_table["scheme"] == "loco"].copy()
    rows = []
    for target, target_table in table.groupby("target"):
        pivot = target_table.pivot_table(
            index=["case_id", "Ma", "Kn", "geom"], columns="config", values="relL2", aggfunc="mean"
        ).reset_index()
        for baseline, candidate in COMPARISONS:
            pivot["gain"] = pivot[baseline] - pivot[candidate]
            for factor in ("Ma", "Kn", "geom"):
                for value, group in pivot.groupby(factor):
                    values = group["gain"].to_numpy(float)
                    mean, lower, upper = s3.bootstrap_mean(
                        values, args.bootstrap,
                        s3.stable_seed("stage4_regime", target, baseline, candidate, factor, value),
                    )
                    rows.append({
                        "target": target, "comparison": f"{baseline}-{candidate}",
                        "factor": factor, "value": str(value), "n_cases": len(values),
                        "mean_gain": mean, "mean_gain_pp": 100.0 * mean,
                        "CI2p5_pp": 100.0 * lower, "CI97p5_pp": 100.0 * upper,
                    })
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "stage4_regime_gains.csv", index=False)
    return result


def lookup(table: pd.DataFrame, scheme: str, target: str, comparison_or_config: str, value: str):
    match = table[
        (table["scheme"] == scheme) & (table["target"] == target)
        & (table[comparison_or_config] == value)
    ]
    if len(match) != 1:
        raise RuntimeError(f"Expected one row for {scheme}/{target}/{value}; found {len(match)}.")
    return match.iloc[0]


def robust_positive(gains: pd.DataFrame, target: str, comparison: str) -> Tuple[bool, Dict[str, float]]:
    loco = lookup(gains, "loco", target, "comparison", comparison)
    pair = lookup(gains, "pairout", target, "comparison", comparison)
    passed = bool(float(loco["bootstrap_CI2p5"]) > 0 and float(pair["mean_gain"]) > 0)
    return passed, {
        "loco_mean_gain_pp": float(loco["mean_gain_pp"]),
        "loco_CI2p5_pp": float(loco["CI2p5_pp"]),
        "loco_CI97p5_pp": float(loco["CI97p5_pp"]),
        "pairout_mean_gain_pp": float(pair["mean_gain_pp"]),
    }


def make_decision(summary: pd.DataFrame, gains: pd.DataFrame, args: argparse.Namespace, out_dir: Path):
    checks = []
    for target in TARGETS:
        full_loco = lookup(summary, "loco", target, "config", "A_full")
        full_pair = lookup(summary, "pairout", target, "config", "A_full")
        accuracy = bool(
            float(full_loco["bootstrap_CI97p5"]) <= args.loco_tolerance_map[target]
            and float(full_pair["mean_relL2"]) <= args.pairout_tolerance_map[target]
        )
        increment, increment_numbers = robust_positive(gains, target, "A0-A_full")
        patch_structure, patch_numbers = robust_positive(gains, target, "A_patchmean-A_full")
        cell_structure, cell_numbers = robust_positive(gains, target, "A_cellperm-A_full")
        case_specific, case_numbers = robust_positive(gains, target, "A_casepool-A_full")
        alignment, alignment_numbers = robust_positive(gains, target, "A_surfaceperm-A_full")
        channel_support = {}
        for channel in PHYSICAL_CHANNELS:
            passed, numbers = robust_positive(gains, target, f"A_no_{channel}-A_full")
            channel_support[channel] = {"pass": passed, **numbers}
        if not increment:
            classification = "NO_INCREMENTAL_FIELD_SUPPORT"
        elif patch_structure and cell_structure and alignment:
            classification = "STRUCTURED_WALL_ALIGNED_SUPPORT"
        elif not patch_structure and not cell_structure:
            classification = "PATCH_STATISTICS_SUFFICIENT"
        else:
            classification = "PARTIAL_OR_UNSTABLE_SPATIAL_SUPPORT"
        checks.append({
            "target": target, "classification": classification,
            "accuracy_pass": accuracy, "field_increment_pass": increment,
            "patchmean_control_pass": patch_structure,
            "cell_permutation_control_pass": cell_structure,
            "case_pool_control_pass": case_specific,
            "surface_permutation_control_pass": alignment,
            "A_full_loco_mean_relL2": float(full_loco["mean_relL2"]),
            "A_full_loco_CI97p5": float(full_loco["bootstrap_CI97p5"]),
            "A_full_pairout_mean_relL2": float(full_pair["mean_relL2"]),
            "field_increment": increment_numbers, "patchmean": patch_numbers,
            "cell_permutation": cell_numbers, "case_pool": case_numbers,
            "surface_permutation": alignment_numbers, "channel_support": channel_support,
        })
    replication = replication_audit(summary, args, out_dir)
    cp = next(item for item in checks if item["target"] == "Cp")
    pof_candidate = bool(
        cp["accuracy_pass"] and cp["field_increment_pass"]
        and cp["classification"] == "STRUCTURED_WALL_ALIGNED_SUPPORT"
    )
    if not replication["pass"]:
        verdict = "STAGE4_REPLICATION_FAILURE"
        action = "Do not interpret Stage 4; diagnose why A0/A_full did not reproduce Stage-3 P0/P_full."
    elif pof_candidate:
        verdict = "POF_REFRAME_CANDIDATE"
        action = (
            "Reframe around quantity-dependent predictability versus representation support; "
            "retain Stage 3 as the locked confirmatory result."
        )
    else:
        verdict = "STOP_CURRENT_POF_OBSERVABILITY_ROUTE"
        action = "Do not submit the observability route to PoF; preserve the negative audit and redirect the dataset."
    decision = {
        "verdict": verdict, "action": action, "venue": "Physics of Fluids",
        "stage3_confirmatory_verdict_locked": "NO_ACTIONABLE_SPATIAL_BULK_SIGNAL",
        "stage4_role": "exploratory representation audit; not a replacement confirmatory test",
        "stage3_replication": replication,
        "checks": checks, "new_dsmc_runs": 0, "higher_order_moments": False,
        "allowed_claim": "quantity-dependent predictive representation support",
        "forbidden_claims": ["causal information horizon", "physical observability radius"],
    }
    (out_dir / "stage4_pof_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    lines = [f"VERDICT: {verdict}", f"ACTION: {action}", ""]
    for item in checks:
        lines.append(
            f"{item['target']}: accuracy={item['accuracy_pass']}, "
            f"field_increment={item['field_increment_pass']}, "
            f"classification={item['classification']}"
        )
    (out_dir / "stage4_pof_decision.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def replication_audit(summary: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> Dict[str, object]:
    """Verify the repeated A0/A_full calculation against frozen Stage-3 means."""
    frozen = pd.read_csv(args.stage3_summary)
    rows = []
    for scheme in ("loco", "pairout"):
        for target in TARGETS:
            for old_config, new_config in (("P0", "A0"), ("P_full", "A_full")):
                old = frozen[
                    (frozen["scheme"] == scheme) & (frozen["target"] == target)
                    & (frozen["config"] == old_config)
                ]
                new = summary[
                    (summary["scheme"] == scheme) & (summary["target"] == target)
                    & (summary["config"] == new_config)
                ]
                if len(old) != 1 or len(new) != 1:
                    raise RuntimeError(
                        f"Replication lookup failed for {scheme}/{target}/{old_config}/{new_config}."
                    )
                old_error = float(old.iloc[0]["mean_relL2"])
                new_error = float(new.iloc[0]["mean_relL2"])
                delta_pp = 100.0 * (new_error - old_error)
                rows.append({
                    "scheme": scheme, "target": target,
                    "stage3_config": old_config, "stage4_config": new_config,
                    "stage3_mean_relL2": old_error, "stage4_mean_relL2": new_error,
                    "delta_pp": delta_pp,
                    "pass": abs(delta_pp) <= args.replication_tolerance_pp,
                })
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "stage4_stage3_replication_audit.csv", index=False)
    return {
        "pass": bool(table["pass"].all()),
        "tolerance_pp": float(args.replication_tolerance_pp),
        "max_abs_delta_pp": float(table["delta_pp"].abs().max()),
        "failed_comparisons": table.loc[~table["pass"], [
            "scheme", "target", "stage3_config", "stage4_config", "delta_pp"
        ]].to_dict(orient="records"),
    }


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for row, scheme in enumerate(("loco", "pairout")):
        for column, target in enumerate(TARGETS):
            ax = axes[row, column]
            part = summary[
                (summary["scheme"] == scheme) & (summary["target"] == target)
            ].set_index("config").reindex(AUDIT_CONFIGS)
            x = np.arange(len(part))
            mean = 100.0 * part["mean_relL2"].to_numpy(float)
            lower = 100.0 * part["bootstrap_CI2p5"].to_numpy(float)
            upper = 100.0 * part["bootstrap_CI97p5"].to_numpy(float)
            ax.errorbar(x, mean, yerr=np.vstack((mean - lower, upper - mean)), marker="o", capsize=3)
            ax.set_xticks(x, [name.removeprefix("A_") for name in AUDIT_CONFIGS], rotation=45, ha="right")
            ax.set_title(f"{scheme}: {target}")
            ax.set_ylabel("ensemble relative L2 error (%)")
            ax.grid(alpha=0.25)
    fig.suptitle("PoF Stage 4: representation and channel falsification audit")
    fig.tight_layout()
    fig.savefig(out_dir / "stage4_pof_representation_audit.pdf")
    fig.savefig(out_dir / "stage4_pof_representation_audit.png", dpi=200)
    plt.close(fig)


def aggregate_results(data: s3.SpatialData, args: argparse.Namespace):
    out_dir = Path(args.out).expanduser().resolve()
    task_root = Path(args.aggregate_task_root).expanduser().resolve()
    expected = (len(set(data.case_id)) + len(set(s3.pairout_groups(data)))) * len(args.seeds_list)
    completed = len(list(task_root.glob("task_*/TASK_COMPLETE")))
    if completed != expected:
        raise RuntimeError(f"Expected {expected} Stage-4 tasks; found {completed} complete markers.")
    individual_points = s3.concat_task_files(task_root, "surface_predictions.csv")
    validation = s3.concat_task_files(task_root, "validation_predictions.csv")
    individual_cases = s3.concat_task_files(task_root, "case_metrics.csv")
    duplicate = ["scheme", "outer_group", "seed", "target", "config", "case_id"]
    if individual_cases.duplicated(duplicate).any():
        raise RuntimeError("Duplicate Stage-4 case metrics detected.")
    individual_points.to_csv(out_dir / "stage4_individual_seed_surface_predictions.csv", index=False)
    validation.to_csv(out_dir / "stage4_individual_seed_validation_predictions.csv", index=False)
    add_case_metadata(individual_cases, data).to_csv(
        out_dir / "stage4_individual_seed_case_metrics.csv", index=False
    )
    ensemble_points = s3.ensemble_predictions(individual_points, len(args.seeds_list))
    ensemble_cases = add_case_metadata(s3.case_metrics(ensemble_points.assign(seed=-1)), data)
    ensemble_points.to_csv(out_dir / "stage4_ensemble_surface_predictions.csv", index=False)
    ensemble_cases.to_csv(out_dir / "stage4_ensemble_case_metrics.csv", index=False)
    summary, gains = summarize(ensemble_cases, args, out_dir)
    regime_gains(ensemble_cases, args, out_dir)
    decision = make_decision(summary, gains, args, out_dir)
    plot_summary(summary, out_dir)
    return decision


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stage3-decision", required=True)
    parser.add_argument("--stage3-summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cv", default="loco,pairout")
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--only-scheme", choices=("loco", "pairout"))
    parser.add_argument("--only-outer-group")
    parser.add_argument("--aggregate-task-root")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--latent", type=int, default=48)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.03)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--near-radius-hs", type=float, default=0.75)
    parser.add_argument("--max-parameters", type=int, default=100000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--loco-tolerances", default="Cp=0.10,Cq=0.10,tau_abs=0.20")
    parser.add_argument("--pairout-tolerances", default="Cp=0.15,Cq=0.15,tau_abs=0.25")
    parser.add_argument("--replication-tolerance-pp", type=float, default=0.50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.cv_list = [value.strip() for value in args.cv.split(",") if value.strip()]
    args.seeds_list = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    args.device_obj = s3.resolve_device(args.device)
    args.loco_tolerance_map = s3.parse_target_map(args.loco_tolerances)
    args.pairout_tolerance_map = s3.parse_target_map(args.pairout_tolerances)
    if args.only_scheme and len(args.seeds_list) != 1:
        raise ValueError("A Stage-4 worker receives exactly one seed.")
    return args


def main() -> None:
    args = parse_arguments()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset).expanduser().resolve()
    data = s3.load_dataset(dataset_path)
    write_run_record(dataset_path, data, args)
    if args.aggregate_task_root:
        decision = aggregate_results(data, args)
        print(f"[DONE] Stage-4 PoF verdict={decision['verdict']}", flush=True)
    else:
        run_one_task(data, args)


if __name__ == "__main__":
    main()
