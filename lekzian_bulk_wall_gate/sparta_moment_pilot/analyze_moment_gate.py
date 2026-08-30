#!/usr/bin/env python3
"""Evaluate the SPARTA bulk-to-wall moment-sufficiency pilot.

The analysis deliberately averages the four non-overlapping DSMC blocks before
training.  Blocks are used to estimate target uncertainty, never as independent
machine-learning samples.  Generalization is evaluated by leaving one complete
geometry out at a time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor


K_B = 1.380649e-23
MASS_ARGON_KG = 6.6335209e-26
DEPTHS_LAMBDA = (0.5, 1.0, 2.0)
FEATURE_SETS = ("S0", "S1", "S2")
TARGETS = ("cp", "cf")


@dataclass
class CaseData:
    case_id: str
    geometry: str
    knudsen: float
    surface_id: np.ndarray
    midpoint: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    region: dict[str, np.ndarray]
    features: dict[str, np.ndarray]
    feature_names: dict[str, list[str]]
    targets: np.ndarray
    target_blocks: np.ndarray
    ncoll: np.ndarray
    neighbor_keys: np.ndarray


def _column_map(columns: np.ndarray) -> dict[str, int]:
    return {str(name): i for i, name in enumerate(columns.tolist())}


def _local_basis(wall: np.ndarray, columns: dict[str, int]) -> tuple[np.ndarray, ...]:
    v1 = wall[:, [columns["v1x"], columns["v1y"]]]
    v2 = wall[:, [columns["v2x"], columns["v2y"]]]
    midpoint = 0.5 * (v1 + v2)
    tangent = v2 - v1
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    # A common left-to-right orientation makes signed shear comparable on both
    # protrusion faces and gives a gas-facing normal on the lower wall.
    flip = (tangent[:, 0] < 0.0) | ((tangent[:, 0] == 0.0) & (tangent[:, 1] < 0.0))
    tangent[flip] *= -1.0
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    return midpoint, tangent, normal


def _collapse_duplicate_centers(
    grid: np.ndarray, columns: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the statistically populated subcell at duplicate parent centers."""
    coords = grid[:, [columns["xc"], columns["yc"]]]
    order = np.argsort(-grid[:, columns["n"]], kind="stable")
    _, first = np.unique(coords[order], axis=0, return_index=True)
    keep = order[first]
    return grid[keep], keep


def _project_moments(
    grid_rows: np.ndarray,
    columns: dict[str, int],
    tangent: np.ndarray,
    normal: np.ndarray,
) -> tuple[np.ndarray, ...]:
    pxx = grid_rows[:, columns["momxx"]]
    pyy = grid_rows[:, columns["momyy"]]
    pxy = grid_rows[:, columns["momxy"]]
    pzz = grid_rows[:, columns["momzz"]]
    tx, ty = tangent[:, 0], tangent[:, 1]
    nx, ny = normal[:, 0], normal[:, 1]
    pnn = nx * nx * pxx + 2.0 * nx * ny * pxy + ny * ny * pyy
    ptt = tx * tx * pxx + 2.0 * tx * ty * pxy + ty * ty * pyy
    pnt = nx * tx * pxx + (nx * ty + ny * tx) * pxy + ny * ty * pyy
    return pnn, pnt, ptt, pzz


def load_case(case_dir: Path) -> CaseData:
    metadata = json.loads((case_dir / "metadata.json").read_text())
    with np.load(case_dir / "output" / "moment_blocks.npz", allow_pickle=False) as packed:
        grid_blocks = packed["grid"]
        wall_blocks = packed["wall"]
        gc = _column_map(packed["grid_columns"])
        wc = _column_map(packed["wall_columns"])

    grid = grid_blocks.mean(axis=0)
    wall = wall_blocks.mean(axis=0)
    midpoint, tangent, normal = _local_basis(wall, wc)
    surface_id = wall[:, wc["id"]].astype(int)

    physics = metadata["physics"]
    n_inf = float(physics["number_density_m_minus_3"])
    u_inf = float(physics["stream_speed_m_per_s"])
    t_inf = float(physics["temperature_K"])
    p_inf = n_inf * K_B * t_inf
    rho_inf = n_inf * MASS_ARGON_KG
    q_inf = 0.5 * rho_inf * u_inf**2
    energy_flux_scale = p_inf * u_inf
    lam = float(physics["mean_free_path_m"])

    populated = grid[:, gc["n"]] > 0.0
    candidates, _ = _collapse_duplicate_centers(grid[populated], gc)
    centers = candidates[:, [gc["xc"], gc["yc"]]]
    tree = cKDTree(centers)

    context = np.column_stack(
        (
            midpoint[:, 0],
            midpoint[:, 1] / float(metadata["hp_m"]),
            tangent,
            normal,
            np.full(len(wall), math.log10(float(metadata["physics"]["knudsen"]))),
        )
    )
    context_names = ["x", "y_over_hp", "tx", "ty", "nx", "ny", "log10_Kn"]
    s0_parts = [context]
    s1_extra: list[np.ndarray] = []
    s2_extra: list[np.ndarray] = []
    s0_names = list(context_names)
    s1_names: list[str] = []
    s2_names: list[str] = []
    neighbor_keys = []

    for depth in DEPTHS_LAMBDA:
        query = midpoint + depth * lam * normal
        miss, index = tree.query(query, k=1)
        rows = candidates[index]
        neighbor_keys.append(
            np.column_stack((rows[:, gc["id"]], rows[:, gc["split"]])).astype(int)
        )

        u = rows[:, gc["u"]]
        v = rows[:, gc["v"]]
        u_t = u * tangent[:, 0] + v * tangent[:, 1]
        u_n = u * normal[:, 0] + v * normal[:, 1]
        nrho = rows[:, gc["nrho"]]
        temp = rows[:, gc["temp"]]
        p_eq = nrho * K_B * temp
        primitive = np.column_stack(
            (
                nrho / n_inf,
                u_t / u_inf,
                u_n / u_inf,
                temp / t_inf,
                p_eq / p_inf,
                miss / lam,
            )
        )
        tag = f"d{depth:g}L"
        s0_parts.append(primitive)
        s0_names.extend(
            [
                f"nrho_{tag}",
                f"ut_{tag}",
                f"un_{tag}",
                f"T_{tag}",
                f"peq_{tag}",
                f"neighbor_miss_{tag}",
            ]
        )

        pnn, pnt, ptt, pzz = _project_moments(rows, gc, tangent, normal)
        s1_extra.append(np.column_stack((pnn, pnt, ptt, pzz)) / p_inf)
        s1_names.extend([f"Pnn_{tag}", f"Pnt_{tag}", f"Ptt_{tag}", f"Pzz_{tag}"])

        qx = rows[:, gc["heatx"]]
        qy = rows[:, gc["heaty"]]
        q_t = qx * tangent[:, 0] + qy * tangent[:, 1]
        q_n = qx * normal[:, 0] + qy * normal[:, 1]
        s2_extra.append(np.column_stack((q_n, q_t)) / energy_flux_scale)
        s2_names.extend([f"qn_{tag}", f"qt_{tag}"])

    s0 = np.column_stack(s0_parts)
    s1 = np.column_stack((s0, *s1_extra))
    s2 = np.column_stack((s1, *s2_extra))

    press_blocks = wall_blocks[:, :, wc["press"]]
    shear_blocks = (
        wall_blocks[:, :, wc["shx"]] * tangent[None, :, 0]
        + wall_blocks[:, :, wc["shy"]] * tangent[None, :, 1]
    )
    target_blocks = np.stack(
        ((press_blocks - p_inf) / q_inf, shear_blocks / q_inf), axis=2
    )
    targets = target_blocks.mean(axis=0)
    ncoll = wall_blocks[:, :, wc["ncoll"]].mean(axis=0)

    region = {
        "all": np.ones(len(wall), dtype=bool),
        "nearfield": ((midpoint[:, 0] >= 0.10) & (midpoint[:, 0] <= 0.40))
        | (surface_id >= 981),
        "protrusion": surface_id >= 981,
    }
    return CaseData(
        case_id=metadata["case_id"],
        geometry=metadata["geometry"],
        knudsen=float(metadata["physics"]["knudsen"]),
        surface_id=surface_id,
        midpoint=midpoint,
        tangent=tangent,
        normal=normal,
        region=region,
        features={"S0": s0, "S1": s1, "S2": s2},
        feature_names={
            "S0": s0_names,
            "S1": s0_names + s1_names,
            "S2": s0_names + s1_names + s2_names,
        },
        targets=targets,
        target_blocks=target_blocks,
        ncoll=ncoll,
        neighbor_keys=np.stack(neighbor_keys, axis=1),
    )


def nrmse(y: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sqrt(np.mean(np.square(y))))
    return float(np.sqrt(np.mean(np.square(prediction - y))) / max(denominator, 1e-15))


def _fit_model(x: np.ndarray, y: np.ndarray, seed: int, trees: int) -> ExtraTreesRegressor:
    model = ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=3,
        max_features=0.85,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(x, y)
    return model


def loco_predictions(
    cases: list[CaseData], train_region: str, trees: int, seed: int
) -> dict[str, dict[str, np.ndarray]]:
    geometries = sorted({case.geometry for case in cases})
    result: dict[str, dict[str, list[np.ndarray]]] = {
        feature_set: {target: [] for target in TARGETS} for feature_set in FEATURE_SETS
    }
    result["S1_shuffled"] = {target: [] for target in TARGETS}
    truth: list[np.ndarray] = []
    case_ids: list[np.ndarray] = []
    geometry_ids: list[np.ndarray] = []
    surface_ids: list[np.ndarray] = []
    regions: dict[str, list[np.ndarray]] = {name: [] for name in ("nearfield", "protrusion")}

    for fold, held_geometry in enumerate(geometries):
        train_cases = [case for case in cases if case.geometry != held_geometry]
        test_cases = [case for case in cases if case.geometry == held_geometry]
        x_train = {
            name: np.vstack([case.features[name][case.region[train_region]] for case in train_cases])
            for name in FEATURE_SETS
        }
        y_train = np.vstack([case.targets[case.region[train_region]] for case in train_cases])
        x_test = {
            name: np.vstack([case.features[name][case.region[train_region]] for case in test_cases])
            for name in FEATURE_SETS
        }
        y_test = np.vstack([case.targets[case.region[train_region]] for case in test_cases])
        truth.append(y_test)
        case_ids.append(
            np.concatenate(
                [np.full(case.region[train_region].sum(), case.case_id, dtype="U32") for case in test_cases]
            )
        )
        geometry_ids.append(np.full(len(y_test), held_geometry, dtype="U8"))
        surface_ids.append(
            np.concatenate([case.surface_id[case.region[train_region]] for case in test_cases])
        )
        for region_name in regions:
            regions[region_name].append(
                np.concatenate(
                    [case.region[region_name][case.region[train_region]] for case in test_cases]
                )
            )

        for target_index, target_name in enumerate(TARGETS):
            s1_model: ExtraTreesRegressor | None = None
            for feature_index, feature_set in enumerate(FEATURE_SETS):
                model = _fit_model(
                    x_train[feature_set], y_train[:, target_index],
                    seed + 100 * fold + 10 * target_index + feature_index, trees,
                )
                result[feature_set][target_name].append(model.predict(x_test[feature_set]))
                if feature_set == "S1":
                    s1_model = model

            assert s1_model is not None
            shuffled_parts = []
            offset = 0
            s0_width = x_train["S0"].shape[1]
            rng = np.random.default_rng(seed + 1000 + 10 * fold + target_index)
            for case in test_cases:
                count = int(case.region[train_region].sum())
                part = x_test["S1"][offset : offset + count].copy()
                part[:, s0_width:] = part[rng.permutation(count), s0_width:]
                shuffled_parts.append(part)
                offset += count
            result["S1_shuffled"][target_name].append(
                s1_model.predict(np.vstack(shuffled_parts))
            )

    packed: dict[str, dict[str, np.ndarray]] = {
        feature_set: {target: np.concatenate(values) for target, values in targets.items()}
        for feature_set, targets in result.items()
    }
    packed["meta"] = {
        "truth": np.vstack(truth),
        "case_id": np.concatenate(case_ids),
        "geometry": np.concatenate(geometry_ids),
        "surface_id": np.concatenate(surface_ids),
        **{name: np.concatenate(values) for name, values in regions.items()},
    }
    return packed


def _case_bootstrap_gain(
    y: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    case_id: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    unique = np.unique(case_id)
    rng = np.random.default_rng(seed)
    gains = np.empty(iterations)
    indices = {case: np.flatnonzero(case_id == case) for case in unique}
    for i in range(iterations):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        take = np.concatenate([indices[case] for case in sampled])
        e0, e1 = nrmse(y[take], p0[take]), nrmse(y[take], p1[take])
        gains[i] = 100.0 * (e0 - e1) / max(e0, 1e-15)
    return float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))


def _target_uncertainty(cases: Iterable[CaseData], region: str, target: int) -> float:
    means, sems = [], []
    for case in cases:
        blocks = case.target_blocks[:, case.region[region], target]
        means.append(blocks.mean(axis=0))
        sems.append(blocks.std(axis=0, ddof=1) / math.sqrt(blocks.shape[0]))
    mean = np.concatenate(means)
    sem = np.concatenate(sems)
    return float(100.0 * np.sqrt(np.mean(sem**2)) / max(np.sqrt(np.mean(mean**2)), 1e-15))


def summarize(
    cases: list[CaseData],
    predictions: dict[str, dict[str, np.ndarray]],
    bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    meta = predictions["meta"]
    y_all = meta["truth"]
    rows: list[dict[str, object]] = []
    for region in ("nearfield", "protrusion"):
        mask = meta[region]
        for target_index, target in enumerate(TARGETS):
            y = y_all[mask, target_index]
            errors = {
                feature_set: nrmse(y, predictions[feature_set][target][mask])
                for feature_set in (*FEATURE_SETS, "S1_shuffled")
            }
            gain = 100.0 * (errors["S0"] - errors["S1"]) / max(errors["S0"], 1e-15)
            ci = _case_bootstrap_gain(
                y,
                predictions["S0"][target][mask],
                predictions["S1"][target][mask],
                meta["case_id"][mask],
                bootstrap,
                seed + 17 * target_index + (0 if region == "nearfield" else 100),
            )
            rows.append(
                {
                    "region": region,
                    "target": target,
                    "samples": int(mask.sum()),
                    "nrmse_S0": errors["S0"],
                    "nrmse_S1": errors["S1"],
                    "nrmse_S2": errors["S2"],
                    "nrmse_S1_shuffled": errors["S1_shuffled"],
                    "S1_gain_percent": gain,
                    "gain_ci95_low": ci[0],
                    "gain_ci95_high": ci[1],
                    "target_block_sem_percent": _target_uncertainty(cases, region, target_index),
                }
            )

    geometry_gain: dict[str, float] = {}
    mask_region = meta["nearfield"]
    for geometry in sorted(np.unique(meta["geometry"])):
        mask = mask_region & (meta["geometry"] == geometry)
        y = y_all[mask, 1]
        e0 = nrmse(y, predictions["S0"]["cf"][mask])
        e1 = nrmse(y, predictions["S1"]["cf"][mask])
        geometry_gain[str(geometry)] = 100.0 * (e0 - e1) / max(e0, 1e-15)

    primary = next(row for row in rows if row["region"] == "nearfield" and row["target"] == "cf")
    improvement_points = 100.0 * (float(primary["nrmse_S0"]) - float(primary["nrmse_S1"]))
    checks = {
        "relative_gain_at_least_20_percent": float(primary["S1_gain_percent"]) >= 20.0,
        "bootstrap_ci_excludes_zero": float(primary["gain_ci95_low"]) > 0.0,
        "gain_exceeds_block_uncertainty": improvement_points > float(primary["target_block_sem_percent"]),
        "aligned_beats_shuffled": float(primary["nrmse_S1"]) < float(primary["nrmse_S1_shuffled"]),
        "gain_positive_for_every_held_geometry": all(value > 0.0 for value in geometry_gain.values()),
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    return rows, {"verdict": verdict, "checks": checks, "geometry_shear_gain_percent": geometry_gain}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_figure(path: Path, rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    labels = ["S0 primitives", "S1 + Pij", "S2 + Pij + qi", "S1 shuffled"]
    keys = ["nrmse_S0", "nrmse_S1", "nrmse_S2", "nrmse_S1_shuffled"]
    colors = ["#777777", "#1677b8", "#e57c1f", "#b8b8b8"]
    for axis, target in zip(axes, TARGETS):
        row = next(item for item in rows if item["region"] == "nearfield" and item["target"] == target)
        values = [100.0 * float(row[key]) for key in keys]
        axis.bar(np.arange(4), values, color=colors)
        axis.set_xticks(np.arange(4), labels, rotation=25, ha="right")
        axis.set_ylabel("LOCO NRMSE [%]")
        axis.set_title("Wall pressure" if target == "cp" else "Signed wall shear")
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_report(
    path: Path,
    cases: list[CaseData],
    rows: list[dict[str, object]],
    decision: dict[str, object],
    train_region: str,
) -> None:
    lines = [
        "# SPARTA moment-sufficiency pilot gate",
        "",
        f"**Verdict: {decision['verdict']}**",
        "",
        "The four DSMC blocks were averaged before training and were used only to estimate sampling uncertainty. "
        "All scores are leave-one-complete-geometry-out (LOCO); no wall element from the held geometry enters training.",
        "",
        f"Training/evaluation support: `{train_region}` (`0.10 <= x <= 0.40 m`, including every protrusion element).",
        "",
        "| Region | Target | S0 NRMSE | S1 NRMSE | S2 NRMSE | S1 shuffled | S1 gain | 95% CI | DSMC block SEM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['region']} | {row['target']} | {100*float(row['nrmse_S0']):.2f}% | "
            f"{100*float(row['nrmse_S1']):.2f}% | {100*float(row['nrmse_S2']):.2f}% | "
            f"{100*float(row['nrmse_S1_shuffled']):.2f}% | {float(row['S1_gain_percent']):.2f}% | "
            f"[{float(row['gain_ci95_low']):.2f}, {float(row['gain_ci95_high']):.2f}]% | "
            f"{float(row['target_block_sem_percent']):.2f}% |"
        )
    lines.extend(["", "## Gate checks", ""])
    for name, value in decision["checks"].items():
        lines.append(f"- {'PASS' if value else 'FAIL'}: `{name}`")
    lines.extend(["", "Held-geometry signed-shear gains:", ""])
    for geometry, gain in decision["geometry_shear_gain_percent"].items():
        lines.append(f"- {geometry}: {gain:.2f}%")
    lines.extend(
        [
            "",
            "## Structural notes",
            "",
            "- All six cases contain four aligned grid and wall blocks and finite numeric values.",
            "- BWD has 88,001 grid rows because parent cell 4980 is split into two cut-cell subcells; `id+split` is stable across all blocks.",
            "- The data are stock-SPARTA reproductions of the archived parameter slice, not restart continuations of the private FPPC runs.",
            "- Surface tallies are targets only. The model samples bulk moments at 0.5, 1, and 2 mean-free-path distances along the gas-facing normal.",
            "- Case-level bootstrap resampling is intentionally conservative because only six statistically distinct cases are available.",
            "",
            "![Gate comparison](gate_nrmse.svg)",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Directory containing case_list.txt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-region", choices=("nearfield",), default="nearfield")
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    case_ids = [line.strip() for line in (args.root / "case_list.txt").read_text().splitlines() if line.strip()]
    cases = [load_case(args.root / case_id) for case_id in case_ids]
    predictions = loco_predictions(cases, args.train_region, args.trees, args.seed)
    rows, decision = summarize(cases, predictions, args.bootstrap, args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "gate_metrics.csv", rows)
    (args.output / "gate_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    _write_figure(args.output / "gate_nrmse.png", rows)
    _write_figure(args.output / "gate_nrmse.svg", rows)
    _write_report(args.output / "GATE_REPORT.md", cases, rows, decision, args.train_region)
    np.savez_compressed(
        args.output / "loco_predictions.npz",
        truth=predictions["meta"]["truth"],
        case_id=predictions["meta"]["case_id"],
        geometry=predictions["meta"]["geometry"],
        surface_id=predictions["meta"]["surface_id"],
        nearfield=predictions["meta"]["nearfield"],
        protrusion=predictions["meta"]["protrusion"],
        **{
            f"{feature_set}_{target}": predictions[feature_set][target]
            for feature_set in (*FEATURE_SETS, "S1_shuffled")
            for target in TARGETS
        },
    )
    print(json.dumps(decision, indent=2))
    print(f"REPORT={args.output / 'GATE_REPORT.md'}")


if __name__ == "__main__":
    main()
