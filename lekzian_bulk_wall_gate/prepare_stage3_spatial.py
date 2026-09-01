#!/usr/bin/env python3
"""Build wall-aligned spatial patches from the existing DSMC macroscopic fields.

No SPARTA/DSMC calculation is launched here.  The script reuses the audited
legacy readers, samples the already-saved ``(u, v, T, logP)`` field around each
wall point, and writes one compact NPZ dataset for Stage 3.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


FIELD_NAMES = ("u", "v", "T", "logP")
TARGET_NAMES = ("Cp", "Cq", "tau_abs")


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numeric_array(value, n_rows: int) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim != 2 or array.shape[0] != n_rows or array.shape[1] < 4:
        return None
    return array


def discover_gas_values(case) -> Tuple[np.ndarray, str]:
    """Find the legacy four-channel gas array without assuming one code version."""
    n_gas = len(np.asarray(case.gas_xy))
    for name in ("gas_z", "gas_values", "gas_fields", "field_values", "gas_features"):
        if hasattr(case, name):
            array = _numeric_array(getattr(case, name), n_gas)
            if array is not None:
                if array.shape[1] == 4:
                    return array, name
                # Some reader versions retain x,y before the four field values.
                xy = np.asarray(case.gas_xy, dtype=float)
                if array.shape[1] >= 6 and np.allclose(array[:, :2], xy, rtol=1e-5, atol=1e-8):
                    return array[:, 2:6], f"{name}[:,2:6]"
                return array[:, :4], f"{name}[:,:4]"

    for name, value in vars(case).items():
        if isinstance(value, pd.DataFrame) and len(value) == n_gas:
            lower = {str(column).lower(): column for column in value.columns}
            choices = []
            for aliases in (("u", "ux", "velx"), ("v", "uy", "vely"), ("t", "temp", "temperature")):
                column = next((lower[a] for a in aliases if a in lower), None)
                if column is None:
                    break
                choices.append(np.asarray(value[column], dtype=float))
            if len(choices) == 3:
                log_column = next((lower[a] for a in ("logp", "log_p", "lnp") if a in lower), None)
                p_column = next((lower[a] for a in ("p", "pressure") if a in lower), None)
                if log_column is not None:
                    choices.append(np.asarray(value[log_column], dtype=float))
                elif p_column is not None:
                    choices.append(np.log(np.maximum(np.asarray(value[p_column], dtype=float), 1e-30)))
                if len(choices) == 4:
                    return np.column_stack(choices), f"{name}[named columns]"

    candidates = []
    for name, value in vars(case).items():
        array = _numeric_array(value, n_gas)
        if array is not None and name != "gas_xy":
            candidates.append((name, array))
    if len(candidates) == 1:
        name, array = candidates[0]
        return array[:, :4], f"{name}[:,:4]"
    names = [name for name, _ in candidates]
    raise RuntimeError(
        "Could not identify the four existing macroscopic gas channels. "
        f"Candidate arrays were {names}; case attributes are {sorted(vars(case))}."
    )


def discover_wall_xy(case) -> Tuple[np.ndarray, str]:
    surface = case.surface
    if isinstance(surface, pd.DataFrame):
        lower = {str(column).lower(): column for column in surface.columns}
        pairs = (
            ("x", "y"), ("x_mid", "y_mid"), ("mid_x", "mid_y"),
            ("xc", "yc"), ("wall_x", "wall_y"),
        )
        for x_name, y_name in pairs:
            if x_name in lower and y_name in lower:
                xy = surface[[lower[x_name], lower[y_name]]].to_numpy(float)
                return xy, f"surface[{lower[x_name]},{lower[y_name]}]"
        # The audited Phase-1 surface workbooks store each wall panel by its
        # two vertices rather than by an explicit centre coordinate.  Targets
        # (Cp, Cq, shear) are panel-centred, so their collocated wall location
        # is the exact midpoint of the stored vertices.
        vertex_schemas = (
            ("v1x", "v1y", "v2x", "v2y"),
            ("x1", "y1", "x2", "y2"),
        )
        for x1, y1, x2, y2 in vertex_schemas:
            if all(name in lower for name in (x1, y1, x2, y2)):
                first = surface[[lower[x1], lower[y1]]].to_numpy(float)
                second = surface[[lower[x2], lower[y2]]].to_numpy(float)
                xy = 0.5 * (first + second)
                if not np.isfinite(xy).all():
                    raise RuntimeError("Non-finite wall-panel midpoint coordinates were found.")
                return xy, f"midpoint(surface[{x1},{y1}],surface[{x2},{y2}])"
    for name in ("wall_xy", "surface_xy", "surf_xy"):
        if hasattr(case, name):
            xy = np.asarray(getattr(case, name), dtype=float)
            if xy.ndim == 2 and xy.shape == (len(case.surface), 2):
                return xy, name
    raise RuntimeError(
        "Could not identify wall coordinates. Available surface columns are "
        f"{list(getattr(surface, 'columns', []))}."
    )


def ordered_frames(wall_xy: np.ndarray, s01: np.ndarray, gas_xy: np.ndarray, hs: float):
    """Return unit tangent/normal arrays; normals point toward available gas."""
    order = np.argsort(s01)
    ordered = wall_xy[order]
    tangent_ordered = np.gradient(ordered, axis=0)
    norms = np.linalg.norm(tangent_ordered, axis=1, keepdims=True)
    tangent_ordered /= np.maximum(norms, 1e-12)
    tangent = np.empty_like(tangent_ordered)
    tangent[order] = tangent_ordered
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))

    tree = cKDTree(gas_xy)
    epsilon = max(0.08 * float(hs), 1e-10)
    plus = wall_xy + epsilon * normal
    minus = wall_xy - epsilon * normal
    d_plus = tree.query(plus, k=1)[0]
    d_minus = tree.query(minus, k=1)[0]
    # Orient each point locally, then smooth isolated flips along the wall.
    flip = d_minus < d_plus
    normal[flip] *= -1.0
    normal_ordered = normal[order]
    for i in range(1, len(normal_ordered)):
        if np.dot(normal_ordered[i], normal_ordered[i - 1]) < 0:
            normal_ordered[i] *= -1.0
    normal[order] = normal_ordered
    return tangent.astype(np.float32), normal.astype(np.float32)


def sample_wall_patches(
    gas_xy: np.ndarray,
    gas_values: np.ndarray,
    wall_xy: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    hs: float,
    t_grid: np.ndarray,
    n_grid: np.ndarray,
    neighbours: int = 4,
    valid_distance_hs: float = 0.35,
) -> np.ndarray:
    """Inverse-distance interpolate existing fields onto wall-aligned grids."""
    gas_xy = np.asarray(gas_xy, dtype=float)
    gas_values = np.asarray(gas_values, dtype=float)
    finite = np.isfinite(gas_xy).all(axis=1) & np.isfinite(gas_values).all(axis=1)
    gas_xy, gas_values = gas_xy[finite], gas_values[finite]
    if len(gas_xy) < neighbours:
        raise RuntimeError("Too few finite gas points for spatial interpolation.")
    tree = cKDTree(gas_xy)
    tt, nn = np.meshgrid(t_grid, n_grid)
    offsets = np.column_stack((tt.ravel(), nn.ravel()))
    result = np.empty((len(wall_xy), 6, len(n_grid), len(t_grid)), dtype=np.float32)
    for i in range(len(wall_xy)):
        query = wall_xy[i] + float(hs) * (
            offsets[:, :1] * tangent[i] + offsets[:, 1:] * normal[i]
        )
        distance, index = tree.query(query, k=neighbours)
        if neighbours == 1:
            distance, index = distance[:, None], index[:, None]
        weight = 1.0 / np.maximum(distance, max(abs(float(hs)), 1e-12) * 1e-6) ** 2
        values = np.sum(gas_values[index] * weight[..., None], axis=1) / np.sum(weight, axis=1)[:, None]
        nearest_hs = distance[:, 0] / max(abs(float(hs)), 1e-12)
        valid = nearest_hs <= valid_distance_hs
        patch = np.column_stack((values, nearest_hs, valid.astype(float)))
        result[i] = patch.T.reshape(6, len(n_grid), len(t_grid))
    return result


def direct_wall_weights(surface_base, surface: pd.DataFrame, targets: np.ndarray) -> np.ndarray:
    """Use the legacy wall weights exactly once (Stage 2 accidentally boosted twice)."""
    settings = SimpleNamespace(
        surface_apex_weight=2.0,
        surface_peak_weight=2.0,
        surface_peak_sigma=0.06,
        surface_weight_clip=8.0,
    )
    try:
        weights = np.asarray(
            surface_base.compute_surface_weights(surface, targets, settings), dtype=float
        ).reshape(-1)
    except (AttributeError, TypeError):
        weights = np.ones(len(surface), dtype=float)
    weights = np.clip(weights, 0.5, 8.0)
    return (weights / np.mean(weights)).astype(np.float32)


def surface_order(surface: pd.DataFrame) -> np.ndarray:
    if "s01" not in surface:
        raise RuntimeError("Legacy surface table has no s01 coordinate.")
    return np.asarray(surface["s01"], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--nonlocal-script", required=True)
    parser.add_argument("--field-base-script", required=True)
    parser.add_argument("--surface-base-script", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-gas-points", type=int, default=60000)
    parser.add_argument("--tangential-points", type=int, default=21)
    parser.add_argument("--normal-points", type=int, default=11)
    parser.add_argument("--tangential-extent-hs", type=float, default=3.0)
    parser.add_argument("--normal-extent-hs", type=float, default=3.0)
    parser.add_argument("--valid-distance-hs", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "stage3_spatial_dataset_phase1.npz"
    complete_path = out_dir / "STAGE3_DATASET_COMPLETE"
    if args.resume and dataset_path.exists() and complete_path.exists():
        print(f"[SKIP] existing Stage-3 dataset: {dataset_path}", flush=True)
        return
    if args.tangential_points < 5 or args.normal_points < 3:
        raise ValueError("Spatial grid is too small.")

    script_paths = {
        "nonlocal": Path(args.nonlocal_script).expanduser().resolve(),
        "field": Path(args.field_base_script).expanduser().resolve(),
        "surface": Path(args.surface_base_script).expanduser().resolve(),
    }
    nonlocal_base = import_module(script_paths["nonlocal"], "lekzian_s3_nonlocal")
    field_base = import_module(script_paths["field"], "lekzian_s3_field")
    surface_base = import_module(script_paths["surface"], "lekzian_s3_surface")
    manifest = field_base.load_manifest(Path(args.audit_dir).expanduser().resolve())
    manifest = nonlocal_base.get_phase1_cases(manifest)
    if len(manifest) != 27:
        raise RuntimeError(f"Stage 3 requires the 27 Phase-1 cases; found {len(manifest)}.")

    t_grid = np.linspace(
        -args.tangential_extent_hs, args.tangential_extent_hs, args.tangential_points
    ).astype(np.float32)
    n_grid = np.linspace(
        0.05, args.normal_extent_hs, args.normal_points
    ).astype(np.float32)
    rng = np.random.default_rng(args.seed)
    legacy_args = SimpleNamespace(max_gas_points=args.max_gas_points)
    blocks = {name: [] for name in (
        "patches", "context", "targets", "weights", "case_id", "surface_i", "s01",
        "Ma", "Kn", "geom", "hphs", "TwTinf", "hs", "hp", "wall_xy", "tangent", "normal",
    )}
    case_records = []
    gas_sources = set()
    wall_sources = set()

    for _, row in manifest.iterrows():
        case = nonlocal_base.read_case_data(field_base, surface_base, row, legacy_args, rng)
        if case is None:
            continue
        gas_xy = np.asarray(case.gas_xy, dtype=float)
        gas_values, gas_source = discover_gas_values(case)
        wall_xy, wall_source = discover_wall_xy(case)
        s01 = surface_order(case.surface)
        tangent, normal = ordered_frames(wall_xy, s01, gas_xy, case.hs)
        patches = sample_wall_patches(
            gas_xy, gas_values, wall_xy, tangent, normal, case.hs, t_grid, n_grid,
            valid_distance_hs=args.valid_distance_hs,
        )
        geom = surface_base.geometry_from_row(row)
        case_features = np.asarray(surface_base.case_features(row), dtype=np.float32).reshape(-1)
        surface_features = np.asarray(
            surface_base.build_surface_point_features(case.surface, row=row, geom=geom),
            dtype=np.float32,
        )
        if surface_features.shape[0] != len(wall_xy):
            raise RuntimeError(f"Surface feature alignment failed for {case.case_id}.")
        context = np.column_stack((
            np.repeat(case_features[None, :], len(wall_xy), axis=0), surface_features
        )).astype(np.float32)
        targets = np.asarray(case.targets, dtype=np.float32)
        if targets.shape != (len(wall_xy), 3):
            raise RuntimeError(f"Unexpected target shape {targets.shape} for {case.case_id}.")
        weights = direct_wall_weights(surface_base, case.surface, targets)
        n = len(wall_xy)
        metadata = {
            "case_id": np.repeat(str(case.case_id), n),
            "surface_i": np.arange(n, dtype=np.int32),
            "s01": s01.astype(np.float32),
            "Ma": np.repeat(float(case.Ma), n).astype(np.float32),
            "Kn": np.repeat(float(case.Kn), n).astype(np.float32),
            "geom": np.repeat(str(case.geom), n),
            "hphs": np.repeat(float(case.hphs), n).astype(np.float32),
            "TwTinf": np.repeat(float(case.TwTinf), n).astype(np.float32),
            "hs": np.repeat(float(case.hs), n).astype(np.float32),
            "hp": np.repeat(float(case.hp), n).astype(np.float32),
        }
        blocks["patches"].append(patches)
        blocks["context"].append(context)
        blocks["targets"].append(targets)
        blocks["weights"].append(weights)
        blocks["wall_xy"].append(wall_xy.astype(np.float32))
        blocks["tangent"].append(tangent)
        blocks["normal"].append(normal)
        for name, value in metadata.items():
            blocks[name].append(value)
        gas_sources.add(gas_source)
        wall_sources.add(wall_source)
        case_records.append({
            "case_id": str(case.case_id), "Ma": float(case.Ma), "Kn": float(case.Kn),
            "geom": str(case.geom), "n_gas": len(gas_xy), "n_surface": n,
            "valid_patch_fraction": float(np.mean(patches[:, 5] > 0.5)),
        })
        print(
            f"[INFO] {case.case_id}: gas={len(gas_xy)}, wall={n}, "
            f"patch-valid={case_records[-1]['valid_patch_fraction']:.3f}", flush=True,
        )

    if len(case_records) != 27:
        raise RuntimeError(f"Only {len(case_records)} of 27 Phase-1 cases were readable.")
    arrays = {}
    for name, pieces in blocks.items():
        arrays[name] = np.concatenate(pieces, axis=0)
    arrays.update({
        "t_grid": t_grid,
        "n_grid": n_grid,
        "field_names": np.asarray((*FIELD_NAMES, "nearest_distance_hs", "valid")),
        "target_names": np.asarray(TARGET_NAMES),
        "target_scale_exponents": np.asarray((2.0, 3.0, 2.0), dtype=np.float32),
    })
    np.savez_compressed(dataset_path, **arrays)
    pd.DataFrame(case_records).to_csv(out_dir / "stage3_case_table_phase1.csv", index=False)
    record = vars(args).copy()
    record.update({
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "n_cases": len(case_records),
        "n_surface_samples": int(len(arrays["targets"])),
        "patch_shape": list(arrays["patches"].shape[1:]),
        "gas_value_sources": sorted(gas_sources),
        "wall_coordinate_sources": sorted(wall_sources),
        "legacy_script_sha256": {name: sha256_file(path) for name, path in script_paths.items()},
        "new_dsmc_runs": 0,
        "higher_order_moments": False,
    })
    (out_dir / "stage3_extraction_config.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    complete_path.write_text(record["dataset_sha256"] + "\n", encoding="utf-8")
    print(f"[DONE] Stage-3 spatial dataset: {dataset_path}", flush=True)


if __name__ == "__main__":
    main()
