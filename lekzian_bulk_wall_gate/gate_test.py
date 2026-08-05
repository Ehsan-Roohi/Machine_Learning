#!/usr/bin/env python3
"""Matched-capacity gate test for the Lekzian--Roohi bulk-to-wall study.

The script consumes the *full annular descriptor* table produced from the 27
Phase-1 DSMC cases.  Every candidate receives the exact engineered case/surface
features and branch/trunk/gated-decoder architecture of the strong direct-wall
surrogate, plus the same fixed-width bulk trunk. Unobserved annuli are replaced
by training-set medians (zero after standardisation) and explicit masks.

Models/configurations
---------------------
M0          : parameters/geometry/surface coordinates only
M_shuffled  : all annular descriptors independently permuted (capacity control)
M_R...      : real annular descriptors available only through radius R/h_s
M_full      : every annular descriptor

Outer validation is grouped by physical case (LOCO) and by the (Ma, Kn) pair.
An inner group is held out for early stopping.  Uncertainty is obtained by
bootstrapping physical cases, never trees or surface samples.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError:  # Schema/decision tests can run without the GPU stack.
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None


TARGETS = ("Cp", "Cq", "tau_abs")
ID_COLUMNS = {"case_id", "surface_i", "geom", "config", "gate_sample_weight"}
RING_RE = re.compile(r"^(ring_([^_]+)_([^_]+))_")


def _parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_target_map(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        name, value = token.split("=", 1)
        out[name.strip()] = float(value)
    missing = set(TARGETS) - set(out)
    if missing:
        raise ValueError(f"Missing target tolerances: {sorted(missing)}")
    return out


def _ring_number(token: str) -> float:
    if token.lower() == "inf":
        return math.inf
    return float(token.replace("p", "."))


def _radius_label(radius: float) -> str:
    value = f"{radius:g}".replace(".", "p")
    return f"M_R{value}"


def _stable_seed(*parts: object) -> int:
    text = "|".join(str(x) for x in parts).encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:8], 16)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rel_l2(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> float:
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(true)
    if ok.sum() < 2:
        return float("nan")
    return float(np.linalg.norm(pred[ok] - true[ok]) / (np.linalg.norm(true[ok]) + eps))


def range_mae_pct(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> float:
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(true)
    if ok.sum() < 3:
        return float("nan")
    lo, hi = np.percentile(true[ok], [1.0, 99.0])
    return float(100.0 * np.mean(np.abs(pred[ok] - true[ok])) / max(float(hi - lo), eps))


@dataclass(frozen=True)
class FeatureSchema:
    case_columns: Tuple[str, ...]
    surface_columns: Tuple[str, ...]
    bulk_columns: Tuple[str, ...]
    ring_names: Tuple[str, ...]
    ring_lower: Tuple[float, ...]
    ring_upper: Tuple[float, ...]
    bulk_ring_index: Tuple[int, ...]

    @property
    def input_dimension(self) -> int:
        return (
            len(self.case_columns)
            + len(self.surface_columns)
            + len(self.bulk_columns)
            + len(self.ring_names)
        )

    @property
    def case_dimension(self) -> int:
        return len(self.case_columns)

    @property
    def trunk_dimension(self) -> int:
        return len(self.surface_columns) + len(self.bulk_columns) + len(self.ring_names)

    @property
    def base_columns(self) -> Tuple[str, ...]:
        return self.case_columns + self.surface_columns


def infer_schema(df: pd.DataFrame) -> FeatureSchema:
    ring_to_cols: Dict[str, List[str]] = {}
    ring_bounds: Dict[str, Tuple[float, float]] = {}
    for col in df.columns:
        match = RING_RE.match(str(col))
        if not match:
            continue
        ring, low, high = match.groups()
        ring_to_cols.setdefault(ring, []).append(str(col))
        ring_bounds[ring] = (_ring_number(low), _ring_number(high))

    if not ring_to_cols:
        raise ValueError(
            "No full-ring descriptors found. Expected columns such as "
            "ring_0p0_0p1_u_mean. Use surface_patch_dataset_full_gate.csv."
        )

    rings = sorted(ring_to_cols, key=lambda x: ring_bounds[x][0])
    bulk_columns: List[str] = []
    bulk_ring_index: List[int] = []
    for ring_i, ring in enumerate(rings):
        for col in sorted(ring_to_cols[ring]):
            bulk_columns.append(col)
            bulk_ring_index.append(ring_i)

    case_columns = tuple(str(c) for c in df.columns if str(c).startswith("operator_case_f"))
    surface_columns = tuple(str(c) for c in df.columns if str(c).startswith("operator_surface_f"))
    if len(case_columns) < 7 or not surface_columns:
        raise ValueError(
            "The feature table lacks the exact case/surface representation of the strong direct-wall operator. "
            "Rebuild it with prepare_gate_features.py; a legacy full-ring table is not sufficient for this Gate Test."
        )

    return FeatureSchema(
        case_columns=case_columns,
        surface_columns=surface_columns,
        bulk_columns=tuple(bulk_columns),
        ring_names=tuple(rings),
        ring_lower=tuple(ring_bounds[r][0] for r in rings),
        ring_upper=tuple(ring_bounds[r][1] for r in rings),
        bulk_ring_index=tuple(bulk_ring_index),
    )


def validate_table(df: pd.DataFrame, schema: FeatureSchema) -> None:
    missing = set(TARGETS) | {"case_id", "Ma", "Kn", "gate_sample_weight"}
    missing -= set(df.columns)
    if missing:
        raise ValueError(f"Feature table is missing columns: {sorted(missing)}")
    n_cases = df["case_id"].astype(str).nunique()
    if n_cases != 27:
        print(f"[WARN] expected 27 Phase-1 cases, found {n_cases}", flush=True)
    if len(schema.ring_names) < 2:
        raise ValueError("At least two annular blocks are required for a spatial-footprint test.")
    for target in TARGETS:
        if pd.to_numeric(df[target], errors="coerce").notna().sum() < 10:
            raise ValueError(f"Target {target} has too few finite values.")


def build_configs(schema: FeatureSchema, radii: Sequence[float]) -> List[str]:
    configs = ["M0", "M_shuffled"]
    finite_uppers = [x for x in schema.ring_upper if np.isfinite(x)]
    for radius in sorted(set(float(r) for r in radii)):
        if not finite_uppers or radius + 1e-12 < min(finite_uppers):
            raise ValueError(f"Radius {radius:g} is below the first annular boundary.")
        configs.append(_radius_label(radius))
    configs.append("M_full")
    return configs


def visible_rings(config: str, schema: FeatureSchema) -> np.ndarray:
    if config == "M0":
        return np.zeros(len(schema.ring_names), dtype=bool)
    if config in {"M_full", "M_shuffled"}:
        return np.ones(len(schema.ring_names), dtype=bool)
    if not config.startswith("M_R"):
        raise ValueError(f"Unknown configuration: {config}")
    radius = float(config[3:].replace("p", "."))
    return np.asarray([upper <= radius + 1e-12 for upper in schema.ring_upper], dtype=bool)


@dataclass
class Preprocessor:
    base_median: np.ndarray
    base_mean: np.ndarray
    base_std: np.ndarray
    bulk_median: np.ndarray
    bulk_mean: np.ndarray
    bulk_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray


def _finite_stats(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float).copy()
    x[~np.isfinite(x)] = np.nan
    median = np.nanmedian(x, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    bad = np.where(~np.isfinite(x))
    x[bad] = np.take(median, bad[1])
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return median, mean, std


def fit_preprocessor(train_df: pd.DataFrame, schema: FeatureSchema) -> Preprocessor:
    bmed, bmean, bstd = _finite_stats(train_df[list(schema.base_columns)].to_numpy(float))
    fmed, fmean, fstd = _finite_stats(train_df[list(schema.bulk_columns)].to_numpy(float))
    y = train_df[list(TARGETS)].to_numpy(float)
    if not np.isfinite(y).all():
        raise ValueError("Targets contain NaN/Inf in the outer training fold.")
    ymean = np.mean(y, axis=0)
    ystd = np.std(y, axis=0)
    ystd = np.where(ystd < 1e-8, 1.0, ystd)
    return Preprocessor(bmed, bmean, bstd, fmed, fmean, fstd, ymean, ystd)


def _standardise(values: np.ndarray, median: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float).copy()
    x[~np.isfinite(x)] = np.nan
    bad = np.where(~np.isfinite(x))
    x[bad] = np.take(median, bad[1])
    return ((x - mean) / std).astype(np.float32)


def transform_features(
    df: pd.DataFrame,
    schema: FeatureSchema,
    prep: Preprocessor,
    config: str,
    permutation_seed: Optional[int] = None,
) -> np.ndarray:
    base = _standardise(
        df[list(schema.base_columns)].to_numpy(float), prep.base_median, prep.base_mean, prep.base_std
    )
    bulk = _standardise(
        df[list(schema.bulk_columns)].to_numpy(float), prep.bulk_median, prep.bulk_mean, prep.bulk_std
    )
    mask = visible_rings(config, schema)
    col_ring = np.asarray(schema.bulk_ring_index, dtype=int)
    bulk[:, ~mask[col_ring]] = 0.0

    if config == "M_shuffled":
        if permutation_seed is None:
            raise ValueError("M_shuffled requires a deterministic permutation seed.")
        rng = np.random.default_rng(permutation_seed)
        # Independent column permutations destroy case/surface alignment while
        # preserving every descriptor's marginal distribution and input width.
        for j in range(bulk.shape[1]):
            bulk[:, j] = bulk[rng.permutation(len(bulk)), j]

    mask_block = np.repeat(mask.astype(np.float32)[None, :], len(df), axis=0)
    x = np.column_stack([base, bulk, mask_block]).astype(np.float32)
    if x.shape[1] != schema.input_dimension:
        raise AssertionError("Configuration changed input dimension.")
    return x


if nn is not None:
    class MLP(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int, dropout: float):
            super().__init__()
            layers: List[nn.Module] = []
            width_in = in_dim
            for _ in range(depth):
                layers.extend([nn.Linear(width_in, hidden), nn.SiLU()])
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                width_in = hidden
            layers.append(nn.Linear(width_in, out_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)


    class FixedCapacitySurfaceOperator(nn.Module):
        """The strong surface-only operator with a fixed-width bulk-aware trunk."""

        def __init__(
            self,
            case_dim: int,
            trunk_dim: int,
            latent: int,
            hidden: int,
            depth: int,
            dropout: float,
            full_residual_scale: float,
        ):
            super().__init__()
            if case_dim < 7:
                raise ValueError("The direct-wall operator requires at least seven ordered case features.")
            self.case_dim = int(case_dim)
            self.full_residual_scale = float(full_residual_scale)
            self.base_branch = MLP(5, latent, hidden, depth, dropout)
            self.h_branch = MLP(4, latent, hidden, depth, dropout)
            self.tw_branch = MLP(4, latent, hidden, depth, dropout)
            self.full_branch = MLP(case_dim, latent, hidden, max(1, depth - 1), dropout)
            self.trunk = MLP(trunk_dim, latent, hidden, depth, dropout)
            self.wind_decoder = MLP(latent, len(TARGETS), hidden, depth, dropout)
            self.lee_decoder = MLP(latent, len(TARGETS), hidden, depth, dropout)
            self.gate = MLP(trunk_dim, 1, max(32, hidden // 2), 2, 0.0)

        def branch_embedding(self, case_x: "torch.Tensor") -> "torch.Tensor":
            ma_log_geom = torch.cat([case_x[:, 0:2], case_x[:, 4:7]], dim=1)
            log_geom = torch.cat([case_x[:, 1:2], case_x[:, 4:7]], dim=1)
            h_delta = case_x[:, 2:3]
            tw_delta = case_x[:, 3:4]
            z = self.base_branch(ma_log_geom)
            z = z + h_delta * self.h_branch(log_geom)
            z = z + tw_delta * self.tw_branch(log_geom)
            if self.full_residual_scale > 0:
                z = z + self.full_residual_scale * self.full_branch(case_x)
            return z

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            case_x = x[:, : self.case_dim]
            trunk_x = x[:, self.case_dim :]
            z = self.branch_embedding(case_x) * self.trunk(trunk_x)
            wind = self.wind_decoder(z)
            lee = self.lee_decoder(z)
            gate = torch.sigmoid(self.gate(trunk_x))
            return gate * wind + (1.0 - gate) * lee
else:
    class FixedCapacitySurfaceOperator:  # pragma: no cover - clear runtime error only.
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required to train the matched-capacity Gate Test.")


def _resolve_device(name: str) -> torch.device:
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Use the Unity dsmc-gpu environment or install requirements.txt.")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    w_val: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    case_dim: int,
    trunk_dim: int,
    latent: int,
    hidden: int,
    depth: int,
    dropout: float,
    full_residual_scale: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> Tuple[np.ndarray, int, float, int]:
    _seed_everything(seed)
    model = FixedCapacitySurfaceOperator(
        case_dim=case_dim,
        trunk_dim=trunk_dim,
        latent=latent,
        hidden=hidden,
        depth=depth,
        dropout=dropout,
        full_residual_scale=full_residual_scale,
    ).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(w_train)),
        batch_size=min(batch_size, len(x_train)),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    xv = torch.from_numpy(x_val).to(device)
    yv = torch.from_numpy(y_val).to(device)
    wv = torch.from_numpy(w_val).to(device)

    best_loss = math.inf
    best_epoch = 0
    best_state: Optional[Mapping[str, torch.Tensor]] = None
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb, wb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            element = F.huber_loss(model(xb), yb, delta=1.0, reduction="none").mean(dim=1, keepdim=True)
            loss = (element * wb).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            element = F.huber_loss(model(xv), yv, delta=1.0, reduction="none").mean(dim=1, keepdim=True)
            value = float((element * wv).mean().cpu())
        if value < best_loss - 1e-6:
            best_loss = value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    pred_parts: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_test), max(batch_size, 4096)):
            xt = torch.from_numpy(x_test[start : start + max(batch_size, 4096)]).to(device)
            pred_parts.append(model(xt).cpu().numpy())
    return np.vstack(pred_parts), best_epoch, best_loss, parameter_count


def _outer_groups(df: pd.DataFrame, scheme: str) -> pd.Series:
    if scheme == "loco":
        return df["case_id"].astype(str)
    if scheme == "pairout":
        return df.apply(lambda r: f"Ma{float(r['Ma']):g}_Kn{float(r['Kn']):g}", axis=1)
    raise ValueError(f"Unknown CV scheme: {scheme}")


def _choose_inner_group(groups: pd.Series, outer_group: str, seed: int) -> str:
    candidates = sorted(set(groups.astype(str)) - {str(outer_group)})
    if len(candidates) < 2:
        raise ValueError("Too few training groups for nested validation.")
    return candidates[_stable_seed("inner", outer_group, seed) % len(candidates)]


def _append_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _task_key(scheme: str, outer_group: str, seed: int, config: str) -> str:
    return f"{scheme}|{outer_group}|{seed}|{config}"


def run_experiment(df: pd.DataFrame, schema: FeatureSchema, args: argparse.Namespace) -> pd.DataFrame:
    out_dir = Path(args.out)
    partial_metrics = out_dir / "case_metrics.partial.csv"
    partial_predictions = out_dir / "surface_predictions.partial.csv"
    completed_path = out_dir / "completed_tasks.txt"
    completed: set[str] = set()
    if args.resume and completed_path.exists():
        completed = {x.strip() for x in completed_path.read_text().splitlines() if x.strip()}

    configs = build_configs(schema, args.radii_list)
    device = _resolve_device(args.device)
    print(f"[INFO] device={device}; fixed input dimension={schema.input_dimension}", flush=True)
    print(f"[INFO] configs={configs}", flush=True)

    schemes = args.cv_list
    if args.only_scheme:
        if args.only_scheme not in schemes:
            raise ValueError(f"--only-scheme={args.only_scheme} is not present in --cv")
        schemes = [args.only_scheme]
    for scheme in schemes:
        groups = _outer_groups(df, scheme)
        outer_values = sorted(groups.unique())
        if args.only_outer_group:
            if args.only_outer_group not in outer_values:
                raise ValueError(
                    f"Outer group {args.only_outer_group!r} not found for {scheme}; available={outer_values}"
                )
            outer_values = [args.only_outer_group]
        if args.limit_folds > 0:
            outer_values = outer_values[: args.limit_folds]
        for fold_i, outer_group in enumerate(outer_values, 1):
            test_mask = groups == outer_group
            outer_train = df.loc[~test_mask].reset_index(drop=True)
            test_df = df.loc[test_mask].reset_index(drop=True)
            train_groups = _outer_groups(outer_train, scheme)
            for seed in args.seeds_list:
                inner_group = _choose_inner_group(train_groups, str(outer_group), seed)
                val_mask = train_groups.astype(str) == inner_group
                train_df = outer_train.loc[~val_mask].reset_index(drop=True)
                val_df = outer_train.loc[val_mask].reset_index(drop=True)
                prep = fit_preprocessor(train_df, schema)
                ytr = ((train_df[list(TARGETS)].to_numpy(float) - prep.y_mean) / prep.y_std).astype(np.float32)
                yva = ((val_df[list(TARGETS)].to_numpy(float) - prep.y_mean) / prep.y_std).astype(np.float32)
                wtr = train_df["gate_sample_weight"].to_numpy(np.float32).reshape(-1, 1)
                wva = val_df["gate_sample_weight"].to_numpy(np.float32).reshape(-1, 1)

                for config in configs:
                    task = _task_key(scheme, str(outer_group), seed, config)
                    if task in completed:
                        print(f"[SKIP] {task}", flush=True)
                        continue
                    print(
                        f"[RUN] {scheme} fold {fold_i}/{len(outer_values)} outer={outer_group} "
                        f"inner={inner_group} seed={seed} config={config}",
                        flush=True,
                    )
                    perm_base = _stable_seed("shuffle", scheme, outer_group, seed, config)
                    xtr = transform_features(train_df, schema, prep, config, perm_base + 1)
                    xva = transform_features(val_df, schema, prep, config, perm_base + 2)
                    xte = transform_features(test_df, schema, prep, config, perm_base + 3)
                    # Identical initial weights for paired configurations.
                    model_seed = _stable_seed("model", scheme, outer_group, seed)
                    pred_z, best_epoch, best_loss, n_parameters = train_predict(
                        xtr,
                        ytr,
                        wtr,
                        xva,
                        yva,
                        wva,
                        xte,
                        seed=model_seed,
                        device=device,
                        case_dim=schema.case_dimension,
                        trunk_dim=schema.trunk_dimension,
                        latent=args.latent,
                        hidden=args.hidden,
                        depth=args.depth,
                        dropout=args.dropout,
                        full_residual_scale=args.full_residual_scale,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        patience=args.patience,
                    )
                    pred = pred_z * prep.y_std[None, :] + prep.y_mean[None, :]
                    truth = test_df[list(TARGETS)].to_numpy(float)

                    metrics_rows: List[Dict[str, object]] = []
                    prediction_rows: List[Dict[str, object]] = []
                    for case_id, loc in test_df.groupby(test_df["case_id"].astype(str)).groups.items():
                        idx = np.asarray(list(loc), dtype=int)
                        for j, target in enumerate(TARGETS):
                            metrics_rows.append(
                                {
                                    "scheme": scheme,
                                    "outer_group": outer_group,
                                    "inner_group": inner_group,
                                    "seed": seed,
                                    "config": config,
                                    "case_id": case_id,
                                    "target": target,
                                    "relL2": rel_l2(pred[idx, j], truth[idx, j]),
                                    "rangeMAE_pct": range_mae_pct(pred[idx, j], truth[idx, j]),
                                    "best_epoch": best_epoch,
                                    "inner_val_loss": best_loss,
                                    "input_dimension": schema.input_dimension,
                                    "model_parameters": n_parameters,
                                }
                            )
                    for row_i, row in test_df.reset_index(drop=True).iterrows():
                        item: Dict[str, object] = {
                            "scheme": scheme,
                            "outer_group": outer_group,
                            "seed": seed,
                            "config": config,
                            "case_id": str(row["case_id"]),
                            "surface_i": int(row.get("surface_i", row_i)),
                            "s01": float(row.get("s01", np.nan)),
                        }
                        for j, target in enumerate(TARGETS):
                            item[f"true_{target}"] = float(truth[row_i, j])
                            item[f"pred_{target}"] = float(pred[row_i, j])
                        prediction_rows.append(item)

                    _append_csv(partial_metrics, pd.DataFrame(metrics_rows))
                    if not args.no_surface_predictions:
                        _append_csv(partial_predictions, pd.DataFrame(prediction_rows))
                    with completed_path.open("a") as handle:
                        handle.write(task + "\n")
                    completed.add(task)

    if not partial_metrics.exists():
        raise RuntimeError("No metrics were produced.")
    metrics = pd.read_csv(partial_metrics)
    metrics.to_csv(out_dir / "case_metrics.csv", index=False)
    if partial_predictions.exists():
        pd.read_csv(partial_predictions).to_csv(out_dir / "surface_predictions.csv", index=False)
    return metrics


def aggregate_task_results(task_root: Path, out_dir: Path, save_predictions: bool) -> pd.DataFrame:
    metric_paths = sorted(task_root.glob("task_*/case_metrics.csv"))
    if not metric_paths:
        raise RuntimeError(f"No completed task metrics found under {task_root}")
    metrics = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
    task_cols = ["scheme", "outer_group", "seed", "config", "case_id", "target"]
    duplicate_count = int(metrics.duplicated(task_cols).sum())
    if duplicate_count:
        raise RuntimeError(f"Aggregation found {duplicate_count} duplicate case-metric rows.")
    counts = metrics.groupby(["scheme", "outer_group", "seed"])["config"].nunique()
    if counts.nunique() != 1 or int(counts.iloc[0]) < 3:
        raise RuntimeError("At least one array task has an incomplete configuration set.")
    metrics.to_csv(out_dir / "case_metrics.csv", index=False)

    if save_predictions:
        prediction_paths = sorted(task_root.glob("task_*/surface_predictions.csv"))
        if len(prediction_paths) != len(metric_paths):
            raise RuntimeError("Some array tasks are missing surface_predictions.csv.")
        predictions = pd.concat([pd.read_csv(path) for path in prediction_paths], ignore_index=True)
        predictions.to_csv(out_dir / "surface_predictions.csv", index=False)
    return metrics


def _case_bootstrap(values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(x))
    if len(x) == 1 or n_boot <= 0:
        return point, point, point
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    chunk = 2000
    for start in range(0, n_boot, chunk):
        stop = min(n_boot, start + chunk)
        draws = rng.integers(0, len(x), size=(stop - start, len(x)))
        means[start:stop] = x[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return point, float(lo), float(hi)


def summarise_metrics(metrics: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    by_case = (
        metrics.groupby(["scheme", "config", "target", "case_id"], as_index=False)
        .agg(relL2=("relL2", "mean"), rangeMAE_pct=("rangeMAE_pct", "mean"))
    )
    summary_rows: List[Dict[str, object]] = []
    for (scheme, config, target), group in by_case.groupby(["scheme", "config", "target"]):
        mean, lo, hi = _case_bootstrap(
            group["relL2"].to_numpy(float), args.bootstrap, _stable_seed("summary", scheme, config, target)
        )
        summary_rows.append(
            {
                "scheme": scheme,
                "config": config,
                "target": target,
                "n_physical_cases": group["case_id"].nunique(),
                "mean_relL2": mean,
                "median_relL2": float(group["relL2"].median()),
                "case_bootstrap_CI2p5": lo,
                "case_bootstrap_CI97p5": hi,
                "mean_rangeMAE_pct": float(group["rangeMAE_pct"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)

    gain_rows: List[Dict[str, object]] = []
    for scheme in sorted(by_case["scheme"].unique()):
        for target in TARGETS:
            sub = by_case[(by_case["scheme"] == scheme) & (by_case["target"] == target)]
            pivot = sub.pivot(index="case_id", columns="config", values="relL2")
            for baseline in ["M0", "M_shuffled"]:
                if baseline not in pivot or "M_full" not in pivot:
                    continue
                diff = (pivot[baseline] - pivot["M_full"]).dropna().to_numpy(float)
                point, lo, hi = _case_bootstrap(
                    diff, args.bootstrap, _stable_seed("gain", scheme, target, baseline)
                )
                gain_rows.append(
                    {
                        "scheme": scheme,
                        "target": target,
                        "comparison": f"{baseline}-M_full",
                        "n_physical_cases": len(diff),
                        "mean_gain": point,
                        "mean_gain_percentage_points": 100.0 * point,
                        "case_bootstrap_CI2p5": lo,
                        "case_bootstrap_CI97p5": hi,
                        "CI2p5_percentage_points": 100.0 * lo,
                        "CI97p5_percentage_points": 100.0 * hi,
                    }
                )
    gains = pd.DataFrame(gain_rows)
    gains.to_csv(out_dir / "paired_gains.csv", index=False)
    return summary, gains


def compute_adequacy(summary: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    radius_configs = sorted(
        [c for c in summary["config"].unique() if str(c).startswith("M_R")],
        key=lambda c: float(str(c)[3:].replace("p", ".")),
    )
    rows: List[Dict[str, object]] = []
    for scheme in sorted(summary["scheme"].unique()):
        for target in TARGETS:
            sub = summary[(summary["scheme"] == scheme) & (summary["target"] == target)].set_index("config")
            if "M_full" not in sub.index:
                continue
            full = float(sub.loc["M_full", "mean_relL2"])
            tolerance = float(args.tolerance_map[target])
            chosen = "not_reached"
            chosen_radius = float("nan")
            chosen_error = float("nan")
            for config in radius_configs:
                if config not in sub.index:
                    continue
                err = float(sub.loc[config, "mean_relL2"])
                if err <= tolerance and err - full <= args.delta_full:
                    chosen = config
                    chosen_radius = float(config[3:].replace("p", "."))
                    chosen_error = err
                    break
            rows.append(
                {
                    "scheme": scheme,
                    "target": target,
                    "absolute_tolerance": tolerance,
                    "delta_to_full_tolerance": args.delta_full,
                    "full_mean_relL2": full,
                    "R_adequate_config": chosen,
                    "R_adequate_over_hs": chosen_radius,
                    "error_at_R_adequate": chosen_error,
                    "is_censored": chosen == "not_reached",
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "adequacy_radius.csv", index=False)
    return result


def make_decision(
    summary: pd.DataFrame,
    gains: pd.DataFrame,
    adequacy: pd.DataFrame,
    args: argparse.Namespace,
    out_dir: Path,
) -> Dict[str, object]:
    checks: List[Dict[str, object]] = []
    for scheme in args.cv_list:
        for target in TARGETS:
            row = summary[
                (summary["scheme"] == scheme)
                & (summary["target"] == target)
                & (summary["config"] == "M_full")
            ]
            gain0 = gains[
                (gains["scheme"] == scheme)
                & (gains["target"] == target)
                & (gains["comparison"] == "M0-M_full")
            ]
            gain_shuffle = gains[
                (gains["scheme"] == scheme)
                & (gains["target"] == target)
                & (gains["comparison"] == "M_shuffled-M_full")
            ]
            if row.empty or gain0.empty or gain_shuffle.empty:
                continue
            full_mean = float(row.iloc[0]["mean_relL2"])
            full_hi = float(row.iloc[0]["case_bootstrap_CI97p5"])
            g0 = float(gain0.iloc[0]["mean_gain"])
            g0_lo = float(gain0.iloc[0]["case_bootstrap_CI2p5"])
            gs = float(gain_shuffle.iloc[0]["mean_gain"])
            gs_lo = float(gain_shuffle.iloc[0]["case_bootstrap_CI2p5"])
            tol = float(args.tolerance_map[target])
            checks.append(
                {
                    "scheme": scheme,
                    "target": target,
                    "full_mean_relL2": full_mean,
                    "full_CI97p5": full_hi,
                    "absolute_tolerance": tol,
                    "absolute_accuracy_pass": bool(full_hi <= tol),
                    "M0_minus_full_mean_gain": g0,
                    "M0_minus_full_CI2p5": g0_lo,
                    "bulk_gain_pass": bool(g0 >= args.min_gain and g0_lo > 0.0),
                    "shuffled_minus_full_mean_gain": gs,
                    "shuffled_minus_full_CI2p5": gs_lo,
                    "negative_control_pass": bool(gs_lo > 0.0),
                }
            )

    all_accuracy = bool(checks) and all(bool(x["absolute_accuracy_pass"]) for x in checks)
    all_gain = bool(checks) and all(bool(x["bulk_gain_pass"]) for x in checks)
    all_negative = bool(checks) and all(bool(x["negative_control_pass"]) for x in checks)

    if all_accuracy and all_gain and all_negative:
        verdict = "PROCEED_WITH_PREDICTIVE_SPATIAL_FOOTPRINT"
        action = (
            "The matched-capacity gate passed in both grouped validations. Report an "
            "accuracy-qualified predictive spatial footprint; do not call it a physical or causal horizon."
        )
    elif not all_accuracy:
        verdict = "STOP_HORIZON_CLAIM_ABSOLUTE_ACCURACY_INADEQUATE"
        action = (
            "At least one full-field model fails the predeclared absolute-accuracy bound. "
            "Do not interpret a short relative radius as closure; improve data/noise or the physical design first."
        )
    elif not all_gain or not all_negative:
        verdict = "PIVOT_BULK_NECESSITY_NOT_ESTABLISHED"
        action = (
            "Real bulk descriptors do not beat the matched M0 and shuffled controls robustly. "
            "Drop the information-horizon claim and recast the work around direct wall prediction or molecular provenance."
        )
    else:
        verdict = "INCONCLUSIVE"
        action = "Increase independent physical cases or seeds before making a bulk-necessity claim."

    decision: Dict[str, object] = {
        "verdict": verdict,
        "recommended_action": action,
        "rules": {
            "absolute_accuracy": "upper 95% physical-case bootstrap bound for M_full <= target tolerance",
            "bulk_gain": f"mean(M0-M_full) >= {100*args.min_gain:.1f} percentage points and lower 95% bound > 0",
            "negative_control": "lower 95% bound of M_shuffled-M_full > 0",
            "must_hold_in": args.cv_list,
            "terminology": "predictive spatial footprint, not physical/causal information horizon",
        },
        "checks": checks,
        "adequacy_radius": adequacy.replace({np.nan: None}).to_dict(orient="records"),
    }
    (out_dir / "gate_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    lines = ["LEKZIAN BULK-TO-WALL GATE DECISION", "=" * 38, "", verdict, "", action, "", "Checks:"]
    for item in checks:
        lines.append(
            f"- {item['scheme']:7s} {item['target']:7s}: "
            f"accuracy={item['absolute_accuracy_pass']}, "
            f"bulk_gain={item['bulk_gain_pass']}, negative_control={item['negative_control_pass']}"
        )
    lines.extend(["", "Read adequacy_radius.csv before reporting any finite radius."])
    (out_dir / "gate_decision.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    config_order = ["M0", "M_shuffled"] + sorted(
        [c for c in summary["config"].unique() if str(c).startswith("M_R")],
        key=lambda c: float(str(c)[3:].replace("p", ".")),
    ) + ["M_full"]
    schemes = sorted(summary["scheme"].unique())
    fig, axes = plt.subplots(len(schemes), len(TARGETS), figsize=(15, 4.8 * len(schemes)), squeeze=False)
    for i, scheme in enumerate(schemes):
        for j, target in enumerate(TARGETS):
            ax = axes[i, j]
            sub = summary[(summary["scheme"] == scheme) & (summary["target"] == target)].set_index("config")
            configs = [c for c in config_order if c in sub.index]
            y = np.asarray([100.0 * float(sub.loc[c, "mean_relL2"]) for c in configs])
            lo = np.asarray([100.0 * float(sub.loc[c, "case_bootstrap_CI2p5"]) for c in configs])
            hi = np.asarray([100.0 * float(sub.loc[c, "case_bootstrap_CI97p5"]) for c in configs])
            x = np.arange(len(configs))
            ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), marker="o", capsize=3, lw=1.8)
            ax.set_xticks(x)
            ax.set_xticklabels(configs, rotation=35, ha="right")
            ax.set_ylabel("case-mean relative L2 error (%)")
            ax.set_title(f"{scheme}: {target}")
            ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "gate_error_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "gate_error_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-table", required=True, help="surface_patch_dataset_full_gate.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--radii", default="0.1,0.25,0.5,1.0,2.0")
    parser.add_argument("--cv", default="loco,pairout", help="loco,pairout")
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--latent", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.03)
    parser.add_argument("--full-residual-scale", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--absolute-tolerances", default="Cp=0.10,Cq=0.10,tau_abs=0.20")
    parser.add_argument("--min-gain-pp", type=float, default=2.0)
    parser.add_argument("--delta-full-pp", type=float, default=2.0)
    parser.add_argument("--limit-folds", type=int, default=0, help="Testing only; 0 uses every outer fold.")
    parser.add_argument("--only-scheme", choices=["loco", "pairout"])
    parser.add_argument("--only-outer-group")
    parser.add_argument("--skip-summary", action="store_true", help="Array worker mode.")
    parser.add_argument("--aggregate-task-root", help="Aggregate completed task_* directories instead of training.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-surface-predictions", action="store_true")
    args = parser.parse_args()
    args.radii_list = _parse_float_list(args.radii)
    args.cv_list = [x.strip() for x in args.cv.split(",") if x.strip()]
    args.seeds_list = _parse_int_list(args.seeds)
    args.tolerance_map = _parse_target_map(args.absolute_tolerances)
    args.min_gain = args.min_gain_pp / 100.0
    args.delta_full = args.delta_full_pp / 100.0

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = Path(args.feature_table).expanduser().resolve()
    print(f"[INFO] feature table: {feature_path}", flush=True)
    print(f"[INFO] output directory: {out_dir}", flush=True)
    df = pd.read_csv(feature_path)
    df["case_id"] = df["case_id"].astype(str)
    schema = infer_schema(df)
    validate_table(df, schema)

    signature_payload = {
        "feature_sha256": _sha256_file(feature_path),
        "code_sha256": _sha256_file(Path(__file__).resolve()),
        "radii": args.radii_list,
        "cv": args.cv_list,
        "seeds": args.seeds_list,
        "hidden": args.hidden,
        "latent": args.latent,
        "depth": args.depth,
        "dropout": args.dropout,
        "full_residual_scale": args.full_residual_scale,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "only_scheme": args.only_scheme,
        "only_outer_group": args.only_outer_group,
        "skip_summary": args.skip_summary,
        "aggregate_task_root": args.aggregate_task_root,
    }
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()
    signature_path = out_dir / "run_signature.json"
    partials = [out_dir / "case_metrics.partial.csv", out_dir / "completed_tasks.txt"]
    if signature_path.exists():
        old = json.loads(signature_path.read_text(encoding="utf-8"))
        if old.get("signature") != signature:
            raise RuntimeError(
                "Existing resumable results were generated from different data/code/hyperparameters. "
                "Choose a new --out directory; mixed runs are forbidden."
            )
    elif any(path.exists() for path in partials):
        raise RuntimeError("Partial results exist without run_signature.json; choose a clean --out directory.")
    signature_path.write_text(
        json.dumps({"signature": signature, "payload": signature_payload}, indent=2), encoding="utf-8"
    )

    config_record = {k: v for k, v in vars(args).items() if not k.endswith("_list") and k != "tolerance_map"}
    config_record.update(
        {
            "feature_table": str(feature_path),
            "radii": args.radii_list,
            "cv": args.cv_list,
            "seeds": args.seeds_list,
            "absolute_tolerances": args.tolerance_map,
            "min_gain": args.min_gain,
            "delta_full": args.delta_full,
            "schema": asdict(schema),
            "run_signature": signature,
            "torch_version": None if torch is None else torch.__version__,
            "cuda_available": False if torch is None else torch.cuda.is_available(),
            "hostname": os.uname().nodename,
            "started_unix": time.time(),
        }
    )
    (out_dir / "run_config.json").write_text(json.dumps(config_record, indent=2), encoding="utf-8")

    if args.aggregate_task_root:
        metrics = aggregate_task_results(
            Path(args.aggregate_task_root).expanduser().resolve(), out_dir, not args.no_surface_predictions
        )
    else:
        metrics = run_experiment(df, schema, args)
    if args.skip_summary:
        print(f"[DONE] array worker outputs under {out_dir}", flush=True)
        return
    summary, gains = summarise_metrics(metrics, args, out_dir)
    adequacy = compute_adequacy(summary, args, out_dir)
    decision = make_decision(summary, gains, adequacy, args, out_dir)
    plot_summary(summary, out_dir)
    print(f"[DONE] verdict={decision['verdict']}", flush=True)
    print(f"[DONE] read {out_dir / 'gate_decision.txt'}", flush=True)


if __name__ == "__main__":
    main()
