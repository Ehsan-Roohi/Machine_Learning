#!/usr/bin/env python3
"""Stage-1 confirmatory test using the existing macroscopic DSMC descriptors.

This second-generation test is designed for a substantially new Physics of
Fluids manuscript.  It does not require stress, heat-flux, higher-order moment,
or particle/VDF output from SPARTA.  It fixes four limitations of the first
Gate Test:

* each wall quantity is trained independently;
* every annulus is compressed by one shared ring encoder, so revealing more
  rings does not increase the latent input dimension;
* a single mask-conditioned model produces M0, finite-radius, matched-count
  near/far/interleaved controls, and M_full predictions;
* five seeds are combined as a prediction ensemble before profile errors and
  grouped bootstrap intervals are computed.

The permitted interpretation is an accuracy-qualified *predictive spatial
support*.  The code never calls the result a physical or causal horizon.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import gate_test as v1


TARGETS = v1.TARGETS
FINITE_CONFIGS = ("M_R0p1", "M_R0p25", "M_R0p5", "M_R1", "M_R2")
REAL_CONFIGS = (
    "M0",
    *FINITE_CONFIGS,
    "M_far_K3",
    "M_interleaved_K3",
    "M_full",
)
ATTENTION_CONFIGS = ("M_R0p5", "M_full")


def parse_target_map(text: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for token in str(text).split(","):
        name, value = token.strip().split("=", 1)
        result[name.strip()] = float(value)
    missing = set(TARGETS) - set(result)
    if missing:
        raise ValueError(f"Missing tolerances for {sorted(missing)}")
    return result


def stable_seed(*parts: object) -> int:
    return v1._stable_seed("stage1_v2", *parts)


def rel_l2(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> float:
    return v1.rel_l2(pred, true, eps)


def range_mae_pct(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> float:
    return v1.range_mae_pct(pred, true, eps)


def mean_profile_rel_l2(
    pred: np.ndarray, true: np.ndarray, case_ids: Sequence[str]
) -> float:
    frame = pd.DataFrame({"case_id": np.asarray(case_ids, dtype=str), "pred": pred, "true": true})
    values = [rel_l2(group["pred"].to_numpy(), group["true"].to_numpy()) for _, group in frame.groupby("case_id")]
    return float(np.mean(values))


def mask_configs(schema: v1.FeatureSchema) -> Dict[str, np.ndarray]:
    n_rings = len(schema.ring_names)
    if n_rings != 6:
        raise ValueError(f"Stage-1 V2 requires the six annuli used by the paper; found {n_rings}.")
    result: Dict[str, np.ndarray] = {
        "M0": np.zeros(n_rings, dtype=bool),
        "M_far_K3": np.asarray([False, False, False, True, True, True]),
        "M_interleaved_K3": np.asarray([True, False, True, False, True, False]),
        "M_full": np.ones(n_rings, dtype=bool),
    }
    for radius in (0.1, 0.25, 0.5, 1.0, 2.0):
        name = v1._radius_label(radius)
        result[name] = v1.visible_rings(name, schema)
    return {name: result[name] for name in REAL_CONFIGS}


def ring_column_groups(schema: v1.FeatureSchema) -> List[np.ndarray]:
    index = np.asarray(schema.bulk_ring_index, dtype=int)
    groups = [np.flatnonzero(index == ring_i) for ring_i in range(len(schema.ring_names))]
    widths = {len(group) for group in groups}
    if len(widths) != 1:
        raise ValueError(f"Shared ring encoder requires equal descriptor counts; found {sorted(widths)}")
    return groups


def explicit_surface_coordinates(frame: pd.DataFrame, order: int = 4) -> np.ndarray:
    s = frame["s01"].to_numpy(np.float32)
    columns: List[np.ndarray] = [2.0 * s - 1.0, np.abs(2.0 * s - 1.0)]
    for k in range(1, order + 1):
        columns.append(np.sin(2.0 * np.pi * k * s))
        columns.append(np.cos(2.0 * np.pi * k * s))
    return np.column_stack(columns).astype(np.float32)


def transform_arrays(
    frame: pd.DataFrame,
    schema: v1.FeatureSchema,
    prep: v1.Preprocessor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = v1._standardise(
        frame[list(schema.base_columns)].to_numpy(float),
        prep.base_median,
        prep.base_mean,
        prep.base_std,
    )
    case = base[:, : schema.case_dimension]
    surface = np.column_stack(
        [base[:, schema.case_dimension :], explicit_surface_coordinates(frame)]
    ).astype(np.float32)
    bulk = v1._standardise(
        frame[list(schema.bulk_columns)].to_numpy(float),
        prep.bulk_median,
        prep.bulk_mean,
        prep.bulk_std,
    )
    groups = ring_column_groups(schema)
    rings = np.stack([bulk[:, group] for group in groups], axis=1).astype(np.float32)
    return case.astype(np.float32), surface, rings


def permute_case_profiles(
    frame: pd.DataFrame,
    rings: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, str]]:
    """Reassign coherent bulk profiles across physical cases within geometry.

    All descriptors and annuli from one source case remain together.  Values
    are interpolated on ``s01`` if source and destination surface grids differ.
    Targets, case parameters, surface coordinates, and geometry are untouched.
    """

    rng = np.random.default_rng(seed)
    result = np.empty_like(rings)
    mapping: Dict[str, str] = {}
    metadata = frame[["case_id", "geom"]].drop_duplicates().copy()
    for _, block in metadata.groupby("geom"):
        cases = sorted(block["case_id"].astype(str).tolist())
        if len(cases) < 2:
            raise ValueError("Case-level conditional permutation needs at least two cases per geometry.")
        shift = int(rng.integers(1, len(cases)))
        sources = cases[shift:] + cases[:shift]
        mapping.update(dict(zip(cases, sources)))

    s_all = frame["s01"].to_numpy(float)
    ids = frame["case_id"].astype(str).to_numpy()
    for destination, source in mapping.items():
        dst_idx = np.flatnonzero(ids == destination)
        src_idx = np.flatnonzero(ids == source)
        src_order = np.argsort(s_all[src_idx])
        src_idx = src_idx[src_order]
        src_s = s_all[src_idx]
        dst_s = s_all[dst_idx]
        for ring_i in range(rings.shape[1]):
            for feature_i in range(rings.shape[2]):
                result[dst_idx, ring_i, feature_i] = np.interp(
                    dst_s,
                    src_s,
                    rings[src_idx, ring_i, feature_i],
                )
    return result, mapping


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        width = in_dim
        for _ in range(depth):
            layers.extend([nn.Linear(width, hidden), nn.SiLU()])
            if dropout:
                layers.append(nn.Dropout(dropout))
            width = hidden
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class NestedRingOperator(nn.Module):
    """Strong target-specific wall surrogate plus a fixed-width ring residual.

    The parameter/surface path retains the structured branch, multiplicative
    trunk, and windward/leeward gate used by the manuscript's strong direct-wall
    surrogate.  M0 is therefore a first-class prediction path, not a weak
    external baseline.
    """

    def __init__(
        self,
        case_dim: int,
        surface_dim: int,
        ring_dim: int,
        n_rings: int,
        latent: int,
        hidden: int,
        depth: int,
        dropout: float,
        full_residual_scale: float,
    ):
        super().__init__()
        if case_dim < 7:
            raise ValueError("The structured direct-wall path requires at least seven case features.")
        self.n_rings = int(n_rings)
        self.latent = int(latent)
        self.full_residual_scale = float(full_residual_scale)
        self.base_branch = MLP(5, latent, hidden, depth, dropout)
        self.h_branch = MLP(4, latent, hidden, depth, dropout)
        self.tw_branch = MLP(4, latent, hidden, depth, dropout)
        self.full_branch = MLP(case_dim, latent, hidden, max(1, depth - 1), dropout)
        self.surface_encoder = MLP(surface_dim, latent, hidden, depth, dropout)
        self.wind_decoder = MLP(latent, 1, hidden, depth, dropout)
        self.lee_decoder = MLP(latent, 1, hidden, depth, dropout)
        self.surface_gate = MLP(surface_dim, 1, max(32, hidden // 2), 2, 0.0)
        self.ring_encoder = MLP(ring_dim, latent, hidden, max(1, depth - 1), dropout)
        self.ring_position = nn.Parameter(torch.zeros(n_rings, latent))
        nn.init.normal_(self.ring_position, std=0.03)
        self.query = nn.Linear(latent, latent)
        self.key = nn.Linear(latent, latent)
        self.value = nn.Linear(latent, latent)
        self.bulk_decoder = MLP(2 * latent + 1, 1, hidden, depth, dropout)

    def branch_embedding(self, case: torch.Tensor) -> torch.Tensor:
        ma_log_geom = torch.cat([case[:, 0:2], case[:, 4:7]], dim=1)
        log_geom = torch.cat([case[:, 1:2], case[:, 4:7]], dim=1)
        embedding = self.base_branch(ma_log_geom)
        embedding = embedding + case[:, 2:3] * self.h_branch(log_geom)
        embedding = embedding + case[:, 3:4] * self.tw_branch(log_geom)
        if self.full_residual_scale > 0:
            embedding = embedding + self.full_residual_scale * self.full_branch(case)
        return embedding

    def forward(
        self,
        case: torch.Tensor,
        surface: torch.Tensor,
        rings: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        context = self.branch_embedding(case) * self.surface_encoder(surface)
        batch, n_rings, ring_dim = rings.shape
        encoded = self.ring_encoder(rings.reshape(batch * n_rings, ring_dim)).reshape(
            batch, n_rings, self.latent
        )
        encoded = encoded + self.ring_position.unsqueeze(0)
        query = self.query(context).unsqueeze(1)
        scores = (query * self.key(encoded)).sum(dim=-1) / math.sqrt(self.latent)
        visible = mask.to(dtype=scores.dtype)
        shifted = scores - scores.max(dim=1, keepdim=True).values
        weights = torch.exp(shifted) * visible
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        aggregate = (weights.unsqueeze(-1) * self.value(encoded)).sum(dim=1)
        has_bulk = visible.any(dim=1, keepdim=True).to(dtype=context.dtype)
        fraction = visible.mean(dim=1, keepdim=True)
        gate = torch.sigmoid(self.surface_gate(surface))
        base = gate * self.wind_decoder(context) + (1.0 - gate) * self.lee_decoder(context)
        residual = self.bulk_decoder(torch.cat([context, aggregate, fraction], dim=1))
        prediction = base + has_bulk * residual
        if return_attention:
            return prediction, weights
        return prediction


def torch_mask(mask: np.ndarray, n: int, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.repeat(mask[None, :], n, axis=0)).to(device=device, dtype=torch.bool)


def weighted_huber(pred: torch.Tensor, true: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    element = F.huber_loss(pred, true, delta=1.0, reduction="none")
    return (element * weight).sum() / weight.sum().clamp_min(1e-8)


def train_operator(
    train_arrays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    y_train: np.ndarray,
    w_train: np.ndarray,
    val_arrays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    y_val: np.ndarray,
    w_val: np.ndarray,
    masks: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    seed: int,
) -> Tuple[NestedRingOperator, int, float, int]:
    v1._seed_everything(seed)
    case_train, surface_train, rings_train = train_arrays
    case_val, surface_val, rings_val = val_arrays
    model = NestedRingOperator(
        case_dim=case_train.shape[1],
        surface_dim=surface_train.shape[1],
        ring_dim=rings_train.shape[2],
        n_rings=rings_train.shape[1],
        latent=args.latent,
        hidden=args.hidden,
        depth=args.depth,
        dropout=args.dropout,
        full_residual_scale=args.full_residual_scale,
    ).to(args.device_obj)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(case_train),
            torch.from_numpy(surface_train),
            torch.from_numpy(rings_train),
            torch.from_numpy(y_train),
            torch.from_numpy(w_train),
        ),
        batch_size=min(args.batch_size, len(case_train)),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    cval = torch.from_numpy(case_val).to(args.device_obj)
    sval = torch.from_numpy(surface_val).to(args.device_obj)
    rval = torch.from_numpy(rings_val).to(args.device_obj)
    yval = torch.from_numpy(y_val).to(args.device_obj)
    wval = torch.from_numpy(w_val).to(args.device_obj)

    variable_masks = [name for name in REAL_CONFIGS if name not in {"M0", "M_full"}]
    validation_masks = ["M0", "M_R0p5", "M_full"]
    best_state: Optional[Mapping[str, torch.Tensor]] = None
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_i, (case_b, surface_b, rings_b, y_b, w_b) in enumerate(loader):
            case_b = case_b.to(args.device_obj, non_blocking=True)
            surface_b = surface_b.to(args.device_obj, non_blocking=True)
            rings_b = rings_b.to(args.device_obj, non_blocking=True)
            y_b = y_b.to(args.device_obj, non_blocking=True)
            w_b = w_b.to(args.device_obj, non_blocking=True)
            selected = variable_masks[(epoch + batch_i) % len(variable_masks)]
            names = ("M0", selected, "M_full")
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for name in names:
                mask_b = torch_mask(masks[name], len(case_b), args.device_obj)
                losses.append(weighted_huber(model(case_b, surface_b, rings_b, mask_b), y_b, w_b))
            loss = torch.stack(losses).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            values = []
            for name in validation_masks:
                mask_v = torch_mask(masks[name], len(cval), args.device_obj)
                values.append(weighted_huber(model(cval, sval, rval, mask_v), yval, wval))
            value = float(torch.stack(values).mean().cpu())
        if value < best_loss - 1e-6:
            best_loss = value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation checkpoint.")
    model.load_state_dict(best_state)
    return model, best_epoch, best_loss, parameter_count


def predict_operator(
    model: NestedRingOperator,
    arrays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    mask: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    case, surface, rings = arrays
    predictions: List[np.ndarray] = []
    attentions: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(case), max(batch_size, 4096)):
            stop = min(len(case), start + max(batch_size, 4096))
            case_b = torch.from_numpy(case[start:stop]).to(device)
            surface_b = torch.from_numpy(surface[start:stop]).to(device)
            rings_b = torch.from_numpy(rings[start:stop]).to(device)
            mask_b = torch_mask(mask, stop - start, device)
            pred, attention = model(case_b, surface_b, rings_b, mask_b, return_attention=True)
            predictions.append(pred.cpu().numpy().reshape(-1))
            attentions.append(attention.cpu().numpy())
    return np.concatenate(predictions), np.vstack(attentions)


def physical_predictions(
    model: NestedRingOperator,
    arrays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    masks: Mapping[str, np.ndarray],
    y_mean: float,
    y_std: float,
    args: argparse.Namespace,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    values: Dict[str, np.ndarray] = {}
    attention: Dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        pred_z, weights = predict_operator(model, arrays, mask, args.device_obj, args.batch_size)
        values[name] = pred_z * y_std + y_mean
        attention[name] = weights
    return values, attention


def choose_inner_radius(
    predictions: Mapping[str, np.ndarray],
    truth: np.ndarray,
    case_ids: Sequence[str],
    absolute_tolerance: float,
    noninferiority_margin: float,
) -> Tuple[str, bool, pd.DataFrame]:
    rows = []
    for config in (*FINITE_CONFIGS, "M_full"):
        rows.append(
            {
                "config": config,
                "validation_relL2": mean_profile_rel_l2(predictions[config], truth, case_ids),
                "radius_over_hs": (
                    float(config[3:].replace("p", ".")) if config != "M_full" else math.inf
                ),
            }
        )
    table = pd.DataFrame(rows)
    best_error = float(table["validation_relL2"].min())
    table["absolute_tolerance"] = float(absolute_tolerance)
    table["noninferiority_margin"] = float(noninferiority_margin)
    table["qualified"] = (
        (table["validation_relL2"] <= absolute_tolerance)
        & (table["validation_relL2"] <= best_error + noninferiority_margin)
    )
    finite = table[(table["config"] != "M_full") & table["qualified"]].sort_values(
        "radius_over_hs"
    )
    if finite.empty:
        selected = "M_full"
        censored = True
    else:
        selected = str(finite.iloc[0]["config"])
        censored = False
    table["selected_config"] = selected
    table["radius_censored"] = censored
    return selected, censored, table.sort_values("radius_over_hs")


def normalize_weights(frame: pd.DataFrame) -> np.ndarray:
    weights = frame["gate_sample_weight"].to_numpy(np.float32).reshape(-1, 1)
    return weights / max(float(weights.mean()), 1e-8)


def run_one_task(
    frame: pd.DataFrame,
    schema: v1.FeatureSchema,
    args: argparse.Namespace,
) -> None:
    if not args.only_scheme or not args.only_outer_group or not args.only_target:
        raise ValueError("Worker mode requires --only-scheme, --only-outer-group, and --only-target.")
    scheme = args.only_scheme
    outer_group = args.only_outer_group
    target = args.only_target
    seed = args.seeds_list[0]
    out_dir = Path(args.out)
    marker = out_dir / "TASK_COMPLETE"
    if args.resume and marker.exists():
        print(f"[SKIP] completed task under {out_dir}", flush=True)
        return

    groups = v1._outer_groups(frame, scheme).astype(str)
    if outer_group not in set(groups):
        raise ValueError(f"Outer group {outer_group!r} not found for {scheme}.")
    test_mask = groups == outer_group
    outer_train = frame.loc[~test_mask].reset_index(drop=True)
    test_df = frame.loc[test_mask].reset_index(drop=True)
    train_groups = v1._outer_groups(outer_train, scheme).astype(str)
    # Every seed uses the same physical validation group. This permits a true
    # validation-prediction ensemble before radius selection.
    inner_group = v1._choose_inner_group(
        train_groups,
        outer_group,
        stable_seed("shared_inner_group", scheme, outer_group),
    )
    val_mask = train_groups == inner_group
    train_df = outer_train.loc[~val_mask].reset_index(drop=True)
    val_df = outer_train.loc[val_mask].reset_index(drop=True)

    prep = v1.fit_preprocessor(train_df, schema)
    target_i = TARGETS.index(target)
    y_mean = float(prep.y_mean[target_i])
    y_std = float(prep.y_std[target_i])
    y_train = ((train_df[target].to_numpy(float) - y_mean) / y_std).astype(np.float32).reshape(-1, 1)
    y_val = ((val_df[target].to_numpy(float) - y_mean) / y_std).astype(np.float32).reshape(-1, 1)
    truth_val = val_df[target].to_numpy(float)
    truth_test = test_df[target].to_numpy(float)
    arrays_train = transform_arrays(train_df, schema, prep)
    arrays_val = transform_arrays(val_df, schema, prep)
    arrays_test = transform_arrays(test_df, schema, prep)
    masks = mask_configs(schema)
    model_seed = stable_seed("model", scheme, outer_group, target, seed)

    real_model, best_epoch, best_loss, n_parameters = train_operator(
        arrays_train,
        y_train,
        normalize_weights(train_df),
        arrays_val,
        y_val,
        normalize_weights(val_df),
        masks,
        args,
        model_seed,
    )
    val_predictions, _ = physical_predictions(
        real_model, arrays_val, masks, y_mean, y_std, args
    )
    diagnostic_config, diagnostic_censored, selection_table = choose_inner_radius(
        val_predictions,
        truth_val,
        val_df["case_id"].astype(str).to_numpy(),
        args.selection_tolerance_map[target],
        args.selection_margin,
    )
    test_predictions, test_attention = physical_predictions(
        real_model, arrays_test, masks, y_mean, y_std, args
    )

    permuted_rings, permutation_map = permute_case_profiles(
        train_df,
        arrays_train[2],
        stable_seed("case_profile_permutation", scheme, outer_group, target, seed),
    )
    permuted_train = (arrays_train[0], arrays_train[1], permuted_rings)
    null_model, null_epoch, null_loss, null_parameters = train_operator(
        permuted_train,
        y_train,
        normalize_weights(train_df),
        arrays_val,
        y_val,
        normalize_weights(val_df),
        masks,
        args,
        model_seed,
    )
    if null_parameters != n_parameters:
        raise AssertionError("Real and permutation-control models changed parameter count.")
    null_predictions, _ = physical_predictions(
        null_model, arrays_test, masks, y_mean, y_std, args
    )
    test_predictions["M_permtrain_R0p5"] = null_predictions["M_R0p5"]
    test_predictions["M_permtrain_full"] = null_predictions["M_full"]

    metric_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    validation_prediction_rows: List[Dict[str, object]] = []
    attention_rows: List[Dict[str, object]] = []
    ids = test_df["case_id"].astype(str).to_numpy()
    for case_id in sorted(set(ids)):
        loc = np.flatnonzero(ids == case_id)
        for config, pred in test_predictions.items():
            metric_rows.append(
                {
                    "scheme": scheme,
                    "outer_group": outer_group,
                    "inner_group": inner_group,
                    "seed": seed,
                    "target": target,
                    "config": config,
                    "case_id": case_id,
                    "relL2": rel_l2(pred[loc], truth_test[loc]),
                    "rangeMAE_pct": range_mae_pct(pred[loc], truth_test[loc]),
                    "best_epoch": best_epoch,
                    "inner_val_loss": best_loss,
                    "null_best_epoch": null_epoch,
                    "null_inner_val_loss": null_loss,
                    "model_parameters": n_parameters,
                    "ring_latent_dimension": args.latent,
                }
            )

    for row_i, row in test_df.reset_index(drop=True).iterrows():
        for config, pred in test_predictions.items():
            prediction_rows.append(
                {
                    "scheme": scheme,
                    "outer_group": outer_group,
                    "seed": seed,
                    "target": target,
                    "config": config,
                    "case_id": str(row["case_id"]),
                    "surface_i": int(row.get("surface_i", row_i)),
                    "s01": float(row["s01"]),
                    "true": float(truth_test[row_i]),
                    "pred": float(pred[row_i]),
                }
            )
        for config in ATTENTION_CONFIGS:
            for ring_i, ring_name in enumerate(schema.ring_names):
                attention_rows.append(
                    {
                        "scheme": scheme,
                        "outer_group": outer_group,
                        "seed": seed,
                        "target": target,
                        "config": config,
                        "case_id": str(row["case_id"]),
                        "surface_i": int(row.get("surface_i", row_i)),
                        "s01": float(row["s01"]),
                        "ring": ring_name,
                        "attention": float(test_attention[config][row_i, ring_i]),
                    }
                )

    for row_i, row in val_df.reset_index(drop=True).iterrows():
        for config, pred in val_predictions.items():
            validation_prediction_rows.append(
                {
                    "scheme": scheme,
                    "outer_group": outer_group,
                    "inner_group": inner_group,
                    "seed": seed,
                    "target": target,
                    "config": config,
                    "case_id": str(row["case_id"]),
                    "surface_i": int(row.get("surface_i", row_i)),
                    "s01": float(row["s01"]),
                    "true": float(truth_val[row_i]),
                    "pred": float(pred[row_i]),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(out_dir / "case_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(out_dir / "surface_predictions.csv", index=False)
    pd.DataFrame(validation_prediction_rows).to_csv(
        out_dir / "validation_predictions.csv", index=False
    )
    pd.DataFrame(attention_rows).to_csv(out_dir / "attention_profiles.csv", index=False)
    selection_table.assign(
        scheme=scheme,
        outer_group=outer_group,
        inner_group=inner_group,
        seed=seed,
        target=target,
        diagnostic_selected_config=diagnostic_config,
        diagnostic_radius_censored=diagnostic_censored,
    ).to_csv(out_dir / "seed_radius_diagnostic.csv", index=False)
    (out_dir / "permutation_map.json").write_text(
        json.dumps(permutation_map, indent=2), encoding="utf-8"
    )
    marker.write_text("complete\n", encoding="utf-8")
    print(
        f"[DONE] {scheme}|{outer_group}|{seed}|{target}; "
        f"seed_diagnostic={diagnostic_config}",
        flush=True,
    )


def concat_task_files(task_root: Path, filename: str, required: bool = True) -> pd.DataFrame:
    paths = sorted(task_root.glob(f"task_*/{filename}"))
    if not paths and required:
        raise RuntimeError(f"No {filename} files found under {task_root}")
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def ensemble_case_metrics(
    predictions: pd.DataFrame,
    expected_seeds: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    point_key = [
        "scheme",
        "outer_group",
        "target",
        "config",
        "case_id",
        "surface_i",
        "s01",
    ]
    ensemble = (
        predictions.groupby(point_key, as_index=False)
        .agg(true=("true", "first"), pred=("pred", "mean"), n_seeds=("seed", "nunique"))
    )
    minimum_seeds = int(ensemble["n_seeds"].min())
    maximum_seeds = int(ensemble["n_seeds"].max())
    if expected_seeds is None:
        if minimum_seeds < 2:
            raise RuntimeError("Prediction ensemble has fewer than two seeds for at least one profile.")
    elif minimum_seeds != expected_seeds or maximum_seeds != expected_seeds:
        raise RuntimeError(
            "Incomplete prediction ensemble: "
            f"expected exactly {expected_seeds} seeds per point, found {minimum_seeds}--{maximum_seeds}."
        )
    return ensemble, case_metrics_from_ensemble(ensemble)


def case_metrics_from_ensemble(ensemble: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for keys, group in ensemble.groupby(
        ["scheme", "outer_group", "target", "config", "case_id"]
    ):
        scheme, outer_group, target, config, case_id = keys
        rows.append(
            {
                "scheme": scheme,
                "outer_group": outer_group,
                "target": target,
                "config": config,
                "case_id": case_id,
                "n_seeds": int(group["n_seeds"].min()),
                "relL2": rel_l2(group["pred"].to_numpy(), group["true"].to_numpy()),
                "rangeMAE_pct": range_mae_pct(
                    group["pred"].to_numpy(), group["true"].to_numpy()
                ),
            }
        )
    return pd.DataFrame(rows)


def select_radii_from_validation(
    validation_points: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for (scheme, outer_group, target), group in validation_points.groupby(
        ["scheme", "outer_group", "target"]
    ):
        predictions: Dict[str, np.ndarray] = {}
        truth: Optional[np.ndarray] = None
        case_ids: Optional[np.ndarray] = None
        for config in (*FINITE_CONFIGS, "M_full"):
            part = group[group["config"] == config].sort_values(["case_id", "surface_i"])
            if part.empty:
                raise RuntimeError(
                    f"Missing validation prediction for {scheme}/{outer_group}/{target}/{config}"
                )
            predictions[config] = part["pred"].to_numpy(float)
            if truth is None:
                truth = part["true"].to_numpy(float)
                case_ids = part["case_id"].astype(str).to_numpy()
            elif not np.allclose(truth, part["true"].to_numpy(float), equal_nan=True):
                raise RuntimeError("Validation truth changed across radius configurations.")
        assert truth is not None and case_ids is not None
        _, _, table = choose_inner_radius(
            predictions,
            truth,
            case_ids,
            args.selection_tolerance_map[str(target)],
            args.selection_margin,
        )
        rows.append(
            table.assign(scheme=scheme, outer_group=outer_group, target=target)
        )
    return pd.concat(rows, ignore_index=True)


def append_selected_predictions(
    ensemble_points: pd.DataFrame,
    selections: pd.DataFrame,
) -> pd.DataFrame:
    selected_rows = selections[
        selections["config"] == selections["selected_config"]
    ][["scheme", "outer_group", "target", "selected_config", "radius_censored"]]
    if selected_rows.duplicated(["scheme", "outer_group", "target"]).any():
        raise RuntimeError("Radius selection produced more than one selected configuration.")
    pieces: List[pd.DataFrame] = []
    for row in selected_rows.itertuples(index=False):
        part = ensemble_points[
            (ensemble_points["scheme"] == row.scheme)
            & (ensemble_points["outer_group"] == row.outer_group)
            & (ensemble_points["target"] == row.target)
            & (ensemble_points["config"] == row.selected_config)
        ].copy()
        if part.empty:
            raise RuntimeError(
                f"Selected test prediction is missing for {row.scheme}/{row.outer_group}/{row.target}."
            )
        part["source_config"] = row.selected_config
        part["radius_censored"] = bool(row.radius_censored)
        part["config"] = "M_selected"
        pieces.append(part)
    return pd.concat([ensemble_points, *pieces], ignore_index=True)


def bootstrap_units(
    values: np.ndarray,
    n_boot: int,
    seed: int,
) -> Tuple[float, float, float]:
    return v1._case_bootstrap(values, n_boot, seed)


def unit_metrics(case_metrics: pd.DataFrame) -> pd.DataFrame:
    frame = case_metrics.copy()
    frame["unit_id"] = np.where(
        frame["scheme"] == "loco", frame["case_id"], frame["outer_group"]
    )
    return (
        frame.groupby(["scheme", "target", "config", "unit_id"], as_index=False)
        .agg(relL2=("relL2", "mean"), rangeMAE_pct=("rangeMAE_pct", "mean"), n_cases=("case_id", "nunique"))
    )


def summarize_ensemble(
    case_metrics: pd.DataFrame,
    args: argparse.Namespace,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    units = unit_metrics(case_metrics)
    summary_rows: List[Dict[str, object]] = []
    for (scheme, target, config), group in units.groupby(["scheme", "target", "config"]):
        mean, lo, hi = bootstrap_units(
            group["relL2"].to_numpy(float),
            args.bootstrap,
            stable_seed("summary", scheme, target, config),
        )
        summary_rows.append(
            {
                "scheme": scheme,
                "target": target,
                "config": config,
                "n_bootstrap_units": group["unit_id"].nunique(),
                "mean_relL2": mean,
                "median_relL2": float(group["relL2"].median()),
                "bootstrap_CI2p5": lo,
                "bootstrap_CI97p5": hi,
                "mean_rangeMAE_pct": float(group["rangeMAE_pct"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "ensemble_summary.csv", index=False)

    comparisons = [
        ("M0", "M_selected"),
        ("M0", "M_R0p5"),
        ("M_far_K3", "M_R0p5"),
        ("M_interleaved_K3", "M_R0p5"),
        ("M_permtrain_R0p5", "M_R0p5"),
        ("M_permtrain_full", "M_full"),
        ("M_full", "M_selected"),
    ]
    gain_rows: List[Dict[str, object]] = []
    for (scheme, target), group in units.groupby(["scheme", "target"]):
        pivot = group.pivot(index="unit_id", columns="config", values="relL2")
        for baseline, candidate in comparisons:
            if baseline not in pivot or candidate not in pivot:
                continue
            diff = (pivot[baseline] - pivot[candidate]).dropna().to_numpy(float)
            point, lo, hi = bootstrap_units(
                diff,
                args.bootstrap,
                stable_seed("gain", scheme, target, baseline, candidate),
            )
            gain_rows.append(
                {
                    "scheme": scheme,
                    "target": target,
                    "comparison": f"{baseline}-{candidate}",
                    "n_bootstrap_units": len(diff),
                    "mean_gain": point,
                    "mean_gain_pp": 100.0 * point,
                    "bootstrap_CI2p5": lo,
                    "bootstrap_CI97p5": hi,
                    "CI2p5_pp": 100.0 * lo,
                    "CI97p5_pp": 100.0 * hi,
                }
            )
    gains = pd.DataFrame(gain_rows)
    gains.to_csv(out_dir / "ensemble_paired_gains.csv", index=False)
    return summary, gains


def selected_radius_summary(
    selections: pd.DataFrame,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    chosen = selections[selections["config"] == selections["selected_config"]].copy()
    distribution = (
        chosen.groupby(
            ["scheme", "target", "selected_config", "radius_over_hs", "radius_censored"],
            as_index=False,
            dropna=False,
        )
        .agg(n_selections=("outer_group", "size"), n_outer_groups=("outer_group", "nunique"))
    )
    censoring = (
        chosen.groupby(["scheme", "target"], as_index=False)
        .agg(
            n_outer_groups=("outer_group", "nunique"),
            n_censored=("radius_censored", "sum"),
            fraction_censored=("radius_censored", "mean"),
        )
    )
    distribution.to_csv(out_dir / "selected_radius_distribution.csv", index=False)
    censoring.to_csv(out_dir / "selected_radius_censoring.csv", index=False)
    return distribution, censoring


def lookup_censoring(censoring: pd.DataFrame, scheme: str, target: str) -> pd.Series:
    row = censoring[(censoring["scheme"] == scheme) & (censoring["target"] == target)]
    if row.empty:
        raise RuntimeError(f"Missing censoring summary for {scheme}/{target}")
    return row.iloc[0]


def lookup_summary(summary: pd.DataFrame, scheme: str, target: str, config: str) -> pd.Series:
    row = summary[
        (summary["scheme"] == scheme)
        & (summary["target"] == target)
        & (summary["config"] == config)
    ]
    if row.empty:
        raise RuntimeError(f"Missing ensemble summary for {scheme}/{target}/{config}")
    return row.iloc[0]


def lookup_gain(gains: pd.DataFrame, scheme: str, target: str, comparison: str) -> pd.Series:
    row = gains[
        (gains["scheme"] == scheme)
        & (gains["target"] == target)
        & (gains["comparison"] == comparison)
    ]
    if row.empty:
        raise RuntimeError(f"Missing paired gain for {scheme}/{target}/{comparison}")
    return row.iloc[0]


def make_decision(
    summary: pd.DataFrame,
    gains: pd.DataFrame,
    censoring: pd.DataFrame,
    args: argparse.Namespace,
    out_dir: Path,
) -> Dict[str, object]:
    checks: List[Dict[str, object]] = []
    all_primary = True
    for target in TARGETS:
        loco = lookup_summary(summary, "loco", target, "M_selected")
        pairout = lookup_summary(summary, "pairout", target, "M_selected")
        loco_gain = lookup_gain(gains, "loco", target, "M0-M_selected")
        pair_gain = lookup_gain(gains, "pairout", target, "M0-M_selected")
        locality = lookup_gain(gains, "loco", target, "M_far_K3-M_R0p5")
        permutation = lookup_gain(gains, "loco", target, "M_permtrain_R0p5-M_R0p5")
        loco_censoring = lookup_censoring(censoring, "loco", target)
        pairout_censoring = lookup_censoring(censoring, "pairout", target)
        is_primary_quantity = target in {"Cp", "Cq"}
        loco_accuracy = float(loco["bootstrap_CI97p5"]) <= args.loco_tolerance_map[target]
        pair_accuracy = float(pairout["mean_relL2"]) <= args.pairout_tolerance_map[target]
        bulk_gain = (
            float(loco_gain["mean_gain"]) >= args.min_gain
            and float(loco_gain["bootstrap_CI2p5"]) > 0.0
            and (
                not is_primary_quantity
                or (
                    float(pair_gain["mean_gain"]) >= args.min_gain
                    and float(pair_gain["bootstrap_CI2p5"]) > 0.0
                )
            )
        )
        locality_pass = float(locality["bootstrap_CI2p5"]) > 0.0
        permutation_pass = float(permutation["bootstrap_CI2p5"]) > 0.0
        censoring_pass = (
            float(loco_censoring["fraction_censored"]) <= args.max_censor_fraction
            and float(pairout_censoring["fraction_censored"]) <= args.max_censor_fraction
        )
        target_primary_pass = loco_accuracy and pair_accuracy
        if is_primary_quantity:
            target_primary_pass = (
                target_primary_pass
                and bulk_gain
                and locality_pass
                and permutation_pass
                and censoring_pass
            )
        all_primary = all_primary and target_primary_pass
        checks.append(
            {
                "target": target,
                "primary_quantity": is_primary_quantity,
                "loco_selected_mean_relL2": float(loco["mean_relL2"]),
                "loco_selected_CI97p5": float(loco["bootstrap_CI97p5"]),
                "loco_absolute_tolerance": args.loco_tolerance_map[target],
                "loco_accuracy_pass": loco_accuracy,
                "pairout_selected_mean_relL2": float(pairout["mean_relL2"]),
                "pairout_absolute_tolerance": args.pairout_tolerance_map[target],
                "pairout_accuracy_pass": pair_accuracy,
                "bulk_gain_pass": bulk_gain,
                "near3_beats_far3_pass": locality_pass,
                "real_beats_case_permutation_pass": permutation_pass,
                "loco_fraction_censored": float(loco_censoring["fraction_censored"]),
                "pairout_fraction_censored": float(pairout_censoring["fraction_censored"]),
                "censoring_pass": censoring_pass,
                "target_pass": target_primary_pass,
            }
        )

    if all_primary:
        verdict = "READY_FOR_NEW_POF_MANUSCRIPT"
        action = (
            "Stage-1 passes with existing macroscopic DSMC data. Rewrite as a new manuscript around "
            "accuracy-qualified, model-conditional predictive spatial support; Cp and Cq are primary, "
            "while shear is the quantity-dependence contrast."
        )
    else:
        cp_cq = [item for item in checks if item["primary_quantity"]]
        if all(item["loco_accuracy_pass"] and item["pairout_accuracy_pass"] for item in cp_cq):
            verdict = "ACCURATE_BUT_LOCALITY_CONTROL_INCOMPLETE"
            action = (
                "Prediction accuracy is adequate, but matched-count locality or permutation controls fail. "
                "Do not report a finite support; report bulk augmentation only or revise the operator."
            )
        else:
            verdict = "IMPROVE_STAGE1_MODEL_BEFORE_POF"
            action = (
                "At least one primary quantity misses its absolute bound. Inspect hard-case and apex errors; "
                "do not submit until the predeclared Stage-1 criteria are met."
            )

    decision: Dict[str, object] = {
        "verdict": verdict,
        "recommended_action": action,
        "terminology": "predictive spatial support; never physical/causal information horizon",
        "rules": {
            "primary_quantities": ["Cp", "Cq"],
            "loco_accuracy": "upper 95% grouped-bootstrap bound <= target tolerance",
            "pairout_accuracy": "mean grouped-holdout error <= target tolerance",
            "bulk_gain": f"paired ensemble gain >= {100*args.min_gain:.1f} pp with lower 95% bound > 0",
            "locality": "near three rings beat far three rings with lower 95% paired bound > 0",
            "negative_control": "real near-three alignment beats case-profile-permuted training",
            "radius_selection": "finite radius selected only on the inner physical group",
            "censoring": (
                f"M_full/right-censored fraction <= {args.max_censor_fraction:.2f} "
                "for each primary quantity and validation scheme"
            ),
        },
        "checks": checks,
    }
    (out_dir / "stage1_v2_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    lines = [
        "LEKZIAN–ROOHI STAGE-1 V2 DECISION",
        "=" * 39,
        "",
        verdict,
        "",
        action,
        "",
        "Checks:",
    ]
    for item in checks:
        lines.append(
            f"- {item['target']:7s}: LOCO_accuracy={item['loco_accuracy_pass']}, "
            f"pairout_accuracy={item['pairout_accuracy_pass']}, bulk_gain={item['bulk_gain_pass']}, "
            f"near_vs_far={item['near3_beats_far3_pass']}, permutation={item['real_beats_case_permutation_pass']}, "
            f"censoring={item['censoring_pass']}"
        )
    lines.extend(
        [
            "",
            "Cp and Cq are the primary bulk-support claims. Shear bulk gain is not required and must be reported honestly.",
        ]
    )
    (out_dir / "stage1_v2_decision.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    order = [
        "M0",
        *FINITE_CONFIGS,
        "M_selected",
        "M_far_K3",
        "M_interleaved_K3",
        "M_full",
        "M_permtrain_R0p5",
        "M_permtrain_full",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
    for row_i, scheme in enumerate(("loco", "pairout")):
        for col_i, target in enumerate(TARGETS):
            ax = axes[row_i, col_i]
            part = summary[(summary["scheme"] == scheme) & (summary["target"] == target)].set_index("config")
            configs = [name for name in order if name in part.index]
            y = np.asarray([100 * float(part.loc[name, "mean_relL2"]) for name in configs])
            lo = np.asarray([100 * float(part.loc[name, "bootstrap_CI2p5"]) for name in configs])
            hi = np.asarray([100 * float(part.loc[name, "bootstrap_CI97p5"]) for name in configs])
            x = np.arange(len(configs))
            ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), marker="o", capsize=3)
            ax.set_xticks(x)
            ax.set_xticklabels(configs, rotation=40, ha="right", fontsize=8)
            ax.set_ylabel("ensemble relative L2 error (%)")
            ax.set_title(f"{scheme}: {target}")
            ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "stage1_v2_error_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "stage1_v2_error_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def aggregate_results(args: argparse.Namespace) -> Dict[str, object]:
    out_dir = Path(args.out)
    task_root = Path(args.aggregate_task_root)
    expected_tasks = (27 + 9) * len(args.seeds_list) * len(TARGETS)
    completed_tasks = len(list(task_root.glob("task_*/TASK_COMPLETE")))
    if completed_tasks != expected_tasks:
        raise RuntimeError(
            f"Expected {expected_tasks} completed task markers, found {completed_tasks}."
        )
    metrics = concat_task_files(task_root, "case_metrics.csv")
    predictions = concat_task_files(task_root, "surface_predictions.csv")
    validation_predictions = concat_task_files(task_root, "validation_predictions.csv")
    attention = concat_task_files(task_root, "attention_profiles.csv")
    seed_diagnostics = concat_task_files(task_root, "seed_radius_diagnostic.csv")
    duplicate_key = ["scheme", "outer_group", "seed", "target", "config", "case_id"]
    if metrics.duplicated(duplicate_key).any():
        raise RuntimeError("Duplicate task metrics detected during aggregation.")
    metrics.to_csv(out_dir / "individual_seed_case_metrics.csv", index=False)
    predictions.to_csv(out_dir / "individual_seed_surface_predictions.csv", index=False)
    validation_predictions.to_csv(
        out_dir / "individual_seed_validation_predictions.csv", index=False
    )
    attention.to_csv(out_dir / "individual_seed_attention_profiles.csv", index=False)
    seed_diagnostics.to_csv(out_dir / "seed_radius_diagnostics.csv", index=False)
    inner_counts = validation_predictions.groupby(
        ["scheme", "outer_group", "target"]
    )["inner_group"].nunique()
    if int(inner_counts.max()) != 1:
        raise RuntimeError("Seeds used different inner validation groups.")
    validation_points, _ = ensemble_case_metrics(
        validation_predictions, expected_seeds=len(args.seeds_list)
    )
    selections = select_radii_from_validation(validation_points, args)
    selections.to_csv(out_dir / "inner_radius_selection.csv", index=False)
    validation_points.to_csv(
        out_dir / "ensemble_validation_predictions.csv", index=False
    )
    ensemble_points, ensemble_cases = ensemble_case_metrics(
        predictions, expected_seeds=len(args.seeds_list)
    )
    ensemble_points = append_selected_predictions(ensemble_points, selections)
    ensemble_cases = case_metrics_from_ensemble(ensemble_points)
    ensemble_points.to_csv(out_dir / "ensemble_surface_predictions.csv", index=False)
    ensemble_cases.to_csv(out_dir / "ensemble_case_metrics.csv", index=False)
    summary, gains = summarize_ensemble(ensemble_cases, args, out_dir)
    _, censoring = selected_radius_summary(selections, out_dir)
    decision = make_decision(summary, gains, censoring, args, out_dir)
    plot_summary(summary, out_dir)
    return decision


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def write_run_record(
    feature_path: Path,
    schema: v1.FeatureSchema,
    args: argparse.Namespace,
    out_dir: Path,
) -> None:
    payload = {
        "feature_sha256": v1._sha256_file(feature_path),
        "code_sha256": v1._sha256_file(Path(__file__).resolve()),
        "seeds": args.seeds_list,
        "cv": args.cv_list,
        "only_scheme": args.only_scheme,
        "only_outer_group": args.only_outer_group,
        "only_target": args.only_target,
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
        "aggregate_task_root": args.aggregate_task_root,
        "selection_tolerances": args.selection_tolerance_map,
        "selection_margin": args.selection_margin,
        "max_censor_fraction": args.max_censor_fraction,
    }
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    record = {
        "signature": signature,
        "payload": payload,
        "feature_table": str(feature_path),
        "schema": asdict(schema),
        "loco_tolerances": args.loco_tolerance_map,
        "pairout_tolerances": args.pairout_tolerance_map,
        "min_gain": args.min_gain,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "hostname": os.uname().nodename,
        "started_unix": time.time(),
    }
    path = out_dir / "stage1_v2_run_config.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("signature") != signature:
            raise RuntimeError("Output directory contains a different Stage-1 V2 signature.")
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-table", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cv", default="loco,pairout")
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--only-scheme", choices=["loco", "pairout"])
    parser.add_argument("--only-outer-group")
    parser.add_argument("--only-target", choices=list(TARGETS))
    parser.add_argument("--aggregate-task-root")
    parser.add_argument("--hidden", type=int, default=160)
    parser.add_argument("--latent", type=int, default=96)
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
    parser.add_argument("--loco-tolerances", default="Cp=0.10,Cq=0.10,tau_abs=0.20")
    parser.add_argument("--pairout-tolerances", default="Cp=0.15,Cq=0.15,tau_abs=0.25")
    parser.add_argument("--selection-tolerances", default="Cp=0.15,Cq=0.15,tau_abs=0.25")
    parser.add_argument("--selection-margin-pp", type=float, default=1.0)
    parser.add_argument("--max-censor-fraction", type=float, default=0.20)
    parser.add_argument("--min-gain-pp", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.cv_list = [token.strip() for token in args.cv.split(",") if token.strip()]
    args.seeds_list = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    args.loco_tolerance_map = parse_target_map(args.loco_tolerances)
    args.pairout_tolerance_map = parse_target_map(args.pairout_tolerances)
    args.selection_tolerance_map = parse_target_map(args.selection_tolerances)
    args.selection_margin = args.selection_margin_pp / 100.0
    args.min_gain = args.min_gain_pp / 100.0
    if args.selection_margin < 0:
        raise ValueError("Selection noninferiority margin must be non-negative.")
    if not 0.0 <= args.max_censor_fraction <= 1.0:
        raise ValueError("Maximum censoring fraction must lie in [0, 1].")
    args.device_obj = resolve_device(args.device)
    if args.only_target and len(args.seeds_list) != 1:
        raise ValueError("A worker task must receive exactly one seed.")

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = Path(args.feature_table).expanduser().resolve()
    frame = pd.read_csv(feature_path)
    frame["case_id"] = frame["case_id"].astype(str)
    schema = v1.infer_schema(frame)
    v1.validate_table(frame, schema)
    mask_configs(schema)
    ring_column_groups(schema)
    write_run_record(feature_path, schema, args, out_dir)

    if args.aggregate_task_root:
        decision = aggregate_results(args)
        print(f"[DONE] verdict={decision['verdict']}", flush=True)
    else:
        run_one_task(frame, schema, args)


if __name__ == "__main__":
    main()
