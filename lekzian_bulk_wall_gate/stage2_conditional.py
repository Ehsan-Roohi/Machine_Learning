#!/usr/bin/env python3
"""Conditional-increment Stage-2 for the Lekzian--Roohi bulk/wall study.

Stage-1 V2 found useful bulk augmentation but no identifiable radial locality:
near, far, interleaved, and fixed-permutation controls were statistically
indistinguishable, and full-ring attention was nearly uniform.  This script is
the prospective existing-data follow-up.  It requires no new SPARTA output.

The design separates three questions that the earlier end-to-end operator mixed:

1. What can a strong parameter/surface-only wall surrogate predict?
2. Does conditionally residualized bulk flow improve that frozen baseline?
3. Is the improvement tied to radial proximity and correct surface alignment?

Only an accuracy-qualified, model-conditional predictive-support claim is
permitted.  A physical or causal information horizon is never inferred.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import stage1_v2 as s1


v1 = s1.v1
TARGETS = s1.TARGETS
FINITE_CONFIGS = ("C_R0p1", "C_R0p25", "C_R0p5", "C_R1", "C_R2")
REAL_CONFIGS = (
    "C0",
    *FINITE_CONFIGS,
    "C_far_K3",
    "C_interleaved_K3",
    "C_full",
)
CONTROL_CONFIGS = ("C_shift_R0p5", "C_radialperm_R0p5")
ATTENTION_CONFIGS = ("C_R0p5", "C_full")
NONPHYSICAL_SUFFIXES = {
    "mean_r_hs",
    "mean_rn",
    "mean_rt",
    "npts",
    "std_r_hs",
}


def config_masks(schema: v1.FeatureSchema) -> Dict[str, np.ndarray]:
    base = s1.mask_configs(schema)
    result: Dict[str, np.ndarray] = {
        "C0": base["M0"],
        "C_far_K3": base["M_far_K3"],
        "C_interleaved_K3": base["M_interleaved_K3"],
        "C_full": base["M_full"],
    }
    for old, new in zip(s1.FINITE_CONFIGS, FINITE_CONFIGS):
        result[new] = base[old]
    return {name: result[name] for name in REAL_CONFIGS}


def radius_value(config: str) -> float:
    if config == "C_full":
        return math.inf
    return float(config[3:].replace("p", "."))


def physical_suffix_indices(schema: v1.FeatureSchema) -> Tuple[List[int], List[str]]:
    groups = s1.ring_column_groups(schema)
    suffixes_by_ring: List[List[str]] = []
    for ring_i, group in enumerate(groups):
        prefix = f"{schema.ring_names[ring_i]}_"
        suffixes_by_ring.append(
            [str(schema.bulk_columns[col_i])[len(prefix) :] for col_i in group]
        )
    if any(values != suffixes_by_ring[0] for values in suffixes_by_ring[1:]):
        raise ValueError("Annuli do not share an identical ordered descriptor schema.")
    keep = [
        index
        for index, suffix in enumerate(suffixes_by_ring[0])
        if suffix not in NONPHYSICAL_SUFFIXES
    ]
    names = [suffixes_by_ring[0][index] for index in keep]
    if len(keep) < 20:
        raise ValueError("Too few physical descriptor channels remain after structural exclusions.")
    return keep, names


def base_arrays(
    frame: pd.DataFrame,
    schema: v1.FeatureSchema,
    prep: v1.Preprocessor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    case, surface, all_rings = s1.transform_arrays(frame, schema, prep)
    keep, _ = physical_suffix_indices(schema)
    return case, surface, all_rings[:, :, keep].astype(np.float32)


def design_matrix(case: np.ndarray, surface: np.ndarray) -> np.ndarray:
    x = np.column_stack([case, surface]).astype(np.float64)
    return np.column_stack([np.ones(len(x)), x])


@dataclass
class ConditionalResidualizer:
    coefficient: np.ndarray
    residual_scale: np.ndarray
    ridge: float

    @classmethod
    def fit(
        cls,
        case: np.ndarray,
        surface: np.ndarray,
        rings: np.ndarray,
        ridge: float,
    ) -> "ConditionalResidualizer":
        x = design_matrix(case, surface)
        y = rings.reshape(len(rings), -1).astype(np.float64)
        penalty = np.eye(x.shape[1], dtype=np.float64) * float(ridge)
        penalty[0, 0] = 0.0
        coefficient = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        residual = y - x @ coefficient
        scale = residual.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        return cls(coefficient, scale, float(ridge))

    def transform(
        self,
        case: np.ndarray,
        surface: np.ndarray,
        rings: np.ndarray,
    ) -> np.ndarray:
        x = design_matrix(case, surface)
        shape = rings.shape
        residual = rings.reshape(len(rings), -1) - x @ self.coefficient
        residual = residual / self.residual_scale
        return residual.reshape(shape).astype(np.float32)


def conditional_contrast_tensor(conditional: np.ndarray) -> np.ndarray:
    outer_excess = conditional - conditional[:, -1:, :]
    inward_jump = np.zeros_like(conditional)
    inward_jump[:, :-1, :] = conditional[:, :-1, :] - conditional[:, 1:, :]
    return np.concatenate([conditional, outer_excess, inward_jump], axis=2).astype(
        np.float32
    )


def ring_redundancy(rings: np.ndarray) -> Dict[str, object]:
    values = rings.transpose(1, 0, 2).reshape(rings.shape[1], -1)
    correlation = np.nan_to_num(
        np.corrcoef(values), nan=0.0, posinf=0.0, neginf=0.0
    )
    off_diagonal = np.abs(correlation[~np.eye(len(correlation), dtype=bool)])
    singular = np.linalg.svd(values - values.mean(axis=1, keepdims=True), compute_uv=False)
    energy = singular**2
    probability = energy / max(float(energy.sum()), 1e-12)
    entropy_rank = float(np.exp(-np.sum(probability * np.log(probability + 1e-12))))
    return {
        "mean_absolute_offdiagonal_correlation": float(np.nanmean(off_diagonal)),
        "max_absolute_offdiagonal_correlation": float(np.nanmax(off_diagonal)),
        "effective_rank": entropy_rank,
        "ring_correlation": correlation.tolist(),
    }


def stage2_weights(frame: pd.DataFrame, apex_boost: float, apex_sigma: float) -> np.ndarray:
    base = frame["gate_sample_weight"].to_numpy(np.float32)
    s = frame["s01"].to_numpy(np.float32)
    apex = 1.0 + float(apex_boost) * np.exp(-0.5 * ((s - 0.5) / apex_sigma) ** 2)
    weight = (base * apex).reshape(-1, 1).astype(np.float32)
    return weight / max(float(weight.mean()), 1e-8)


class StructuredBaseline(nn.Module):
    """Target-specific version of the manuscript's strong wall surrogate."""

    def __init__(
        self,
        case_dim: int,
        surface_dim: int,
        latent: int,
        hidden: int,
        depth: int,
        dropout: float,
        full_residual_scale: float,
    ):
        super().__init__()
        if case_dim < 7:
            raise ValueError("Structured baseline requires at least seven case features.")
        self.full_residual_scale = float(full_residual_scale)
        self.base_branch = s1.MLP(5, latent, hidden, depth, dropout)
        self.h_branch = s1.MLP(4, latent, hidden, depth, dropout)
        self.tw_branch = s1.MLP(4, latent, hidden, depth, dropout)
        self.full_branch = s1.MLP(case_dim, latent, hidden, max(1, depth - 1), dropout)
        self.surface_encoder = s1.MLP(surface_dim, latent, hidden, depth, dropout)
        self.wind_decoder = s1.MLP(latent, 1, hidden, depth, dropout)
        self.lee_decoder = s1.MLP(latent, 1, hidden, depth, dropout)
        self.surface_gate = s1.MLP(surface_dim, 1, max(32, hidden // 2), 2, 0.0)

    def context(self, case: torch.Tensor, surface: torch.Tensor) -> torch.Tensor:
        ma_log_geom = torch.cat([case[:, 0:2], case[:, 4:7]], dim=1)
        log_geom = torch.cat([case[:, 1:2], case[:, 4:7]], dim=1)
        branch = self.base_branch(ma_log_geom)
        branch = branch + case[:, 2:3] * self.h_branch(log_geom)
        branch = branch + case[:, 3:4] * self.tw_branch(log_geom)
        if self.full_residual_scale > 0:
            branch = branch + self.full_residual_scale * self.full_branch(case)
        return branch * self.surface_encoder(surface)

    def decode(self, context: torch.Tensor, surface: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.surface_gate(surface))
        return gate * self.wind_decoder(context) + (1.0 - gate) * self.lee_decoder(context)

    def forward(self, case: torch.Tensor, surface: torch.Tensor) -> torch.Tensor:
        return self.decode(self.context(case, surface), surface)


class ConditionalCorrection(nn.Module):
    """Capacity-matched correction on a frozen wall baseline."""

    def __init__(
        self,
        context_dim: int,
        ring_dim: int,
        n_rings: int,
        latent: int,
        hidden: int,
        depth: int,
        dropout: float,
    ):
        super().__init__()
        self.n_rings = int(n_rings)
        self.latent = int(latent)
        self.context_adapter = nn.Linear(context_dim, latent)
        self.ring_encoder = s1.MLP(ring_dim, latent, hidden, max(1, depth - 1), dropout)
        self.ring_position = nn.Parameter(torch.zeros(n_rings, latent))
        nn.init.normal_(self.ring_position, std=0.03)
        self.query = nn.Linear(latent, latent)
        self.key = nn.Linear(latent, latent)
        self.value = nn.Linear(latent, latent)
        self.context_only = s1.MLP(latent, 1, hidden, depth, dropout)
        self.bulk_increment = s1.MLP(2 * latent + 1, 1, hidden, depth, dropout)

    def forward(
        self,
        context: torch.Tensor,
        rings: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        context = torch.tanh(self.context_adapter(context))
        batch, n_rings, ring_dim = rings.shape
        encoded = self.ring_encoder(rings.reshape(batch * n_rings, ring_dim)).reshape(
            batch, n_rings, self.latent
        )
        encoded = encoded + self.ring_position.unsqueeze(0)
        score = (
            self.query(context).unsqueeze(1) * self.key(encoded)
        ).sum(dim=-1) / math.sqrt(self.latent)
        visible = mask.to(score.dtype)
        shifted = score - score.max(dim=1, keepdim=True).values
        attention = torch.exp(shifted) * visible
        attention = attention / (attention.sum(dim=1, keepdim=True) + 1e-8)
        aggregate = (attention.unsqueeze(-1) * self.value(encoded)).sum(dim=1)
        fraction = visible.mean(dim=1, keepdim=True)
        has_bulk = visible.any(dim=1, keepdim=True).to(context.dtype)
        correction = self.context_only(context)
        correction = correction + has_bulk * self.bulk_increment(
            torch.cat([context, aggregate, fraction], dim=1)
        )
        if return_attention:
            return correction, attention
        return correction


def train_baseline(
    train_arrays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    y_train: np.ndarray,
    w_train: np.ndarray,
    val_arrays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    y_val: np.ndarray,
    w_val: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[StructuredBaseline, int, float, int]:
    v1._seed_everything(seed)
    case_train, surface_train, _ = train_arrays
    case_val, surface_val, _ = val_arrays
    model = StructuredBaseline(
        case_train.shape[1],
        surface_train.shape[1],
        args.latent,
        args.hidden,
        args.depth,
        args.dropout,
        args.full_residual_scale,
    ).to(args.device_obj)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(case_train),
            torch.from_numpy(surface_train),
            torch.from_numpy(y_train),
            torch.from_numpy(w_train),
        ),
        batch_size=min(args.batch_size, len(case_train)),
        shuffle=True,
        generator=generator,
    )
    cval = torch.from_numpy(case_val).to(args.device_obj)
    sval = torch.from_numpy(surface_val).to(args.device_obj)
    yval = torch.from_numpy(y_val).to(args.device_obj)
    wval = torch.from_numpy(w_val).to(args.device_obj)
    best_state: Optional[Mapping[str, torch.Tensor]] = None
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.baseline_epochs + 1):
        model.train()
        for case_b, surface_b, y_b, w_b in loader:
            case_b = case_b.to(args.device_obj)
            surface_b = surface_b.to(args.device_obj)
            y_b = y_b.to(args.device_obj)
            w_b = w_b.to(args.device_obj)
            optimizer.zero_grad(set_to_none=True)
            loss = s1.weighted_huber(model(case_b, surface_b), y_b, w_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            value = float(s1.weighted_huber(model(cval, sval), yval, wval).cpu())
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
        raise RuntimeError("Baseline training produced no finite validation checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, best_epoch, best_loss, parameter_count


def baseline_latents(
    model: StructuredBaseline,
    arrays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    case, surface, _ = arrays
    predictions: List[np.ndarray] = []
    contexts: List[np.ndarray] = []
    step = max(args.batch_size, 4096)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(case), step):
            stop = min(len(case), start + step)
            c = torch.from_numpy(case[start:stop]).to(args.device_obj)
            s = torch.from_numpy(surface[start:stop]).to(args.device_obj)
            context = model.context(c, s)
            predictions.append(model.decode(context, s).cpu().numpy())
            contexts.append(context.cpu().numpy())
    return np.vstack(predictions).astype(np.float32), np.vstack(contexts).astype(np.float32)


def train_correction(
    context_train: np.ndarray,
    base_train: np.ndarray,
    rings_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    context_val: np.ndarray,
    base_val: np.ndarray,
    rings_val: np.ndarray,
    y_val: np.ndarray,
    w_val: np.ndarray,
    masks: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    seed: int,
) -> Tuple[ConditionalCorrection, int, float, int]:
    v1._seed_everything(seed)
    model = ConditionalCorrection(
        context_train.shape[1],
        rings_train.shape[2],
        rings_train.shape[1],
        args.latent,
        args.hidden,
        args.depth,
        args.dropout,
    ).to(args.device_obj)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.correction_lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(context_train),
            torch.from_numpy(base_train),
            torch.from_numpy(rings_train),
            torch.from_numpy(y_train),
            torch.from_numpy(w_train),
        ),
        batch_size=min(args.batch_size, len(context_train)),
        shuffle=True,
        generator=generator,
    )
    cv = torch.from_numpy(context_val).to(args.device_obj)
    bv = torch.from_numpy(base_val).to(args.device_obj)
    rv = torch.from_numpy(rings_val).to(args.device_obj)
    yv = torch.from_numpy(y_val).to(args.device_obj)
    wv = torch.from_numpy(w_val).to(args.device_obj)
    variable = [name for name in REAL_CONFIGS if name not in {"C0", "C_full"}]
    best_state: Optional[Mapping[str, torch.Tensor]] = None
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.correction_epochs + 1):
        model.train()
        for batch_i, (context_b, base_b, rings_b, y_b, w_b) in enumerate(loader):
            context_b = context_b.to(args.device_obj)
            base_b = base_b.to(args.device_obj)
            rings_b = rings_b.to(args.device_obj)
            y_b = y_b.to(args.device_obj)
            w_b = w_b.to(args.device_obj)
            chosen = variable[(epoch + batch_i) % len(variable)]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for name in ("C0", chosen, "C_full"):
                mask = s1.torch_mask(masks[name], len(context_b), args.device_obj)
                final = base_b + model(context_b, rings_b, mask)
                losses.append(s1.weighted_huber(final, y_b, w_b))
            loss = torch.stack(losses).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            values = []
            for name in ("C0", "C_R0p5", "C_full"):
                mask = s1.torch_mask(masks[name], len(cv), args.device_obj)
                values.append(s1.weighted_huber(bv + model(cv, rv, mask), yv, wv))
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
        raise RuntimeError("Correction training produced no finite validation checkpoint.")
    model.load_state_dict(best_state)
    return model, best_epoch, best_loss, parameter_count


def predict_correction(
    model: ConditionalCorrection,
    context: np.ndarray,
    base: np.ndarray,
    rings: np.ndarray,
    mask: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    predictions: List[np.ndarray] = []
    attentions: List[np.ndarray] = []
    step = max(args.batch_size, 4096)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(context), step):
            stop = min(len(context), start + step)
            c = torch.from_numpy(context[start:stop]).to(args.device_obj)
            b = torch.from_numpy(base[start:stop]).to(args.device_obj)
            r = torch.from_numpy(rings[start:stop]).to(args.device_obj)
            m = s1.torch_mask(mask, stop - start, args.device_obj)
            correction, attention = model(c, r, m, return_attention=True)
            predictions.append((b + correction).cpu().numpy().reshape(-1))
            attentions.append(attention.cpu().numpy())
    return np.concatenate(predictions), np.vstack(attentions)


def shifted_ring_tensor(
    frame: pd.DataFrame,
    rings: np.ndarray,
    fraction: float,
) -> np.ndarray:
    result = rings.copy()
    ids = frame["case_id"].astype(str).to_numpy()
    s = frame["s01"].to_numpy(float)
    for case_id in sorted(set(ids)):
        loc = np.flatnonzero(ids == case_id)
        loc = loc[np.argsort(s[loc])]
        shift = max(1, int(round(float(fraction) * len(loc)))) % len(loc)
        if shift == 0:
            shift = 1
        result[loc] = np.roll(rings[loc], shift, axis=0)
    return result


def control_predictions(
    model: ConditionalCorrection,
    context: np.ndarray,
    base: np.ndarray,
    rings: np.ndarray,
    frame: pd.DataFrame,
    masks: Mapping[str, np.ndarray],
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    shifted = []
    for fraction in np.linspace(0.11, 0.89, args.surface_shifts):
        tensor = shifted_ring_tensor(frame, rings, float(fraction))
        pred, _ = predict_correction(model, context, base, tensor, masks["C_R0p5"], args)
        shifted.append(pred)

    radial = []
    for permutation in itertools.permutations(range(3)):
        if permutation == (0, 1, 2):
            continue
        tensor = rings.copy()
        tensor[:, :3, :] = rings[:, permutation, :]
        pred, _ = predict_correction(model, context, base, tensor, masks["C_R0p5"], args)
        radial.append(pred)
    return {
        "C_shift_R0p5": np.mean(shifted, axis=0),
        "C_radialperm_R0p5": np.mean(radial, axis=0),
    }


def choose_validation_radius(
    predictions: Mapping[str, np.ndarray],
    truth: np.ndarray,
    case_ids: Sequence[str],
    absolute_tolerance: float,
    noninferiority_margin: float,
) -> Tuple[str, bool, pd.DataFrame]:
    rows = []
    for config in (*FINITE_CONFIGS, "C_full"):
        rows.append(
            {
                "config": config,
                "validation_relL2": s1.mean_profile_rel_l2(
                    predictions[config], truth, case_ids
                ),
                "radius_over_hs": radius_value(config),
            }
        )
    table = pd.DataFrame(rows)
    best = float(table["validation_relL2"].min())
    table["absolute_tolerance"] = float(absolute_tolerance)
    table["noninferiority_margin"] = float(noninferiority_margin)
    table["qualified"] = (
        (table["validation_relL2"] <= absolute_tolerance)
        & (table["validation_relL2"] <= best + noninferiority_margin)
    )
    finite = table[(table["config"] != "C_full") & table["qualified"]].sort_values(
        "radius_over_hs"
    )
    selected = "C_full" if finite.empty else str(finite.iloc[0]["config"])
    censored = bool(finite.empty)
    table["selected_config"] = selected
    table["radius_censored"] = censored
    return selected, censored, table.sort_values("radius_over_hs")


def prediction_rows(
    frame: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    truth: np.ndarray,
    scheme: str,
    outer_group: str,
    seed: int,
    target: str,
    inner_group: Optional[str] = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row_i, row in frame.reset_index(drop=True).iterrows():
        for config, pred in predictions.items():
            item: Dict[str, object] = {
                "scheme": scheme,
                "outer_group": outer_group,
                "seed": seed,
                "target": target,
                "config": config,
                "case_id": str(row["case_id"]),
                "surface_i": int(row.get("surface_i", row_i)),
                "s01": float(row["s01"]),
                "true": float(truth[row_i]),
                "pred": float(pred[row_i]),
            }
            if inner_group is not None:
                item["inner_group"] = inner_group
            rows.append(item)
    return rows


def run_one_task(
    frame: pd.DataFrame,
    schema: v1.FeatureSchema,
    args: argparse.Namespace,
) -> None:
    if not args.only_scheme or not args.only_outer_group or not args.only_target:
        raise ValueError("Worker mode requires one scheme, outer group, and target.")
    scheme = args.only_scheme
    outer_group = args.only_outer_group
    target = args.only_target
    seed = args.seeds_list[0]
    out_dir = Path(args.out)
    marker = out_dir / "TASK_COMPLETE"
    if args.resume and marker.exists():
        print(f"[SKIP] completed task {out_dir}", flush=True)
        return

    groups = v1._outer_groups(frame, scheme).astype(str)
    if outer_group not in set(groups):
        raise ValueError(f"Unknown outer group {outer_group!r} for {scheme}.")
    test_mask = groups == outer_group
    outer_train = frame.loc[~test_mask].reset_index(drop=True)
    test_df = frame.loc[test_mask].reset_index(drop=True)
    train_groups = v1._outer_groups(outer_train, scheme).astype(str)
    inner_group = v1._choose_inner_group(
        train_groups,
        outer_group,
        s1.stable_seed("stage2_shared_inner", scheme, outer_group),
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

    raw_train = base_arrays(train_df, schema, prep)
    raw_val = base_arrays(val_df, schema, prep)
    raw_test = base_arrays(test_df, schema, prep)
    residualizer = ConditionalResidualizer.fit(
        raw_train[0], raw_train[1], raw_train[2], args.ridge
    )
    conditional_train = residualizer.transform(*raw_train)
    conditional_val = residualizer.transform(*raw_val)
    conditional_test = residualizer.transform(*raw_test)
    rings_train = conditional_contrast_tensor(conditional_train)
    rings_val = conditional_contrast_tensor(conditional_val)
    rings_test = conditional_contrast_tensor(conditional_test)
    arrays_train = (raw_train[0], raw_train[1], rings_train)
    arrays_val = (raw_val[0], raw_val[1], rings_val)
    arrays_test = (raw_test[0], raw_test[1], rings_test)
    masks = config_masks(schema)

    baseline_seed = s1.stable_seed("stage2_baseline", scheme, outer_group, target, seed)
    baseline, base_epoch, base_loss, base_parameters = train_baseline(
        arrays_train,
        y_train,
        stage2_weights(train_df, args.apex_boost, args.apex_sigma),
        arrays_val,
        y_val,
        stage2_weights(val_df, args.apex_boost, args.apex_sigma),
        args,
        baseline_seed,
    )
    base_train, context_train = baseline_latents(baseline, arrays_train, args)
    base_val, context_val = baseline_latents(baseline, arrays_val, args)
    base_test, context_test = baseline_latents(baseline, arrays_test, args)
    correction_seed = s1.stable_seed("stage2_correction", scheme, outer_group, target, seed)
    correction, correction_epoch, correction_loss, correction_parameters = train_correction(
        context_train,
        base_train,
        rings_train,
        y_train,
        stage2_weights(train_df, args.apex_boost, args.apex_sigma),
        context_val,
        base_val,
        rings_val,
        y_val,
        stage2_weights(val_df, args.apex_boost, args.apex_sigma),
        masks,
        args,
        correction_seed,
    )

    val_predictions: Dict[str, np.ndarray] = {}
    for config in REAL_CONFIGS:
        pred_z, _ = predict_correction(
            correction, context_val, base_val, rings_val, masks[config], args
        )
        val_predictions[config] = pred_z * y_std + y_mean
    diagnostic_config, diagnostic_censored, diagnostic_table = choose_validation_radius(
        val_predictions,
        truth_val,
        val_df["case_id"].astype(str).to_numpy(),
        args.selection_tolerance_map[target],
        args.selection_margin,
    )

    test_predictions: Dict[str, np.ndarray] = {
        "B0": base_test.reshape(-1) * y_std + y_mean
    }
    attention: Dict[str, np.ndarray] = {}
    for config in REAL_CONFIGS:
        pred_z, weights = predict_correction(
            correction, context_test, base_test, rings_test, masks[config], args
        )
        test_predictions[config] = pred_z * y_std + y_mean
        attention[config] = weights
    for config, pred_z in control_predictions(
        correction,
        context_test,
        base_test,
        rings_test,
        test_df,
        masks,
        args,
    ).items():
        test_predictions[config] = pred_z * y_std + y_mean

    ids = test_df["case_id"].astype(str).to_numpy()
    metric_rows: List[Dict[str, object]] = []
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
                    "relL2": s1.rel_l2(pred[loc], truth_test[loc]),
                    "rangeMAE_pct": s1.range_mae_pct(pred[loc], truth_test[loc]),
                    "baseline_best_epoch": base_epoch,
                    "baseline_inner_loss": base_loss,
                    "correction_best_epoch": correction_epoch,
                    "correction_inner_loss": correction_loss,
                    "baseline_parameters": base_parameters,
                    "correction_parameters": correction_parameters,
                    "conditional_ring_dimension": rings_test.shape[2],
                }
            )

    attention_rows: List[Dict[str, object]] = []
    for row_i, row in test_df.reset_index(drop=True).iterrows():
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
                        "attention": float(attention[config][row_i, ring_i]),
                    }
                )

    redundancy = {
        "raw_physical_rings": ring_redundancy(raw_train[2]),
        "conditional_rings": ring_redundancy(conditional_train),
        "conditional_contrast_tensor": ring_redundancy(rings_train),
        "excluded_nonphysical_suffixes": sorted(NONPHYSICAL_SUFFIXES),
        "physical_suffixes": physical_suffix_indices(schema)[1],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(out_dir / "case_metrics.csv", index=False)
    pd.DataFrame(
        prediction_rows(
            test_df,
            test_predictions,
            truth_test,
            scheme,
            outer_group,
            seed,
            target,
        )
    ).to_csv(out_dir / "surface_predictions.csv", index=False)
    pd.DataFrame(
        prediction_rows(
            val_df,
            val_predictions,
            truth_val,
            scheme,
            outer_group,
            seed,
            target,
            inner_group,
        )
    ).to_csv(out_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(attention_rows).to_csv(out_dir / "attention_profiles.csv", index=False)
    diagnostic_table.assign(
        scheme=scheme,
        outer_group=outer_group,
        inner_group=inner_group,
        seed=seed,
        target=target,
        diagnostic_selected_config=diagnostic_config,
        diagnostic_radius_censored=diagnostic_censored,
    ).to_csv(out_dir / "seed_radius_diagnostic.csv", index=False)
    (out_dir / "conditional_redundancy.json").write_text(
        json.dumps(redundancy, indent=2), encoding="utf-8"
    )
    marker.write_text("complete\n", encoding="utf-8")
    print(
        f"[DONE] {scheme}|{outer_group}|{seed}|{target}; "
        f"diagnostic={diagnostic_config}",
        flush=True,
    )


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
        for config in (*FINITE_CONFIGS, "C_full"):
            part = group[group["config"] == config].sort_values(["case_id", "surface_i"])
            if part.empty:
                raise RuntimeError(
                    f"Missing validation prediction for {scheme}/{outer_group}/{target}/{config}."
                )
            predictions[config] = part["pred"].to_numpy(float)
            current_truth = part["true"].to_numpy(float)
            if truth is None:
                truth = current_truth
                case_ids = part["case_id"].astype(str).to_numpy()
            elif truth.shape != current_truth.shape or not np.allclose(
                truth, current_truth, equal_nan=True
            ):
                raise RuntimeError("Validation truth changed across Stage-2 configurations.")
        assert truth is not None and case_ids is not None
        _, _, table = choose_validation_radius(
            predictions,
            truth,
            case_ids,
            args.selection_tolerance_map[str(target)],
            args.selection_margin,
        )
        rows.append(table.assign(scheme=scheme, outer_group=outer_group, target=target))
    return pd.concat(rows, ignore_index=True)


def append_selected_predictions(
    ensemble_points: pd.DataFrame,
    selections: pd.DataFrame,
) -> pd.DataFrame:
    selected = selections[selections["config"] == selections["selected_config"]][
        ["scheme", "outer_group", "target", "selected_config", "radius_censored"]
    ]
    if selected.duplicated(["scheme", "outer_group", "target"]).any():
        raise RuntimeError("Stage-2 selected more than one radius for an outer group.")
    pieces: List[pd.DataFrame] = []
    for row in selected.itertuples(index=False):
        part = ensemble_points[
            (ensemble_points["scheme"] == row.scheme)
            & (ensemble_points["outer_group"] == row.outer_group)
            & (ensemble_points["target"] == row.target)
            & (ensemble_points["config"] == row.selected_config)
        ].copy()
        if part.empty:
            raise RuntimeError(
                f"Missing selected test prediction for {row.scheme}/{row.outer_group}/{row.target}."
            )
        part["source_config"] = row.selected_config
        part["radius_censored"] = bool(row.radius_censored)
        part["config"] = "C_selected"
        pieces.append(part)
    return pd.concat([ensemble_points, *pieces], ignore_index=True)


def summarize_ensemble(
    case_metrics: pd.DataFrame,
    args: argparse.Namespace,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    units = s1.unit_metrics(case_metrics)
    summary_rows: List[Dict[str, object]] = []
    for (scheme, target, config), group in units.groupby(["scheme", "target", "config"]):
        mean, lo, hi = s1.bootstrap_units(
            group["relL2"].to_numpy(float),
            args.bootstrap,
            s1.stable_seed("stage2_summary", scheme, target, config),
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
    summary.to_csv(out_dir / "stage2_ensemble_summary.csv", index=False)

    comparisons = [
        ("B0", "C0"),
        ("C0", "C_selected"),
        ("C0", "C_R0p5"),
        ("C_far_K3", "C_R0p5"),
        ("C_interleaved_K3", "C_R0p5"),
        ("C_shift_R0p5", "C_R0p5"),
        ("C_radialperm_R0p5", "C_R0p5"),
        ("C_full", "C_selected"),
    ]
    gain_rows: List[Dict[str, object]] = []
    for (scheme, target), group in units.groupby(["scheme", "target"]):
        pivot = group.pivot(index="unit_id", columns="config", values="relL2")
        for baseline, candidate in comparisons:
            if baseline not in pivot or candidate not in pivot:
                continue
            difference = (pivot[baseline] - pivot[candidate]).dropna().to_numpy(float)
            mean, lo, hi = s1.bootstrap_units(
                difference,
                args.bootstrap,
                s1.stable_seed("stage2_gain", scheme, target, baseline, candidate),
            )
            gain_rows.append(
                {
                    "scheme": scheme,
                    "target": target,
                    "comparison": f"{baseline}-{candidate}",
                    "n_bootstrap_units": len(difference),
                    "mean_gain": mean,
                    "mean_gain_pp": 100.0 * mean,
                    "bootstrap_CI2p5": lo,
                    "bootstrap_CI97p5": hi,
                    "CI2p5_pp": 100.0 * lo,
                    "CI97p5_pp": 100.0 * hi,
                }
            )
    gains = pd.DataFrame(gain_rows)
    gains.to_csv(out_dir / "stage2_paired_gains.csv", index=False)
    return summary, gains


def selection_summaries(
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
        .agg(n_selections=("outer_group", "size"))
    )
    censoring = (
        chosen.groupby(["scheme", "target"], as_index=False)
        .agg(
            n_outer_groups=("outer_group", "nunique"),
            n_censored=("radius_censored", "sum"),
            fraction_censored=("radius_censored", "mean"),
        )
    )
    distribution.to_csv(out_dir / "stage2_radius_distribution.csv", index=False)
    censoring.to_csv(out_dir / "stage2_radius_censoring.csv", index=False)
    return distribution, censoring


def lookup(
    table: pd.DataFrame,
    scheme: str,
    target: str,
    column: str,
    value: str,
) -> pd.Series:
    row = table[
        (table["scheme"] == scheme)
        & (table["target"] == target)
        & (table[column] == value)
    ]
    if row.empty:
        raise RuntimeError(f"Missing Stage-2 row {scheme}/{target}/{value}")
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
    any_conditional_gain = False
    for target in TARGETS:
        loco = lookup(summary, "loco", target, "config", "C_selected")
        pairout = lookup(summary, "pairout", target, "config", "C_selected")
        loco_gain = lookup(gains, "loco", target, "comparison", "C0-C_selected")
        pair_gain = lookup(gains, "pairout", target, "comparison", "C0-C_selected")
        loco_locality = lookup(
            gains, "loco", target, "comparison", "C_far_K3-C_R0p5"
        )
        pair_locality = lookup(
            gains, "pairout", target, "comparison", "C_far_K3-C_R0p5"
        )
        loco_alignment = lookup(
            gains, "loco", target, "comparison", "C_shift_R0p5-C_R0p5"
        )
        pair_alignment = lookup(
            gains, "pairout", target, "comparison", "C_shift_R0p5-C_R0p5"
        )
        censor_rows = censoring[
            (censoring["target"] == target)
            & (censoring["scheme"].isin(["loco", "pairout"]))
        ]
        is_primary = target in {"Cp", "Cq"}
        accuracy = (
            float(loco["bootstrap_CI97p5"]) <= args.loco_tolerance_map[target]
            and float(pairout["mean_relL2"]) <= args.pairout_tolerance_map[target]
        )
        conditional_gain = (
            float(loco_gain["mean_gain"]) >= args.min_gain
            and float(loco_gain["bootstrap_CI2p5"]) > 0.0
            and float(pair_gain["mean_gain"]) >= args.min_gain
            and float(pair_gain["bootstrap_CI2p5"]) > 0.0
        )
        locality = (
            float(loco_locality["bootstrap_CI2p5"]) > 0.0
            and float(pair_locality["mean_gain"]) > 0.0
        )
        alignment = (
            float(loco_alignment["bootstrap_CI2p5"]) > 0.0
            and float(pair_alignment["mean_gain"]) > 0.0
        )
        censoring_pass = bool(
            len(censor_rows) == 2
            and (censor_rows["fraction_censored"] <= args.max_censor_fraction).all()
        )
        target_pass = accuracy
        if is_primary:
            target_pass = accuracy and conditional_gain and locality and alignment and censoring_pass
        if is_primary:
            all_primary = all_primary and target_pass
        any_conditional_gain = any_conditional_gain or conditional_gain
        checks.append(
            {
                "target": target,
                "primary_quantity": is_primary,
                "loco_mean_relL2": float(loco["mean_relL2"]),
                "loco_CI97p5": float(loco["bootstrap_CI97p5"]),
                "pairout_mean_relL2": float(pairout["mean_relL2"]),
                "accuracy_pass": accuracy,
                "conditional_gain_loco_pp": float(loco_gain["mean_gain_pp"]),
                "conditional_gain_pairout_pp": float(pair_gain["mean_gain_pp"]),
                "conditional_gain_pass": conditional_gain,
                "near_beats_far_pass": locality,
                "real_beats_surface_shift_pass": alignment,
                "censoring_pass": censoring_pass,
                "target_pass": target_pass,
            }
        )

    if all_primary:
        verdict = "READY_FOR_POF_REWRITE"
        action = (
            "Conditional bulk increments are accurate, improve a capacity-matched frozen baseline, "
            "and pass radial-locality and surface-alignment controls. Rewrite the paper around "
            "accuracy-qualified conditional predictive support."
        )
    elif any_conditional_gain:
        verdict = "BULK_INCREMENTAL_BUT_NOT_SPATIALLY_IDENTIFIED"
        action = (
            "Bulk descriptors add conditional predictive value, but proximity/alignment is not "
            "identified. Do not claim spatial support; pivot to conditional bulk-state augmentation."
        )
    else:
        verdict = "CONDITIONAL_SIGNAL_INADEQUATE"
        action = (
            "Existing macroscopic annuli do not provide a robust conditional increment. Stop the "
            "spatial-support route with this dataset rather than tuning thresholds post hoc."
        )
    decision: Dict[str, object] = {
        "verdict": verdict,
        "recommended_action": action,
        "terminology": "conditional predictive support; never physical/causal horizon",
        "prospective_rules": {
            "primary_quantities": ["Cp", "Cq"],
            "conditional_gain": (
                f"C0-C_selected >= {100*args.min_gain:.1f} pp with lower 95% bound > 0 "
                "in both schemes"
            ),
            "locality": "near-three beats far-three significantly in LOCO and has positive pair-out mean",
            "alignment": "real surface alignment beats cyclically shifted profiles significantly in LOCO and has positive pair-out mean",
            "censoring": f"right-censored fraction <= {args.max_censor_fraction:.2f}",
        },
        "checks": checks,
    }
    (out_dir / "stage2_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    lines = [
        "LEKZIAN--ROOHI CONDITIONAL STAGE-2 DECISION",
        "=" * 48,
        "",
        verdict,
        "",
        action,
        "",
        "Checks:",
    ]
    for item in checks:
        lines.append(
            f"- {item['target']:7s}: accuracy={item['accuracy_pass']}, "
            f"conditional_gain={item['conditional_gain_pass']}, "
            f"near_vs_far={item['near_beats_far_pass']}, "
            f"surface_alignment={item['real_beats_surface_shift_pass']}, "
            f"censoring={item['censoring_pass']}"
        )
    (out_dir / "stage2_decision.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return decision


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    order = [
        "B0",
        "C0",
        *FINITE_CONFIGS,
        "C_selected",
        "C_far_K3",
        "C_interleaved_K3",
        "C_full",
        *CONTROL_CONFIGS,
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), squeeze=False)
    for row_i, scheme in enumerate(("loco", "pairout")):
        for col_i, target in enumerate(TARGETS):
            ax = axes[row_i, col_i]
            part = summary[
                (summary["scheme"] == scheme) & (summary["target"] == target)
            ].set_index("config")
            configs = [name for name in order if name in part.index]
            y = np.asarray([100 * float(part.loc[name, "mean_relL2"]) for name in configs])
            lo = np.asarray([100 * float(part.loc[name, "bootstrap_CI2p5"]) for name in configs])
            hi = np.asarray([100 * float(part.loc[name, "bootstrap_CI97p5"]) for name in configs])
            x = np.arange(len(configs))
            ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), marker="o", capsize=3)
            ax.set_xticks(x)
            ax.set_xticklabels(configs, rotation=42, ha="right", fontsize=8)
            ax.set_ylabel("ensemble relative L2 error (%)")
            ax.set_title(f"{scheme}: {target}")
            ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "stage2_error_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "stage2_error_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def aggregate_redundancy(task_root: Path, out_dir: Path) -> None:
    rows = []
    for path in sorted(task_root.glob("task_*/conditional_redundancy.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for representation in (
            "raw_physical_rings",
            "conditional_rings",
            "conditional_contrast_tensor",
        ):
            item = payload[representation]
            rows.append(
                {
                    "task": path.parent.name,
                    "representation": representation,
                    "mean_absolute_offdiagonal_correlation": item[
                        "mean_absolute_offdiagonal_correlation"
                    ],
                    "max_absolute_offdiagonal_correlation": item[
                        "max_absolute_offdiagonal_correlation"
                    ],
                    "effective_rank": item["effective_rank"],
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "stage2_redundancy_summary.csv", index=False)


def aggregate_results(args: argparse.Namespace) -> Dict[str, object]:
    out_dir = Path(args.out)
    task_root = Path(args.aggregate_task_root)
    expected = (27 + 9) * len(args.seeds_list) * len(TARGETS)
    completed = len(list(task_root.glob("task_*/TASK_COMPLETE")))
    if completed != expected:
        raise RuntimeError(f"Expected {expected} Stage-2 tasks, found {completed} complete markers.")
    metrics = s1.concat_task_files(task_root, "case_metrics.csv")
    predictions = s1.concat_task_files(task_root, "surface_predictions.csv")
    validation = s1.concat_task_files(task_root, "validation_predictions.csv")
    attention = s1.concat_task_files(task_root, "attention_profiles.csv")
    diagnostics = s1.concat_task_files(task_root, "seed_radius_diagnostic.csv")
    duplicate_key = ["scheme", "outer_group", "seed", "target", "config", "case_id"]
    if metrics.duplicated(duplicate_key).any():
        raise RuntimeError("Duplicate Stage-2 task metrics detected.")
    metrics.to_csv(out_dir / "stage2_individual_seed_case_metrics.csv", index=False)
    predictions.to_csv(out_dir / "stage2_individual_seed_surface_predictions.csv", index=False)
    validation.to_csv(out_dir / "stage2_individual_seed_validation_predictions.csv", index=False)
    attention.to_csv(out_dir / "stage2_attention_profiles.csv", index=False)
    diagnostics.to_csv(out_dir / "stage2_seed_radius_diagnostics.csv", index=False)
    inner_counts = validation.groupby(["scheme", "outer_group", "target"])[
        "inner_group"
    ].nunique()
    if int(inner_counts.max()) != 1:
        raise RuntimeError("Stage-2 seeds used different inner validation groups.")
    validation_points, _ = s1.ensemble_case_metrics(
        validation, expected_seeds=len(args.seeds_list)
    )
    selections = select_radii_from_validation(validation_points, args)
    selections.to_csv(out_dir / "stage2_inner_radius_selection.csv", index=False)
    ensemble_points, _ = s1.ensemble_case_metrics(
        predictions, expected_seeds=len(args.seeds_list)
    )
    ensemble_points = append_selected_predictions(ensemble_points, selections)
    ensemble_cases = s1.case_metrics_from_ensemble(ensemble_points)
    ensemble_points.to_csv(out_dir / "stage2_ensemble_surface_predictions.csv", index=False)
    ensemble_cases.to_csv(out_dir / "stage2_ensemble_case_metrics.csv", index=False)
    summary, gains = summarize_ensemble(ensemble_cases, args, out_dir)
    _, censoring = selection_summaries(selections, out_dir)
    aggregate_redundancy(task_root, out_dir)
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
        "baseline_epochs": args.baseline_epochs,
        "correction_epochs": args.correction_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "correction_lr": args.correction_lr,
        "weight_decay": args.weight_decay,
        "ridge": args.ridge,
        "apex_boost": args.apex_boost,
        "apex_sigma": args.apex_sigma,
        "surface_shifts": args.surface_shifts,
        "selection_tolerances": args.selection_tolerance_map,
        "selection_margin": args.selection_margin,
        "max_censor_fraction": args.max_censor_fraction,
        "loco_tolerances": args.loco_tolerance_map,
        "pairout_tolerances": args.pairout_tolerance_map,
        "min_gain": args.min_gain,
        "bootstrap": args.bootstrap,
    }
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    record = {
        "signature": signature,
        "payload": payload,
        "feature_table": str(feature_path),
        "schema": asdict(schema),
        "physical_suffixes": physical_suffix_indices(schema)[1],
        "excluded_nonphysical_suffixes": sorted(NONPHYSICAL_SUFFIXES),
        "loco_tolerances": args.loco_tolerance_map,
        "pairout_tolerances": args.pairout_tolerance_map,
        "min_gain": args.min_gain,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "hostname": os.uname().nodename,
        "started_unix": time.time(),
    }
    path = out_dir / "stage2_run_config.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("signature") != signature:
            raise RuntimeError("Output directory contains a different Stage-2 signature.")
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
    parser.add_argument("--baseline-epochs", type=int, default=350)
    parser.add_argument("--correction-epochs", type=int, default=350)
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--correction-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--apex-boost", type=float, default=3.0)
    parser.add_argument("--apex-sigma", type=float, default=0.07)
    parser.add_argument("--surface-shifts", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--loco-tolerances", default="Cp=0.10,Cq=0.10,tau_abs=0.20")
    parser.add_argument("--pairout-tolerances", default="Cp=0.15,Cq=0.15,tau_abs=0.25")
    parser.add_argument("--selection-tolerances", default="Cp=0.15,Cq=0.15,tau_abs=0.25")
    parser.add_argument("--selection-margin-pp", type=float, default=1.0)
    parser.add_argument("--max-censor-fraction", type=float, default=0.20)
    parser.add_argument("--min-gain-pp", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.cv_list = [token.strip() for token in args.cv.split(",") if token.strip()]
    args.seeds_list = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    args.loco_tolerance_map = s1.parse_target_map(args.loco_tolerances)
    args.pairout_tolerance_map = s1.parse_target_map(args.pairout_tolerances)
    args.selection_tolerance_map = s1.parse_target_map(args.selection_tolerances)
    args.selection_margin = args.selection_margin_pp / 100.0
    args.min_gain = args.min_gain_pp / 100.0
    if args.ridge < 0 or args.apex_sigma <= 0 or args.surface_shifts < 2:
        raise ValueError("Invalid residualization, apex, or surface-shift setting.")
    if not 0.0 <= args.max_censor_fraction <= 1.0:
        raise ValueError("Maximum censoring fraction must lie in [0,1].")
    if args.only_target and len(args.seeds_list) != 1:
        raise ValueError("A Stage-2 worker must receive exactly one seed.")
    args.device_obj = resolve_device(args.device)

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = Path(args.feature_table).expanduser().resolve()
    frame = pd.read_csv(feature_path)
    frame["case_id"] = frame["case_id"].astype(str)
    schema = v1.infer_schema(frame)
    v1.validate_table(frame, schema)
    config_masks(schema)
    physical_suffix_indices(schema)
    write_run_record(feature_path, schema, args, out_dir)
    if args.aggregate_task_root:
        decision = aggregate_results(args)
        print(f"[DONE] Stage-2 verdict={decision['verdict']}", flush=True)
    else:
        run_one_task(frame, schema, args)


if __name__ == "__main__":
    main()
