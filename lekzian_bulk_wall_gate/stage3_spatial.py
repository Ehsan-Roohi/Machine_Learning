#!/usr/bin/env python3
"""Compact wall-aligned spatial operator for the existing Lekzian DSMC archive.

Stage 3 replaces lossy annular statistics with fixed wall-aligned patches of the
already available ``u, v, T, logP`` fields.  One capacity-matched multi-target
model evaluates context-only, near, far, upstream, downstream, full, shifted,
and radial-flip inputs under nested grouped validation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TARGETS = ("Cp", "Cq", "tau_abs")
REAL_CONFIGS = ("P0", "P_near", "P_far", "P_upstream", "P_downstream", "P_full")
CONTROL_CONFIGS = ("P_shift", "P_radial_flip")
ALL_CONFIGS = (*REAL_CONFIGS, *CONTROL_CONFIGS)


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel_l2(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.linalg.norm(np.asarray(pred) - np.asarray(true)) / max(np.linalg.norm(true), eps))


def bootstrap_mean(values: np.ndarray, draws: int, seed: int) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    chunk = max(1, min(draws, 2000))
    samples = []
    remaining = draws
    while remaining:
        count = min(chunk, remaining)
        index = rng.integers(0, len(values), size=(count, len(values)))
        samples.append(values[index].mean(axis=1))
        remaining -= count
    means = np.concatenate(samples)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


@dataclass
class SpatialData:
    patches: np.ndarray
    context: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    case_id: np.ndarray
    surface_i: np.ndarray
    s01: np.ndarray
    Ma: np.ndarray
    Kn: np.ndarray
    geom: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    t_grid: np.ndarray
    n_grid: np.ndarray
    target_scale_exponents: np.ndarray

    def take(self, index: np.ndarray) -> "SpatialData":
        per_sample = {
            "patches", "context", "targets", "weights", "case_id", "surface_i",
            "s01", "Ma", "Kn", "geom", "tangent", "normal",
        }
        values = {
            name: (getattr(self, name)[index] if name in per_sample else getattr(self, name))
            for name in self.__dataclass_fields__
        }
        return SpatialData(**values)


def load_dataset(path: Path) -> SpatialData:
    with np.load(path, allow_pickle=False) as data:
        target_names = tuple(map(str, data["target_names"]))
        if target_names != TARGETS:
            raise RuntimeError(f"Unexpected targets {target_names}; expected {TARGETS}.")
        result = SpatialData(
            patches=np.asarray(data["patches"], dtype=np.float32),
            context=np.asarray(data["context"], dtype=np.float32),
            targets=np.asarray(data["targets"], dtype=np.float32),
            weights=np.asarray(data["weights"], dtype=np.float32),
            case_id=np.asarray(data["case_id"]).astype(str),
            surface_i=np.asarray(data["surface_i"], dtype=np.int32),
            s01=np.asarray(data["s01"], dtype=np.float32),
            Ma=np.asarray(data["Ma"], dtype=np.float32),
            Kn=np.asarray(data["Kn"], dtype=np.float32),
            geom=np.asarray(data["geom"]).astype(str),
            tangent=np.asarray(data["tangent"], dtype=np.float32),
            normal=np.asarray(data["normal"], dtype=np.float32),
            t_grid=np.asarray(data["t_grid"], dtype=np.float32),
            n_grid=np.asarray(data["n_grid"], dtype=np.float32),
            target_scale_exponents=np.asarray(data["target_scale_exponents"], dtype=np.float32),
        )
    n = len(result.targets)
    if result.patches.ndim != 4 or result.patches.shape[0] != n or result.patches.shape[1] != 6:
        raise RuntimeError(f"Invalid spatial patch shape {result.patches.shape}.")
    if result.context.shape[0] != n or result.targets.shape != (n, 3):
        raise RuntimeError("Dataset arrays are not surface-aligned.")
    if len(set(result.case_id)) != 27:
        raise RuntimeError(f"Expected 27 Phase-1 cases; found {len(set(result.case_id))}.")
    return result


def pairout_groups(data: SpatialData) -> np.ndarray:
    return np.asarray([f"Ma{float(ma):g}_Kn{float(kn):g}" for ma, kn in zip(data.Ma, data.Kn)])


def outer_groups(data: SpatialData, scheme: str) -> np.ndarray:
    if scheme == "loco":
        return data.case_id.astype(str)
    if scheme == "pairout":
        return pairout_groups(data)
    raise ValueError(f"Unknown CV scheme {scheme!r}.")


def choose_inner_group(groups: np.ndarray, outer_group: str, seed: int) -> str:
    unique = sorted(set(map(str, groups)))
    if len(unique) < 2:
        raise RuntimeError("Nested grouped validation needs at least two training groups.")
    rng = np.random.default_rng(stable_seed("inner", outer_group, seed))
    return unique[int(rng.integers(0, len(unique)))]


def _select_k(mask: np.ndarray, score: np.ndarray, k: int, largest: bool = False) -> np.ndarray:
    location = np.flatnonzero(mask.ravel())
    if not len(location):
        return np.zeros_like(mask, dtype=bool)
    k = min(k, len(location))
    order = np.argsort(score.ravel()[location])
    if largest:
        order = order[::-1]
    chosen = location[order[:k]]
    result = np.zeros(mask.size, dtype=bool)
    result[chosen] = True
    return result.reshape(mask.shape)


def make_spatial_masks(
    t_grid: np.ndarray,
    n_grid: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    near_radius_hs: float = 0.75,
) -> Dict[str, np.ndarray]:
    """Create matched-count masks, including direction in the global flow frame."""
    tt, nn_grid = np.meshgrid(t_grid, n_grid)
    radius = np.sqrt(tt ** 2 + nn_grid ** 2)
    full_2d = np.ones_like(radius, dtype=bool)
    near_2d = radius <= near_radius_hs
    k = int(near_2d.sum())
    if k < 3:
        raise ValueError("Near-radius grid contains fewer than three points.")
    far_2d = _select_k(full_2d & (radius > near_radius_hs), radius, k, largest=True)
    n_samples = len(tangent)
    result = {
        "P0": np.zeros((n_samples, *radius.shape), dtype=bool),
        "P_near": np.repeat(near_2d[None, :, :], n_samples, axis=0),
        "P_far": np.repeat(far_2d[None, :, :], n_samples, axis=0),
        "P_full": np.repeat(full_2d[None, :, :], n_samples, axis=0),
    }
    upstream = np.zeros((n_samples, *radius.shape), dtype=bool)
    downstream = np.zeros_like(upstream)
    for i in range(n_samples):
        global_dx = tt * tangent[i, 0] + nn_grid * normal[i, 0]
        upstream[i] = _select_k(global_dx < 0, radius, k)
        downstream[i] = _select_k(global_dx > 0, radius, k)
    result["P_upstream"] = upstream
    result["P_downstream"] = downstream
    for name, value in result.items():
        if value.shape != (n_samples, len(n_grid), len(t_grid)):
            raise AssertionError(f"Mask {name} has shape {value.shape}.")
    return result


@dataclass
class Preprocessor:
    context_mean: np.ndarray
    context_std: np.ndarray
    patch_mean: np.ndarray
    patch_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    target_exponents: np.ndarray

    @classmethod
    def fit(cls, data: SpatialData) -> "Preprocessor":
        context_mean = data.context.mean(axis=0)
        context_std = data.context.std(axis=0)
        context_std[context_std < 1e-6] = 1.0
        patch_mean = data.patches.mean(axis=(0, 2, 3))
        patch_std = data.patches.std(axis=(0, 2, 3))
        patch_std[patch_std < 1e-6] = 1.0
        target_scaled = scale_targets(data.targets, data.Ma, data.target_scale_exponents)
        target_mean = target_scaled.mean(axis=0)
        target_std = target_scaled.std(axis=0)
        target_std[target_std < 1e-8] = 1.0
        return cls(
            context_mean.astype(np.float32), context_std.astype(np.float32),
            patch_mean.astype(np.float32), patch_std.astype(np.float32),
            target_mean.astype(np.float32), target_std.astype(np.float32),
            data.target_scale_exponents.astype(np.float32),
        )

    def arrays(self, data: SpatialData) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        context = ((data.context - self.context_mean) / self.context_std).astype(np.float32)
        patch = ((data.patches - self.patch_mean[None, :, None, None]) /
                 self.patch_std[None, :, None, None]).astype(np.float32)
        target_scaled = scale_targets(data.targets, data.Ma, self.target_exponents)
        targets = ((target_scaled - self.target_mean) / self.target_std).astype(np.float32)
        weights = np.clip(data.weights / max(float(np.mean(data.weights)), 1e-8), 0.25, 4.0)
        return context, patch, targets, weights.astype(np.float32)

    def invert_targets(self, normalized: np.ndarray, ma: np.ndarray) -> np.ndarray:
        scaled = normalized * self.target_std + self.target_mean
        return unscale_targets(scaled, ma, self.target_exponents)


def scale_targets(targets: np.ndarray, ma: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.asarray(ma, dtype=float), 1e-6)[:, None] ** np.asarray(exponents)[None, :]
    return np.asarray(targets, dtype=float) / scale


def unscale_targets(targets: np.ndarray, ma: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.asarray(ma, dtype=float), 1e-6)[:, None] ** np.asarray(exponents)[None, :]
    return np.asarray(targets, dtype=float) * scale


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int, depth: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        width = input_dim
        for _ in range(depth):
            layers.extend((nn.Linear(width, hidden), nn.SiLU(), nn.Dropout(dropout)))
            width = hidden
        layers.append(nn.Linear(width, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialOperator(nn.Module):
    """A sub-100k parameter context + spatial-increment model."""
    def __init__(
        self,
        context_dim: int,
        patch_channels: int,
        latent: int = 48,
        hidden: int = 64,
        depth: int = 2,
        dropout: float = 0.03,
        residual_scale: float = 0.25,
    ):
        super().__init__()
        self.context = MLP(context_dim, latent, hidden, depth, dropout)
        self.base_head = MLP(latent, len(TARGETS), hidden, 1, dropout)
        self.spatial = nn.Sequential(
            nn.Conv2d(patch_channels + 3, 24, 3, padding=1), nn.SiLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 40, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.bulk_head = MLP(latent + 40 + 1, len(TARGETS), hidden, 2, dropout)
        self.residual_scale = float(residual_scale)

    def forward(
        self,
        context: torch.Tensor,
        patch: torch.Tensor,
        mask: torch.Tensor,
        t_coord: torch.Tensor,
        n_coord: torch.Tensor,
    ) -> torch.Tensor:
        mask_f = mask[:, None].to(dtype=patch.dtype)
        context_z = self.context(context)
        base = self.base_head(context_z)
        coverage = mask_f.mean(dim=(2, 3))
        spatial_input = torch.cat((patch * mask_f, t_coord * mask_f, n_coord * mask_f, mask_f), dim=1)
        spatial_z = self.spatial(spatial_input).flatten(1)
        correction = self.bulk_head(torch.cat((context_z, spatial_z, coverage), dim=1))
        present = (mask_f.sum(dim=(2, 3)) > 0).to(dtype=patch.dtype)
        return base + present * self.residual_scale * correction


def coordinate_tensors(data: SpatialData, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    tt, nn_grid = np.meshgrid(data.t_grid, data.n_grid)
    t = torch.from_numpy(tt.astype(np.float32))[None, None].to(device)
    n = torch.from_numpy(nn_grid.astype(np.float32))[None, None].to(device)
    return t, n


def weighted_huber(pred: torch.Tensor, true: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.smooth_l1_loss(pred, true, reduction="none", beta=0.5).mean(dim=1)
    return torch.sum(loss * weights) / torch.clamp(torch.sum(weights), min=1e-8)


def train_model(
    train_data: SpatialData,
    val_data: SpatialData,
    prep: Preprocessor,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[SpatialOperator, Dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    c_train, p_train, y_train, w_train = prep.arrays(train_data)
    c_val, p_val, y_val, w_val = prep.arrays(val_data)
    masks_train = make_spatial_masks(
        train_data.t_grid, train_data.n_grid, train_data.tangent, train_data.normal, args.near_radius_hs
    )
    masks_val = make_spatial_masks(
        val_data.t_grid, val_data.n_grid, val_data.tangent, val_data.normal, args.near_radius_hs
    )
    model = SpatialOperator(
        c_train.shape[1], p_train.shape[1], args.latent, args.hidden,
        args.depth, args.dropout, args.residual_scale,
    ).to(args.device_obj)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > args.max_parameters:
        raise RuntimeError(f"Spatial model has {parameter_count:,} parameters; cap is {args.max_parameters:,}.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.arange(len(c_train), dtype=torch.long)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(c_train), torch.from_numpy(p_train), torch.from_numpy(y_train),
            torch.from_numpy(w_train), indices,
        ),
        batch_size=min(args.batch_size, len(c_train)), shuffle=True, generator=generator,
    )
    cv = torch.from_numpy(c_val).to(args.device_obj)
    pv = torch.from_numpy(p_val).to(args.device_obj)
    yv = torch.from_numpy(y_val).to(args.device_obj)
    wv = torch.from_numpy(w_val).to(args.device_obj)
    t_coord, n_coord = coordinate_tensors(train_data, args.device_obj)
    best_state: Optional[Mapping[str, torch.Tensor]] = None
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    schedule = list(REAL_CONFIGS)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_i, (context_b, patch_b, y_b, w_b, index_b) in enumerate(loader):
            context_b = context_b.to(args.device_obj, non_blocking=True)
            patch_b = patch_b.to(args.device_obj, non_blocking=True)
            y_b = y_b.to(args.device_obj, non_blocking=True)
            w_b = w_b.to(args.device_obj, non_blocking=True)
            chosen = schedule[(epoch + batch_i) % len(schedule)]
            chosen_mask = torch.from_numpy(masks_train[chosen][index_b.numpy()]).to(args.device_obj)
            anchor = "P0" if (epoch + batch_i) % 2 == 0 else "P_full"
            anchor_mask = torch.from_numpy(masks_train[anchor][index_b.numpy()]).to(args.device_obj)
            optimizer.zero_grad(set_to_none=True)
            loss = 0.5 * (
                weighted_huber(model(context_b, patch_b, chosen_mask, t_coord, n_coord), y_b, w_b)
                + weighted_huber(model(context_b, patch_b, anchor_mask, t_coord, n_coord), y_b, w_b)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_losses = []
            for name in ("P0", "P_near", "P_full"):
                mask = torch.from_numpy(masks_val[name]).to(args.device_obj)
                validation_losses.append(weighted_huber(model(cv, pv, mask, t_coord, n_coord), yv, wv))
            value = float(torch.stack(validation_losses).mean().cpu())
        if np.isfinite(value) and value < best_loss - 1e-6:
            best_loss, best_epoch = value, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Spatial training produced no finite validation checkpoint.")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "parameter_count": parameter_count, "training_seed": seed,
    }


def shifted_patch_tensor(data: SpatialData, patches: np.ndarray, fraction: float = 0.25) -> np.ndarray:
    result = patches.copy()
    for case_id in sorted(set(data.case_id)):
        loc = np.flatnonzero(data.case_id == case_id)
        loc = loc[np.argsort(data.s01[loc])]
        shift = max(1, int(round(fraction * len(loc)))) % len(loc)
        if shift == 0:
            shift = 1
        result[loc] = np.roll(patches[loc], shift, axis=0)
    return result


def predict_configs(
    model: SpatialOperator,
    data: SpatialData,
    prep: Preprocessor,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    context, patches, _, _ = prep.arrays(data)
    masks = make_spatial_masks(data.t_grid, data.n_grid, data.tangent, data.normal, args.near_radius_hs)
    patch_variants = {
        "P_shift": shifted_patch_tensor(data, patches, args.surface_shift_fraction),
        "P_radial_flip": patches[:, :, ::-1, :].copy(),
    }
    result = {}
    t_coord, n_coord = coordinate_tensors(data, args.device_obj)
    model.eval()
    step = max(args.batch_size, 1024)
    for name in ALL_CONFIGS:
        source = patch_variants.get(name, patches)
        mask_name = "P_full" if name in CONTROL_CONFIGS else name
        pieces = []
        with torch.no_grad():
            for start in range(0, len(context), step):
                stop = min(len(context), start + step)
                c = torch.from_numpy(context[start:stop]).to(args.device_obj)
                p = torch.from_numpy(source[start:stop]).to(args.device_obj)
                m = torch.from_numpy(masks[mask_name][start:stop]).to(args.device_obj)
                pieces.append(model(c, p, m, t_coord, n_coord).cpu().numpy())
        normalized = np.concatenate(pieces, axis=0)
        result[name] = prep.invert_targets(normalized, data.Ma).astype(np.float32)
    return result


def prediction_frame(
    data: SpatialData,
    predictions: Mapping[str, np.ndarray],
    scheme: str,
    outer_group: str,
    inner_group: str,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for config, values in predictions.items():
        for target_i, target in enumerate(TARGETS):
            rows.append(pd.DataFrame({
                "scheme": scheme, "outer_group": outer_group, "inner_group": inner_group,
                "seed": seed, "target": target, "config": config,
                "case_id": data.case_id, "surface_i": data.surface_i, "s01": data.s01,
                "Ma": data.Ma, "Kn": data.Kn, "geom": data.geom,
                "true": data.targets[:, target_i], "pred": values[:, target_i],
            }))
    return pd.concat(rows, ignore_index=True)


def case_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["scheme", "outer_group", "seed", "target", "config", "case_id"]
    optional = [name for name in ("inner_group",) if name in predictions]
    for key, group in predictions.groupby([*keys, *optional], sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        item = dict(zip([*keys, *optional], key))
        true = group["true"].to_numpy(float)
        pred = group["pred"].to_numpy(float)
        data_range = max(float(np.ptp(true)), 1e-12)
        item.update({
            "relL2": rel_l2(pred, true),
            "rangeMAE_pct": 100.0 * float(np.mean(np.abs(pred - true))) / data_range,
            "n_surface": len(group),
        })
        rows.append(item)
    return pd.DataFrame(rows)


def run_one_task(data: SpatialData, args: argparse.Namespace) -> None:
    if not args.only_scheme or not args.only_outer_group or len(args.seeds_list) != 1:
        raise ValueError("Worker mode requires one scheme, one outer group, and one seed.")
    out_dir = Path(args.out).expanduser().resolve()
    marker = out_dir / "TASK_COMPLETE"
    if args.resume and marker.exists():
        print(f"[SKIP] completed task {out_dir}", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    scheme, outer_group, seed = args.only_scheme, args.only_outer_group, args.seeds_list[0]
    groups = outer_groups(data, scheme)
    if outer_group not in set(groups):
        raise ValueError(f"Unknown {scheme} group {outer_group!r}.")
    test_index = np.flatnonzero(groups == outer_group)
    outer_train_index = np.flatnonzero(groups != outer_group)
    train_groups = groups[outer_train_index]
    inner_group = choose_inner_group(train_groups, outer_group, stable_seed("stage3_inner", scheme, outer_group))
    val_local = train_groups == inner_group
    train_index = outer_train_index[~val_local]
    val_index = outer_train_index[val_local]
    train_data, val_data, test_data = data.take(train_index), data.take(val_index), data.take(test_index)
    prep = Preprocessor.fit(train_data)
    model, diagnostics = train_model(
        train_data, val_data, prep, args,
        stable_seed("stage3_model", scheme, outer_group, seed),
    )
    val_predictions = predict_configs(model, val_data, prep, args)
    test_predictions = predict_configs(model, test_data, prep, args)
    val_frame = prediction_frame(val_data, val_predictions, scheme, outer_group, inner_group, seed)
    test_frame = prediction_frame(test_data, test_predictions, scheme, outer_group, inner_group, seed)
    val_frame.to_csv(out_dir / "validation_predictions.csv", index=False)
    test_frame.to_csv(out_dir / "surface_predictions.csv", index=False)
    case_metrics(test_frame).to_csv(out_dir / "case_metrics.csv", index=False)
    diagnostics.update({
        "scheme": scheme, "outer_group": outer_group, "inner_group": inner_group, "seed": seed,
        "n_train": len(train_data.targets), "n_validation": len(val_data.targets),
        "n_test": len(test_data.targets), "target_scale_exponents": prep.target_exponents.tolist(),
    })
    (out_dir / "training_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    marker.write_text(json.dumps(diagnostics, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[DONE] Stage3 {scheme}|{outer_group}|{seed}; epoch={diagnostics['best_epoch']}", flush=True)


def concat_task_files(task_root: Path, filename: str) -> pd.DataFrame:
    paths = sorted(task_root.glob(f"task_*/{filename}"))
    if not paths:
        raise RuntimeError(f"No {filename} files found under {task_root}.")
    return pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)


def ensemble_predictions(points: pd.DataFrame, expected_seeds: int) -> pd.DataFrame:
    keys = ["scheme", "outer_group", "target", "config", "case_id", "surface_i"]
    seed_counts = points.groupby(keys)["seed"].nunique()
    if int(seed_counts.min()) != expected_seeds or int(seed_counts.max()) != expected_seeds:
        raise RuntimeError("Incomplete seed ensemble for at least one surface prediction.")
    metadata = ["inner_group", "s01", "Ma", "Kn", "geom", "true"]
    aggregation = {"pred": "mean", **{name: "first" for name in metadata if name in points}}
    return points.groupby(keys, as_index=False).agg(aggregation)


def summarize(case_table: pd.DataFrame, args: argparse.Namespace, out_dir: Path):
    table = case_table.copy()
    table["unit_id"] = np.where(table["scheme"] == "loco", table["case_id"], table["outer_group"])
    units = table.groupby(["scheme", "target", "config", "unit_id"], as_index=False).agg(
        relL2=("relL2", "mean"), rangeMAE_pct=("rangeMAE_pct", "mean")
    )
    rows = []
    for (scheme, target, config), group in units.groupby(["scheme", "target", "config"]):
        mean, lo, hi = bootstrap_mean(
            group["relL2"].to_numpy(float), args.bootstrap,
            stable_seed("stage3_summary", scheme, target, config),
        )
        rows.append({
            "scheme": scheme, "target": target, "config": config,
            "n_bootstrap_units": group["unit_id"].nunique(), "mean_relL2": mean,
            "median_relL2": float(group["relL2"].median()),
            "bootstrap_CI2p5": lo, "bootstrap_CI97p5": hi,
            "mean_rangeMAE_pct": float(group["rangeMAE_pct"].mean()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "stage3_ensemble_summary.csv", index=False)
    comparisons = (
        ("P0", "P_full"), ("P_far", "P_near"), ("P_shift", "P_full"),
        ("P_downstream", "P_upstream"), ("P_radial_flip", "P_full"),
    )
    gains = []
    for (scheme, target), group in units.groupby(["scheme", "target"]):
        pivot = group.pivot(index="unit_id", columns="config", values="relL2")
        for baseline, candidate in comparisons:
            difference = (pivot[baseline] - pivot[candidate]).dropna().to_numpy(float)
            mean, lo, hi = bootstrap_mean(
                difference, args.bootstrap,
                stable_seed("stage3_gain", scheme, target, baseline, candidate),
            )
            gains.append({
                "scheme": scheme, "target": target,
                "comparison": f"{baseline}-{candidate}", "n_bootstrap_units": len(difference),
                "mean_gain": mean, "mean_gain_pp": 100 * mean,
                "bootstrap_CI2p5": lo, "bootstrap_CI97p5": hi,
                "CI2p5_pp": 100 * lo, "CI97p5_pp": 100 * hi,
            })
    gain_table = pd.DataFrame(gains)
    gain_table.to_csv(out_dir / "stage3_paired_gains.csv", index=False)
    return summary, gain_table


def _lookup(table: pd.DataFrame, scheme: str, target: str, column: str, value: str) -> pd.Series:
    match = table[(table["scheme"] == scheme) & (table["target"] == target) & (table[column] == value)]
    if len(match) != 1:
        raise RuntimeError(f"Expected one row for {scheme}/{target}/{value}; found {len(match)}.")
    return match.iloc[0]


def make_decision(summary: pd.DataFrame, gains: pd.DataFrame, args: argparse.Namespace, out_dir: Path):
    checks = []
    for target in TARGETS:
        loco = _lookup(summary, "loco", target, "config", "P_full")
        pair = _lookup(summary, "pairout", target, "config", "P_full")
        loco_gain = _lookup(gains, "loco", target, "comparison", "P0-P_full")
        pair_gain = _lookup(gains, "pairout", target, "comparison", "P0-P_full")
        alignment = _lookup(gains, "loco", target, "comparison", "P_shift-P_full")
        locality = _lookup(gains, "loco", target, "comparison", "P_far-P_near")
        checks.append({
            "target": target,
            "loco_full_mean_relL2": float(loco["mean_relL2"]),
            "loco_full_CI97p5": float(loco["bootstrap_CI97p5"]),
            "pairout_full_mean_relL2": float(pair["mean_relL2"]),
            "accuracy_pass": bool(
                float(loco["bootstrap_CI97p5"]) <= args.loco_tolerance_map[target]
                and float(pair["mean_relL2"]) <= args.pairout_tolerance_map[target]
            ),
            "context_to_spatial_gain_loco_pp": float(loco_gain["mean_gain_pp"]),
            "context_to_spatial_gain_pairout_pp": float(pair_gain["mean_gain_pp"]),
            "spatial_increment_pass": bool(
                float(loco_gain["mean_gain"]) >= args.min_gain
                and float(loco_gain["bootstrap_CI2p5"]) > 0
                and float(pair_gain["mean_gain"]) > 0
            ),
            "aligned_beats_shifted_pass": bool(float(alignment["bootstrap_CI2p5"]) > 0),
            "near_beats_far_pass": bool(float(locality["bootstrap_CI2p5"]) > 0),
        })
    primary = [item for item in checks if item["target"] in ("Cp", "Cq")]
    signal = all(item["accuracy_pass"] and item["spatial_increment_pass"] and item["aligned_beats_shifted_pass"] for item in primary)
    local = signal and all(item["near_beats_far_pass"] for item in primary)
    if local:
        verdict = "SPATIAL_BULK_SIGNAL_WITH_LOCAL_SUPPORT"
        action = "Proceed with the quantity-dependent predictive-support manuscript using spatial patches."
    elif signal:
        verdict = "SPATIAL_BULK_SIGNAL_WITHOUT_LOCAL_SUPPORT"
        action = "Pivot the paper to spatial bulk-state augmentation; do not claim a local information radius."
    else:
        verdict = "NO_ACTIONABLE_SPATIAL_BULK_SIGNAL"
        action = "Do not submit the current observability claim; use the negative result to redesign the paper."
    decision = {
        "verdict": verdict, "action": action, "checks": checks,
        "predeclared_primary_targets": ["Cp", "Cq"],
        "new_dsmc_runs": 0, "higher_order_moments": False,
        "interpretation_limit": "predictive spatial support, not causal information horizon",
    }
    (out_dir / "stage3_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    lines = [f"VERDICT: {verdict}", f"ACTION: {action}", ""]
    for item in checks:
        lines.append(
            f"{item['target']}: accuracy={item['accuracy_pass']}, spatial_increment="
            f"{item['spatial_increment_pass']}, alignment={item['aligned_beats_shifted_pass']}, "
            f"near_vs_far={item['near_beats_far_pass']}"
        )
    (out_dir / "stage3_decision.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False)
    order = list(ALL_CONFIGS)
    for row_i, scheme in enumerate(("loco", "pairout")):
        for col_i, target in enumerate(TARGETS):
            ax = axes[row_i, col_i]
            part = summary[(summary["scheme"] == scheme) & (summary["target"] == target)].set_index("config").reindex(order)
            x = np.arange(len(part))
            mean = part["mean_relL2"].to_numpy(float) * 100
            lo = part["bootstrap_CI2p5"].to_numpy(float) * 100
            hi = part["bootstrap_CI97p5"].to_numpy(float) * 100
            ax.errorbar(x, mean, yerr=np.vstack((mean - lo, hi - mean)), marker="o", capsize=3)
            ax.set_xticks(x, order, rotation=45, ha="right")
            ax.set_title(f"{scheme}: {target}")
            ax.set_ylabel("ensemble relative L2 error (%)")
            ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "stage3_error_summary.pdf")
    fig.savefig(out_dir / "stage3_error_summary.png", dpi=180)
    plt.close(fig)


def aggregate_results(data: SpatialData, args: argparse.Namespace):
    out_dir = Path(args.out).expanduser().resolve()
    task_root = Path(args.aggregate_task_root).expanduser().resolve()
    expected = (len(set(data.case_id)) + len(set(pairout_groups(data)))) * len(args.seeds_list)
    completed = len(list(task_root.glob("task_*/TASK_COMPLETE")))
    if completed != expected:
        raise RuntimeError(f"Expected {expected} Stage-3 tasks; found {completed} complete markers.")
    individual_points = concat_task_files(task_root, "surface_predictions.csv")
    validation = concat_task_files(task_root, "validation_predictions.csv")
    individual_cases = concat_task_files(task_root, "case_metrics.csv")
    duplicate = ["scheme", "outer_group", "seed", "target", "config", "case_id"]
    if individual_cases.duplicated(duplicate).any():
        raise RuntimeError("Duplicate Stage-3 case metrics detected.")
    individual_points.to_csv(out_dir / "stage3_individual_seed_surface_predictions.csv", index=False)
    validation.to_csv(out_dir / "stage3_individual_seed_validation_predictions.csv", index=False)
    individual_cases.to_csv(out_dir / "stage3_individual_seed_case_metrics.csv", index=False)
    ensemble_points = ensemble_predictions(individual_points, len(args.seeds_list))
    ensemble_cases = case_metrics(ensemble_points.assign(seed=-1))
    ensemble_points.to_csv(out_dir / "stage3_ensemble_surface_predictions.csv", index=False)
    ensemble_cases.to_csv(out_dir / "stage3_ensemble_case_metrics.csv", index=False)
    summary, gains = summarize(ensemble_cases, args, out_dir)
    decision = make_decision(summary, gains, args, out_dir)
    plot_summary(summary, out_dir)
    return decision


def parse_target_map(text: str) -> Dict[str, float]:
    result = {}
    for token in text.split(","):
        name, value = token.split("=", 1)
        result[name.strip()] = float(value)
    if set(result) != set(TARGETS):
        raise ValueError(f"Target map must define exactly {TARGETS}.")
    return result


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def write_run_record(dataset_path: Path, data: SpatialData, args: argparse.Namespace) -> None:
    out_dir = Path(args.out).expanduser().resolve()
    payload = {
        "dataset_sha256": sha256_file(dataset_path), "code_sha256": sha256_file(Path(__file__).resolve()),
        "cv": args.cv_list, "seeds": args.seeds_list, "only_scheme": args.only_scheme,
        "only_outer_group": args.only_outer_group, "epochs": args.epochs, "patience": args.patience,
        "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
        "hidden": args.hidden, "latent": args.latent, "depth": args.depth,
        "dropout": args.dropout, "residual_scale": args.residual_scale,
        "near_radius_hs": args.near_radius_hs, "surface_shift_fraction": args.surface_shift_fraction,
        "target_scale_exponents": data.target_scale_exponents.tolist(),
        "max_parameters": args.max_parameters, "bootstrap": args.bootstrap,
        "loco_tolerances": args.loco_tolerance_map, "pairout_tolerances": args.pairout_tolerance_map,
        "min_gain": args.min_gain,
    }
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    path = out_dir / "stage3_run_config.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("signature") != signature:
            raise RuntimeError("Output directory contains a different Stage-3 run signature.")
    record = {
        "signature": signature, "payload": payload, "dataset": str(dataset_path),
        "n_cases": len(set(data.case_id)), "n_surface_samples": len(data.targets),
        "patch_shape": list(data.patches.shape[1:]), "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "hostname": os.uname().nodename,
        "started_unix": time.time(), "new_dsmc_runs": 0, "higher_order_moments": False,
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
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
    parser.add_argument("--surface-shift-fraction", type=float, default=0.25)
    parser.add_argument("--max-parameters", type=int, default=100000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--loco-tolerances", default="Cp=0.10,Cq=0.10,tau_abs=0.20")
    parser.add_argument("--pairout-tolerances", default="Cp=0.15,Cq=0.15,tau_abs=0.25")
    parser.add_argument("--min-gain-pp", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.cv_list = [value.strip() for value in args.cv.split(",") if value.strip()]
    args.seeds_list = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    args.device_obj = resolve_device(args.device)
    args.loco_tolerance_map = parse_target_map(args.loco_tolerances)
    args.pairout_tolerance_map = parse_target_map(args.pairout_tolerances)
    args.min_gain = args.min_gain_pp / 100.0
    if not 0 < args.near_radius_hs <= 3 or not 0 < args.surface_shift_fraction < 1:
        raise ValueError("Invalid spatial radius or shift fraction.")
    if args.only_scheme and len(args.seeds_list) != 1:
        raise ValueError("A Stage-3 worker receives exactly one seed.")
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset).expanduser().resolve()
    data = load_dataset(dataset_path)
    write_run_record(dataset_path, data, args)
    if args.aggregate_task_root:
        decision = aggregate_results(data, args)
        print(f"[DONE] Stage-3 verdict={decision['verdict']}", flush=True)
    else:
        run_one_task(data, args)


if __name__ == "__main__":
    main()
