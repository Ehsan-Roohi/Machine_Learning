#!/usr/bin/env python3
"""Map DSMC and incident half-range Cp directly onto the physical protrusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm


def surface_geometry(case_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(case_dir / "output" / "moment_blocks.npz", allow_pickle=False) as data:
        wall = data["wall"].mean(axis=0)
        columns = {str(name): index for index, name in enumerate(data["wall_columns"])}
    ids = wall[:, columns["id"]].astype(int)
    mask = (ids >= 981) & (ids <= 1040)
    order = np.argsort(ids[mask])
    selected = wall[mask][order]
    selected_ids = ids[mask][order]
    segments = np.stack(
        (
            selected[:, [columns["v1x"], columns["v1y"]]],
            selected[:, [columns["v2x"], columns["v2y"]]],
        ),
        axis=1,
    )
    lengths = selected[:, columns["length"]]
    return selected_ids, segments, lengths


def expand_bin_values(ids: np.ndarray, bins: list[dict], source: str) -> np.ndarray:
    values = np.full(ids.shape, np.nan, dtype=float)
    for entry in bins:
        selected = (ids >= int(entry["start_id"])) & (ids <= int(entry["stop_id"]))
        values[selected] = float(entry[source][0])
    if np.any(~np.isfinite(values)):
        raise ValueError("not every protrusion segment received a Cp value")
    return values


def bin_centers(lengths: np.ndarray, bins: list[dict]) -> np.ndarray:
    edges = np.concatenate(([0.0], np.cumsum(lengths)))
    total = edges[-1]
    result = []
    for entry in bins:
        lo = int(entry["start_id"]) - 981
        hi = int(entry["stop_id"]) - 981 + 1
        result.append(0.5 * (edges[lo] + edges[hi]) / total)
    return np.asarray(result)


def add_surface_map(
    axis: plt.Axes,
    segments_m: np.ndarray,
    values: np.ndarray,
    norm: TwoSlopeNorm,
    title: str,
) -> LineCollection:
    segments_mm = 1.0e3 * segments_m
    collection = LineCollection(
        segments_mm,
        cmap="coolwarm",
        norm=norm,
        linewidths=10.0,
        capstyle="butt",
    )
    collection.set_array(values)
    axis.add_collection(collection)

    xmin = float(np.min(segments_mm[:, :, 0]))
    xmax = float(np.max(segments_mm[:, :, 0]))
    ymax = float(np.max(segments_mm[:, :, 1]))
    pad_x = 0.15 * (xmax - xmin)
    axis.plot([xmin - pad_x, xmin], [0.0, 0.0], color="0.25", linewidth=2.0)
    axis.plot([xmax, xmax + pad_x], [0.0, 0.0], color="0.25", linewidth=2.0)
    axis.annotate(
        "flow",
        xy=(xmin + 0.20 * (xmax - xmin), 0.78 * ymax),
        xytext=(xmin - 0.10 * (xmax - xmin), 0.78 * ymax),
        arrowprops={"arrowstyle": "-|>", "linewidth": 1.5, "color": "0.15"},
        ha="center",
        va="center",
        fontsize=9,
    )
    axis.set_xlim(xmin - pad_x, xmax + pad_x)
    axis.set_ylim(-0.06 * ymax, 1.08 * ymax)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"$x$ (mm)")
    axis.set_ylabel(r"$y$ (mm)")
    axis.set_title(title, fontsize=11, weight="bold")
    axis.grid(alpha=0.18)
    return collection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_json", type=Path)
    parser.add_argument("production_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    norm = TwoSlopeNorm(vmin=-0.06, vcenter=0.0, vmax=1.08)

    fig = plt.figure(figsize=(13.0, 7.5))
    grid = fig.add_gridspec(2, 3, width_ratios=(1.05, 1.05, 1.35), hspace=0.35, wspace=0.28)
    geometry_axes: list[plt.Axes] = []
    curve_axes: list[plt.Axes] = []
    mappable: LineCollection | None = None

    plotted = {"description": "Physical-coordinate Cp map and arclength profiles", "cases": []}
    for row, case in enumerate(payload["cases"]):
        case_id = str(case["case_id"])
        ids, segments, lengths = surface_geometry(args.production_root / case_id)
        bins = case["bins"]
        target_segment = expand_bin_values(ids, bins, "target")
        incident_segment = expand_bin_values(ids, bins, "incident")
        target_bins = np.asarray([entry["target"][0] for entry in bins])
        incident_bins = np.asarray([entry["incident"][0] for entry in bins])
        target_sem = np.asarray([entry["target_sem"][0] for entry in bins])
        centers = bin_centers(lengths, bins)
        records = np.asarray([entry["records"] for entry in bins], dtype=int)

        ax_target = fig.add_subplot(grid[row, 0])
        ax_s3 = fig.add_subplot(grid[row, 1])
        ax_curve = fig.add_subplot(grid[row, 2])
        geometry_axes.extend((ax_target, ax_s3))
        curve_axes.append(ax_curve)

        mappable = add_surface_map(ax_target, segments, target_segment, norm, "DSMC wall tally")
        add_surface_map(ax_s3, segments, incident_segment, norm, "S3 incident half-range")

        ax_curve.errorbar(
            centers,
            target_bins,
            yerr=target_sem,
            color="black",
            marker="o",
            linewidth=2.2,
            capsize=3,
            label="DSMC",
        )
        ax_curve.plot(
            centers,
            incident_bins,
            color="#0072B2",
            marker="s",
            linewidth=2.0,
            label="S3 incident",
        )
        ax_curve.axvline(0.5, color="0.45", linestyle=":", linewidth=1.2)
        ax_curve.axhline(0.0, color="0.55", linewidth=0.8)
        ax_curve.text(0.25, 0.96, "windward face", transform=ax_curve.transAxes, ha="center", va="top", fontsize=9)
        ax_curve.text(0.75, 0.96, "leeward face", transform=ax_curve.transAxes, ha="center", va="top", fontsize=9)
        ax_curve.set_xlim(0.0, 1.0)
        ax_curve.set_xlabel(r"normalized physical arclength, $s/L_w$")
        ax_curve.set_ylabel(r"$C_p=(p_w-p_\infty)/(\frac{1}{2}\rho_\infty U_\infty^2)$")
        ax_curve.grid(alpha=0.25)
        ax_curve.legend(loc="best", frameon=False)
        cp_metric = case["metrics"]["cp"]
        ax_curve.set_title(
            f"Profile comparison: NRMSE={100*float(cp_metric['incident_nrmse']):.1f}%, "
            f"r={float(cp_metric['incident_correlation']):.3f}",
            fontsize=10.5,
            weight="bold",
        )

        row_label = case_id.replace("ISO_Ma6_Kn0p", r"ISO, $Ma=6$, $Kn=0.") + "$"
        fig.text(0.018, 0.72 - 0.455 * row, row_label, rotation=90, ha="center", va="center", fontsize=12, weight="bold")
        plotted["cases"].append(
            {
                "case_id": case_id,
                "normalized_arclength": centers.tolist(),
                "dsmc_cp": target_bins.tolist(),
                "s3_incident_cp": incident_bins.tolist(),
                "dsmc_sem": target_sem.tolist(),
                "collision_records": records.tolist(),
            }
        )

    if mappable is None:
        raise ValueError("no cases were plotted")
    colorbar_axis = fig.add_axes([0.14, 0.09, 0.43, 0.027])
    colorbar = fig.colorbar(mappable, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label(r"wall-pressure coefficient, $C_p$")
    fig.suptitle(
        "Physical wall-pressure recovery on the triangular protrusion",
        fontsize=15,
        weight="bold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.92, bottom=0.18)

    for suffix in ("png", "svg", "pdf"):
        path = args.output_dir / f"cp_physical_geometry.{suffix}"
        fig.savefig(path, dpi=240 if suffix == "png" else None, bbox_inches="tight")
        print(f"WROTE {path}")
    plt.close(fig)
    (args.output_dir / "cp_physical_geometry_values.json").write_text(
        json.dumps(plotted, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
