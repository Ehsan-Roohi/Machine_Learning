#!/usr/bin/env python3
"""Plot DSMC, transferable S1, and incident half-range S3 wall Cp profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def nrmse(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = max(float(np.sqrt(np.mean(target**2))), 1.0e-15)
    return float(np.sqrt(np.mean((prediction - target) ** 2)) / denominator)


def s1_cp_bins(predictions: np.lib.npyio.NpzFile, case_id: str, bins: list[dict]) -> np.ndarray:
    mask = (predictions["case_id"] == case_id) & predictions["protrusion"]
    ids = predictions["surface_id"][mask]
    values = predictions["S1_cp"][mask]
    result = []
    for entry in bins:
        selected = (ids >= int(entry["start_id"])) & (ids <= int(entry["stop_id"]))
        if not np.any(selected):
            raise ValueError(f"{case_id}: no S1 values for IDs {entry['start_id']}-{entry['stop_id']}")
        result.append(float(np.mean(values[selected])))
    return np.asarray(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_json", type=Path)
    parser.add_argument("loco_predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plotted = {"description": "Cp values plotted in cp_profile_comparison", "cases": []}

    plt.rcParams.update({"font.size": 10.5, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.4))
    labels = ["W1", "W2", "W3", "L1", "L2", "L3"]
    x = np.arange(len(labels))

    with np.load(args.loco_predictions, allow_pickle=False) as predictions:
        for axis, case in zip(axes, payload["cases"], strict=True):
            bins = case["bins"]
            target = np.asarray([entry["target"][0] for entry in bins])
            target_sem = np.asarray([entry["target_sem"][0] for entry in bins])
            incident = np.asarray([entry["incident"][0] for entry in bins])
            records = np.asarray([entry["records"] for entry in bins], dtype=int)
            s1 = s1_cp_bins(predictions, str(case["case_id"]), bins)

            all_error = nrmse(target, incident)
            windward_error = nrmse(target[:3], incident[:3])
            corr = float(np.corrcoef(target, incident)[0, 1])

            axis.axvspan(2.5, 5.5, color="#d62728", alpha=0.07, label="Sparse: <50 records/bin")
            axis.errorbar(
                x,
                target,
                yerr=target_sem,
                color="#111111",
                marker="o",
                linewidth=2.2,
                capsize=3,
                label="DSMC wall tally",
                zorder=4,
            )
            axis.plot(x, incident, color="#0072B2", marker="s", linewidth=2.0, label="S3 incident half-range")
            axis.plot(x, s1, color="#D55E00", marker="^", linewidth=1.8, linestyle="--", label="S1 transferable LOCO")
            axis.axvline(2.5, color="#666666", linewidth=0.9, linestyle=":")
            axis.axhline(0.0, color="#888888", linewidth=0.8)
            axis.set_xticks(x, labels)
            axis.set_xlabel("Wall bins: W = windward, L = leeward")
            axis.set_ylabel(r"$C_p$")
            axis.grid(axis="y", alpha=0.25)
            axis.set_title(
                f"{case['case_id'].replace('ISO_Ma6_', '').replace('Kn0p', 'Kn = 0.')}\n"
                f"S3 NRMSE = {100*all_error:.1f}%,  r = {corr:.3f}"
            )
            top = max(float(np.max(target + target_sem)), float(np.max(incident)), float(np.max(s1)))
            bottom = min(float(np.min(target - target_sem)), float(np.min(incident)), float(np.min(s1)))
            margin = max(0.08 * (top - bottom), 0.03)
            axis.set_ylim(bottom - margin, top + 2.2 * margin)
            annotation_y = top + 0.55 * margin
            for index, count in enumerate(records):
                axis.text(index, annotation_y, f"n={count}", ha="center", va="bottom", fontsize=8)

            plotted["cases"].append(
                {
                    "case_id": case["case_id"],
                    "labels": labels,
                    "surface_id_ranges": [f"{entry['start_id']}-{entry['stop_id']}" for entry in bins],
                    "records": records.tolist(),
                    "dsmc_cp": target.tolist(),
                    "dsmc_sem": target_sem.tolist(),
                    "s3_incident_cp": incident.tolist(),
                    "s1_loco_cp": s1.tolist(),
                    "s3_all_bins_nrmse": all_error,
                    "s3_windward_nrmse": windward_error,
                    "s3_correlation": corr,
                }
            )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.14, top=0.76, wspace=0.15)
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Wall-pressure coefficient: DSMC target versus bulk and half-range recovery",
        y=0.98,
        fontsize=14,
    )
    png = args.output_dir / "cp_profile_comparison.png"
    svg = args.output_dir / "cp_profile_comparison.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    (args.output_dir / "cp_profile_values.json").write_text(
        json.dumps(plotted, indent=2) + "\n", encoding="utf-8"
    )
    print(f"WROTE {png}")
    print(f"WROTE {svg}")
    for case in plotted["cases"]:
        print(
            f"{case['case_id']}: all={100*case['s3_all_bins_nrmse']:.2f}% "
            f"windward={100*case['s3_windward_nrmse']:.2f}% r={case['s3_correlation']:.4f}"
        )


if __name__ == "__main__":
    main()
