#!/usr/bin/env python3
"""Construct positive distributions with equal full moments and unequal wall fluxes.

The construction is deliberately stronger than the S0/S1/S2 feature hierarchy:
both distributions have every two-dimensional monomial moment through total
degree three in common.  A bounded null-space perturbation is selected by a
linear program to maximize either diffuse-wall incident pressure or shear.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog


def moment_matrix(vn: np.ndarray, vt: np.ndarray):
    names, rows = [], []
    for total in range(4):
        for power_n in range(total + 1):
            power_t = total-power_n
            names.append(f"vn^{power_n} vt^{power_t}")
            rows.append(vn**power_n*vt**power_t)
    return names, np.asarray(rows)


def wall_kernel(vn: np.ndarray, vt: np.ndarray, quantity: str) -> np.ndarray:
    incoming = vn < 0.0
    if quantity == "pressure":
        # Unit thermal speed and a fully diffuse unit-temperature wall.
        expected_post_normal = np.sqrt(np.pi/2.0)
        return np.where(incoming, (-vn)*(expected_post_normal-vn), 0.0)
    if quantity == "shear":
        return np.where(incoming, (-vn)*vt, 0.0)
    raise ValueError(quantity)


def construct(vn: np.ndarray, vt: np.ndarray, base: np.ndarray,
              matrix: np.ndarray, quantity: str, beta: float):
    kernel = wall_kernel(vn, vt, quantity)
    result = linprog(-kernel, A_eq=matrix, b_eq=np.zeros(len(matrix)),
                     bounds=list(zip(-beta*base, beta*base)), method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    delta = result.x
    minus, plus = base-delta, base+delta
    return {
        "quantity": quantity,
        "minus": minus,
        "plus": plus,
        "delta": delta,
        "kernel": kernel,
        "flux_minus": float(kernel@minus),
        "flux_plus": float(kernel@plus),
        "max_moment_mismatch": float(np.max(np.abs(matrix@(plus-minus)))),
        "minimum_probability": float(min(minus.min(), plus.min())),
        "total_variation_distance": float(np.sum(np.abs(delta))),
    }


def plot(results, velocity: np.ndarray, vn2: np.ndarray, vt2: np.ndarray,
         base2: np.ndarray, output: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14.4, 7.3))
    levels = np.linspace(0.0, float(base2.max()), 18)
    relative_levels = np.linspace(-0.7, 0.7, 19)
    for row, result in enumerate(results):
        minus = result["minus"].reshape(vn2.shape)
        plus = result["plus"].reshape(vn2.shape)
        relative = (plus-minus)/base2
        for ax, values, title in (
            (axes[row, 0], minus, r"$f_-$"),
            (axes[row, 1], plus, r"$f_+$"),
        ):
            contour = ax.contourf(vn2, vt2, values, levels=levels,
                                  cmap="viridis", extend="max")
            ax.axvline(0, color="white", ls="--", lw=1.0)
            ax.set_title(title, weight="bold")
            ax.set_xlabel(r"normal velocity, $v_n/v_{th}$")
            ax.set_ylabel(r"tangential velocity, $v_t/v_{th}$")
            ax.set_aspect("equal")
        diff = axes[row, 2].contourf(vn2, vt2, relative,
                                     levels=relative_levels, cmap="coolwarm",
                                     extend="both")
        axes[row, 2].axvline(0, color="black", ls="--", lw=1.0)
        axes[row, 2].set_title(r"$(f_+-f_-)/f_M$", weight="bold")
        axes[row, 2].set_xlabel(r"normal velocity, $v_n/v_{th}$")
        axes[row, 2].set_ylabel(r"tangential velocity, $v_t/v_{th}$")
        axes[row, 2].set_aspect("equal")

        ax = axes[row, 3]
        if result["quantity"] == "pressure":
            kernel = result["kernel"].reshape(vn2.shape)
            y_minus = np.sum(kernel*minus, axis=1)
            y_plus = np.sum(kernel*plus, axis=1)
            ax.plot(velocity, y_minus, "o-", ms=3, lw=1.4, color="black", label=r"$f_-$")
            ax.plot(velocity, y_plus, "s-", ms=3, lw=1.4, color="#2f74b5", label=r"$f_+$")
            ax.set_xlabel(r"normal velocity, $v_n/v_{th}$")
            ax.set_ylabel("incident pressure-flux density")
        else:
            incoming_weight = np.where(vn2 < 0.0, -vn2, 0.0)
            y_minus = vt2[0]*np.sum(incoming_weight*minus, axis=0)
            y_plus = vt2[0]*np.sum(incoming_weight*plus, axis=0)
            ax.plot(velocity, y_minus, "o-", ms=3, lw=1.4, color="black", label=r"$f_-$")
            ax.plot(velocity, y_plus, "s-", ms=3, lw=1.4, color="#2f74b5", label=r"$f_+$")
            ax.set_xlabel(r"tangential velocity, $v_t/v_{th}$")
            ax.set_ylabel("incident shear-flux density")
        ax.axhline(0, color="0.55", lw=.8)
        ax.grid(alpha=.22)
        ax.set_title(
            f"wall functional: {result['flux_minus']:.4f} vs {result['flux_plus']:.4f}",
            weight="bold", fontsize=10)
        ax.legend(frameon=False)
        axes[row, 0].text(-.33, .5, result["quantity"].capitalize(),
                          transform=axes[row, 0].transAxes, rotation=90,
                          ha="center", va="center", fontsize=12, weight="bold")

    fig.suptitle("Constructive non-uniqueness of wall flux from full-range moments\n"
                 "All monomial moments through total degree 3 are identical",
                 y=.995, fontsize=14, weight="bold")
    fig.subplots_adjust(left=.075, right=.985, top=.88, bottom=.17,
                        hspace=.42, wspace=.34)
    cax_base = fig.add_axes([.15, .055, .31, .020])
    base_bar = fig.colorbar(contour, cax=cax_base, orientation="horizontal",
                            ticks=[0.0, float(base2.max())/2, float(base2.max())])
    base_bar.set_label("discrete probability density")
    cax_diff = fig.add_axes([.58, .055, .15, .020])
    diff_bar = fig.colorbar(diff, cax=cax_diff, orientation="horizontal",
                            ticks=[relative_levels[0], 0.0, relative_levels[-1]])
    diff_bar.set_label("relative redistribution")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output/f"moment_nonuniqueness_velocity_space.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=31)
    parser.add_argument("--vmax", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=0.35)
    args = parser.parse_args()
    if args.points < 11 or not 0.0 < args.beta < 1.0:
        raise SystemExit("Require --points >= 11 and 0 < --beta < 1")
    args.output.mkdir(parents=True, exist_ok=True)
    velocity = np.linspace(-args.vmax, args.vmax, args.points)
    vn2, vt2 = np.meshgrid(velocity, velocity, indexing="ij")
    vn, vt = vn2.ravel(), vt2.ravel()
    base = np.exp(-0.5*(vn**2+vt**2))
    base /= base.sum()
    names, matrix = moment_matrix(vn, vt)
    results = [construct(vn, vt, base, matrix, quantity, args.beta)
               for quantity in ("pressure", "shear")]
    plot(results, velocity, vn2, vt2, base.reshape(vn2.shape), args.output)

    with (args.output/"moment_nonuniqueness_moments.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["construction", "moment", "f_minus", "f_plus", "difference"])
        for result in results:
            minus_moments, plus_moments = matrix@result["minus"], matrix@result["plus"]
            for name, minus, plus in zip(names, minus_moments, plus_moments):
                writer.writerow([result["quantity"], name, minus, plus, plus-minus])
    summary = {
        "constraints": "all 2D monomial moments with total degree <= 3",
        "grid": {"points_per_axis": args.points, "vmax": args.vmax},
        "relative_perturbation_bound": args.beta,
        "constructions": [{k: v for k, v in result.items()
                           if k not in ("minus", "plus", "delta", "kernel")}
                          for result in results],
    }
    (args.output/"moment_nonuniqueness.json").write_text(json.dumps(summary, indent=2)+"\n")
    lines = ["# Constructive moment non-uniqueness", "",
             "Two non-negative discrete velocity distributions were constructed for each wall functional. "
             "Every full-range monomial moment through total degree three is identical, while the incident "
             "half-range wall functional differs.", "",
             "| Functional | f− | f+ | Difference | Maximum moment mismatch | Minimum probability |",
             "|---|---:|---:|---:|---:|---:|"]
    for result in results:
        lines.append(f"| {result['quantity'].capitalize()} | {result['flux_minus']:.6f} | "
                     f"{result['flux_plus']:.6f} | {result['flux_plus']-result['flux_minus']:.6f} | "
                     f"{result['max_moment_mismatch']:.3e} | {result['minimum_probability']:.3e} |")
    lines += ["", "This establishes pointwise non-uniqueness for the finite full-range S0/S1/S2 moment hierarchy. "
              "It does not by itself prove that every finite spatial patch is non-identifying; the DSMC gate "
              "tests that practical question empirically.", ""]
    (args.output/"MOMENT_NONUNIQUENESS_REPORT.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
