#!/usr/bin/env python3
"""Rebuild the revision figures in a consistent landscape journal style.

The script deliberately separates display smoothing from quantitative analysis.
It never modifies the underlying DSMC or half-range arrays.  Field-validation
composites are split into one landscape row per physical holdout; scalar plots
are redrawn from the archived CSV/PDF data with embedded serif fonts.
"""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd
from PIL import Image


BLUE = "#1769aa"
ORANGE = "#d95f02"
GEOM_COLORS = {"BWD": "#7b3294", "FWD": "#008837", "ISO": "#c51b7d"}
GEOM_LABELS = {"BWD": "Backward-facing", "FWD": "Forward-facing", "ISO": "Symmetric"}
TARGET_LABELS = {"Cp": r"$C_p$", "Cq": r"$C_q$", "tau_abs": r"$|\tau|$"}


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.5,
        "axes.labelsize": 11.5,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 10,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def save(fig, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        destination = out / f"{stem}.{ext}"
        temporary = out / f"{stem}.{ext}.tmp"
        fig.savefig(temporary, format=ext, dpi=400, bbox_inches="tight",
                    facecolor="white")
        temporary.replace(destination)
    plt.close(fig)


def add_box(ax, xy, wh, text, color, fontsize=9.2) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.012",
        edgecolor="0.28", facecolor=color, linewidth=1.0))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, linespacing=1.10)


def add_arrow(ax, p0, p1) -> None:
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=12,
        linewidth=1.1, color="0.28", shrinkA=2, shrinkB=2))


def make_framework(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 4.6))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    ax.text(.015, .98, "(a) Evidence hierarchy", va="top", weight="bold", fontsize=12)
    add_box(ax, (.03, .63), (.18, .18), "DSMC bulk\nand wall tallies", "#dbeaf4")
    add_box(ax, (.28, .70), (.17, .13), "$S_0$\nprimitive fields", "#f2f2f2")
    add_box(ax, (.28, .50), (.17, .13), "$S_1$\n$S_0+P_{ij}$", "#e2f0dd")
    add_box(ax, (.28, .30), (.17, .13), "$S_2$\n$S_1+q_i$", "#fff1c9")
    add_box(ax, (.53, .50), (.20, .18),
            "$S_{\\mathrm{HR}}$\nincoming half-range\nstate", "#f5dfe6")
    add_box(ax, (.80, .50), (.17, .18), "Wall kernel\n(diffuse reflection)", "#e8e0f1")
    for yy in (.765, .565, .365):
        add_arrow(ax, (.21, .72), (.28, yy))
    add_arrow(ax, (.21, .72), (.53, .59))
    add_arrow(ax, (.73, .59), (.80, .59))
    ax.text(.375, .18, "Equal-capacity blind interpolation", ha="center", fontsize=9.3)
    ax.text(.75, .18, "Direct kinetic reconstruction", ha="center", fontsize=9.3)
    ax.plot([.49, .49], [.15, .86], color="0.75", lw=.8, ls="--")

    ax = axes[1]
    ax.text(.015, .98, "(b) Physics-first validation logic", va="top", weight="bold", fontsize=12)
    cx, cy = .13, .56
    ax.add_patch(Polygon(np.array([[.04, .26], [.22, .26], [.13, .72]]),
                         closed=False, fill=False, edgecolor="0.35", linewidth=1.8))
    for rr, col in ((.12, "#f3ead8"), (.08, "#e2f0dd"), (.04, "#dbeaf4")):
        ax.add_patch(Circle((cx, cy), rr, facecolor=col, edgecolor="0.65",
                            linewidth=.6, alpha=.75))
    ax.add_patch(Circle((cx, cy), .014, facecolor="#c73e78", edgecolor="0.25"))
    add_box(ax, (.29, .66), (.20, .16), "Raw DSMC\nwall target", "#dbeaf4")
    add_box(ax, (.29, .38), (.20, .16), "Constructive\nnonuniqueness test", "#fff1c9")
    add_box(ax, (.58, .66), (.18, .16), "Co-temporal\nblock test", "#e2f0dd")
    add_box(ax, (.58, .38), (.18, .16), "Independent\nwindow test", "#f5dfe6")
    add_box(ax, (.83, .51), (.15, .18), "$C_p$, signed $C_f$\nwith uncertainty", "#e8e0f1")
    add_arrow(ax, (.22, .56), (.29, .74))
    add_arrow(ax, (.22, .56), (.29, .46))
    add_arrow(ax, (.49, .74), (.58, .74))
    add_arrow(ax, (.49, .46), (.58, .46))
    add_arrow(ax, (.76, .74), (.83, .62))
    add_arrow(ax, (.76, .46), (.83, .58))
    ax.text(.50, .17,
            "The neural surrogate is a comparison tool; the central claim is tested\n"
            "with molecular collision statistics and direct DSMC surface impulses.",
            ha="center", va="center", fontsize=9.4)
    fig.subplots_adjust(left=.02, right=.99, bottom=.04, top=.98, wspace=.06)
    save(fig, out, "fig02_framework_landscape")


def make_radius_error(root: Path, out: Path) -> None:
    df = pd.read_csv(root / "data/error_summary_by_config_target.csv")
    order = ["parameter_only", "local", "R0p05", "R0p10", "R0p20", "R0p35",
             "R0p50", "R0p75", "R1", "R1p50", "R2", "R3", "full"]
    labels = ["Param.", "Local", ".05", ".10", ".20", ".35", ".50",
              ".75", "1", "1.5", "2", "3", "Full"]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.15), sharex=True)
    x = np.arange(len(order))
    for k, (ax, target) in enumerate(zip(axes, ("Cp", "Cq", "tau_abs"))):
        sub = df[df.target == target].set_index("config")
        y = [100*float(sub.loc[c, "relL2"]) for c in order]
        ax.plot(x, y, color=BLUE, marker="o", lw=2.1, ms=5.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel(r"Input / radius $R/h_s$")
        ax.set_ylabel("Mean LOOCV error (%)")
        ax.text(.97, .96, f"({chr(97+k)}) {TARGET_LABELS[target]}",
                transform=ax.transAxes, ha="right", va="top", weight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=.86, pad=1.5))
        ax.grid(axis="y", color="0.84", lw=.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=.06, right=.995, bottom=.25, top=.97, wspace=.24)
    save(fig, out, "fig10_bulk_to_wall_error_landscape")


def _line_y(d):
    return np.array([d["items"][0][1].y] + [item[2].y for item in d["items"]][:12])


def _band_y(d):
    items = d["items"]
    low = [items[0][2].y] + [items[i][2].y for i in range(1, 13)]
    high_rev = [items[13][2].y] + [items[i][2].y for i in range(14, 26)]
    return np.asarray(low), np.asarray(list(reversed(high_rev)))


def make_closed_loop(root: Path, out: Path) -> None:
    doc = fitz.open(root / "data/fig13_closed_loop_error_curves_original_side_by_side.pdf")
    draws = doc[0].get_drawings()
    blue = (0.12156862765550613, 0.46666666865348816, 0.7058823704719543)
    red = (0.8392156958580017, 0.15294118225574493, 0.1568627506494522)
    cfgs = [
        {"y0": 414.3979797363281, "dy": 10.304864501953125, "label": r"$C_p$"},
        {"y0": 404.781494140625, "dy": 4.0499542236328125, "label": r"$C_q$"},
        {"y0": 410.8476257324219, "dy": 3.628887939453125, "label": r"$|\tau|$"},
    ]
    for cfg in cfgs:
        cfg.update(line_blue=None, line_red=None, fill_blue=None, fill_red=None)
    for d in draws:
        xc = (d["rect"].x0 + d["rect"].x1)/2
        pi = 0 if xc < 435 else (1 if xc < 870 else (2 if xc > 880 else None))
        if pi is None:
            continue
        width = d.get("width") or 0
        if d.get("type") == "s" and abs(width-2.4) < .05 and len(d["items"]) >= 12:
            if d.get("color") == blue:
                cfgs[pi]["line_blue"] = d
            if d.get("color") == red:
                cfgs[pi]["line_red"] = d
        if d.get("type") == "fs" and len(d["items"]) == 26 and d.get("fill_opacity", 1) < .5:
            if d.get("fill") == blue:
                cfgs[pi]["fill_blue"] = d
            if d.get("fill") == red:
                cfgs[pi]["fill_red"] = d

    labels = ["Param.", "Local", ".05", ".10", ".20", ".35", ".50",
              ".75", "1", "1.5", "2", "3", "Full"]
    x = np.arange(13)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.15))
    for k, (ax, cfg) in enumerate(zip(axes, cfgs)):
        for line_d, fill_d, color, label in (
                (cfg["line_blue"], cfg["fill_blue"], BLUE, "Raw DSMC field"),
                (cfg["line_red"], cfg["fill_red"], ORANGE, "Surrogate field")):
            median = (cfg["y0"] - _line_y(line_d))/cfg["dy"]
            low_pdf, high_pdf = _band_y(fill_d)
            lower = (cfg["y0"] - low_pdf)/cfg["dy"]
            upper = (cfg["y0"] - high_pdf)/cfg["dy"]
            ax.plot(x, median, color=color, marker="o", lw=2.0, ms=4.8, label=label)
            ax.fill_between(x, lower, upper, color=color, alpha=.16, linewidth=0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel(r"Input / radius $R/h_s$")
        ax.set_ylabel("Median LOOCV error (%)")
        ax.text(.02, .97, f"({chr(97+k)}) {cfg['label']}",
                transform=ax.transAxes, va="top", weight="bold")
        ax.grid(axis="y", color="0.84", lw=.6)
        ax.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.5, 1.01),
               ncol=2, frameon=False)
    fig.subplots_adjust(left=.06, right=.995, bottom=.25, top=.88, wspace=.25)
    save(fig, out, "fig13_closed_loop_error_curves_landscape")


def make_r95_kn(root: Path, out: Path) -> None:
    df = pd.read_csv(root / "data/information_horizon_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.0), sharey=True)
    for k, (ax, target) in enumerate(zip(axes, ("Cp", "Cq", "tau_abs"))):
        for geom in ("BWD", "FWD", "ISO"):
            g = df[(df.target == target) & (df.geom == geom)].sort_values("Kn")
            ax.plot(g.Kn, g.R95_over_hs, color=GEOM_COLORS[geom], marker="o",
                    lw=2.0, ms=5.5, label=GEOM_LABELS[geom])
        ax.set_xscale("log")
        ax.set_xticks([.1, .33, .8])
        ax.set_xticklabels(["0.10", "0.33", "0.80"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Knudsen number")
        ax.set_ylabel(r"Lower-bound $R_{95}/h_s$")
        ax.text(.02, .97, f"({chr(97+k)}) {TARGET_LABELS[target]}",
                transform=ax.transAxes, va="top", weight="bold")
        ax.grid(color="0.84", lw=.6)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.01),
               ncol=3, frameon=False)
    fig.subplots_adjust(left=.06, right=.995, bottom=.18, top=.86, wspace=.12)
    save(fig, out, "fig11_information_horizon_kn_landscape")


def make_r95_uq(root: Path, out: Path) -> None:
    text = (root / "data/treebootstrap_UQ_summary.txt").read_text()
    lines = [line for line in text.splitlines() if line.strip().startswith("Cq ")]
    rows = []
    for line in lines:
        fields = line.split()
        rows.append({
            "geom": fields[1], "Kn": float(fields[2]), "median": float(fields[3]),
            "lo": float(fields[4]), "hi": float(fields[5]), "censored": float(fields[6])})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.0), sharey=True)
    for k, (ax, geom) in enumerate(zip(axes, ("BWD", "FWD", "ISO"))):
        g = df[df.geom == geom].sort_values("Kn")
        y = g["median"].to_numpy()
        lo = g["lo"].to_numpy()
        hi = g["hi"].to_numpy()
        ax.errorbar(g.Kn, y, yerr=[y-lo, hi-y], color=GEOM_COLORS[geom],
                    marker="o", capsize=4, lw=2.0, ms=6)
        for x, yy, cens in zip(g.Kn, y, g.censored):
            ax.text(x, yy+.12, f"{100*cens:.0f}% cens.", ha="center", fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks([.1, .33, .8])
        ax.set_xticklabels(["0.10", "0.33", "0.80"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Knudsen number")
        ax.set_ylabel(r"Lower-bound $R_{95}/h_s$ for $C_q$")
        ax.text(.02, .97, f"({chr(97+k)}) {GEOM_LABELS[geom]}",
                transform=ax.transAxes, va="top", weight="bold")
        ax.set_ylim(0, 3.35)
        ax.grid(color="0.84", lw=.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=.065, right=.995, bottom=.18, top=.96, wspace=.13)
    save(fig, out, "fig14_R95_treebootstrap_UQ_landscape")


def make_bulk_profiles(root: Path, out: Path) -> None:
    df = pd.read_csv(root / "data/loocv_predictions_surface.csv")
    cases = [
        "PHASE1__Ma6_Kn0.33_BWD_hphs1.5_Tw1",
        "PHASE1__Ma6_Kn0.33_FWD_hphs1.5_Tw1",
        "PHASE1__Ma6_Kn0.33_ISO_hphs1.5_Tw1",
    ]
    targets = ("Cp", "Cq", "tau_abs")
    configs = ("parameter_only", "local", "R0p50", "R1p50", "full")
    config_labels = {
        "parameter_only": "Parameters", "local": "Wall-local",
        "R0p50": r"$R/h_s=0.5$", "R1p50": r"$R/h_s=1.5$",
        "full": "Full field"}
    colors = {
        "parameter_only": "#d73027", "local": "#fc8d59",
        "R0p50": "#91bfdb", "R1p50": "#4575b4", "full": "#313695"}
    fig, axes = plt.subplots(3, 3, figsize=(15.2, 7.4), sharex=True)
    for row, case_id in enumerate(cases):
        case = df[df.case_id == case_id]
        geom = "BWD" if "_BWD_" in case_id else ("FWD" if "_FWD_" in case_id else "ISO")
        for col, target in enumerate(targets):
            ax = axes[row, col]
            truth = case[case.config == "full"].sort_values("s01")
            ax.plot(truth.s01, truth[f"true_{target}"], color="black", lw=2.2,
                    label="DSMC", zorder=5)
            for cfg in configs:
                g = case[case.config == cfg].sort_values("s01")
                if not g.empty:
                    ax.plot(g.s01, g[f"pred_{target}"], color=colors[cfg], lw=1.35,
                            label=config_labels[cfg], alpha=.96)
            if row == 0:
                ax.set_title(TARGET_LABELS[target], weight="bold", pad=5)
            if col == 0:
                ax.text(.02, .94, f"({chr(97+row)}) {GEOM_LABELS[geom]}",
                        transform=ax.transAxes, va="top", weight="bold",
                        bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1.2))
            ax.set_xlim(0, 1)
            ax.set_xlabel("Normalized wall arclength")
            ax.set_ylabel(TARGET_LABELS[target])
            ax.grid(color="0.85", lw=.55)
            ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .995),
               ncol=6, frameon=False, columnspacing=1.3)
    fig.subplots_adjust(left=.06, right=.995, bottom=.08, top=.89,
                        wspace=.24, hspace=.34)
    save(fig, out, "fig12_bulk_to_wall_profiles_landscape")


def split_rows(root: Path, out: Path) -> None:
    specs = {
        "fig04_streamwise_velocity_validation_clean.png": "fig04_u",
        "fig05_transverse_velocity_validation_clean.png": "fig05_v",
        "fig06_pressure_validation_clean.png": "fig06_p",
        "fig07_surface_validation_clean.png": "fig07_surface",
        "fig08_entropy_orientation_clean.png": "fig08_entropy_orientation",
        "fig09_entropy_height_clean.png": "fig09_entropy_height",
    }
    for filename, stem in specs.items():
        image = Image.open(root / "figures" / filename).convert("RGB")
        width, height = image.size
        edges = np.linspace(0, height, 4).astype(int)
        rows = []
        for row in range(3):
            crop = image.crop((0, edges[row], width, edges[row+1]))
            rows.append(crop)
            png = out / f"{stem}_row{row+1}.png"
            pdf = out / f"{stem}_row{row+1}.pdf"
            crop.save(png, dpi=(400, 400), optimize=True)
            crop.save(pdf, resolution=400)
        row_height = max(row.height for row in rows)
        canvas = Image.new("RGB", (2*width, 2*row_height), "white")
        for idx, row_image in enumerate(rows):
            if row_image.height != row_height:
                padded = Image.new("RGB", (width, row_height), "white")
                padded.paste(row_image, (0, (row_height-row_image.height)//2))
                row_image = padded
            x = 0 if idx == 0 else (width if idx == 1 else width//2)
            y = 0 if idx < 2 else row_height
            canvas.paste(row_image, (x, y))
        canvas.save(out / f"{stem}_mosaic_landscape.png", dpi=(400, 400), optimize=True)
        canvas.save(out / f"{stem}_mosaic_landscape.pdf", resolution=400)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manuscript_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    set_style()
    make_framework(args.output)
    make_radius_error(args.manuscript_root, args.output)
    make_closed_loop(args.manuscript_root, args.output)
    make_r95_kn(args.manuscript_root, args.output)
    make_r95_uq(args.manuscript_root, args.output)
    make_bulk_profiles(args.manuscript_root, args.output)
    split_rows(args.manuscript_root, args.output)


if __name__ == "__main__":
    main()
