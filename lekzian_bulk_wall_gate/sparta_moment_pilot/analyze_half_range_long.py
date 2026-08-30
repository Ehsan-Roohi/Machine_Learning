#!/usr/bin/env python3
"""Elementwise analysis of the long SPARTA incident half-range pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
from scipy.stats import t as student_t
from scipy.signal import savgol_filter

K_B = 1.380649e-23
M_AR = 6.6335209e-26
HALF_RANGE_LABEL = r"Half-range reconstruction, $S_{\mathrm{HR}}$"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Times"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def snapshots(path: Path, ncol: int):
    lines = path.read_text().splitlines()
    out, i = [], 0
    while i < len(lines):
        if lines[i] != "ITEM: TIMESTEP":
            i += 1
            continue
        step = int(lines[i + 1])
        if lines[i + 2] != "ITEM: NUMBER OF TALLIES":
            raise ValueError(f"Malformed count at step {step}")
        count = int(lines[i + 3])
        h = i + 8
        if not lines[h].startswith("ITEM: TALLIES "):
            raise ValueError(f"Malformed tally header at step {step}")
        a = np.loadtxt(lines[h + 1:h + 1 + count], ndmin=2) if count else np.empty((0, ncol))
        if a.shape != (count, ncol):
            raise ValueError(f"Unexpected shape {a.shape} at step {step}")
        out.append((step, a))
        i = h + 1 + count
    return out


def wall_file(path: Path):
    lines = path.read_text().splitlines()
    h = next(i for i, x in enumerate(lines) if x.startswith("ITEM: SURFS "))
    names = lines[h].split()[2:]
    count_header = lines.index("ITEM: NUMBER OF SURFS")
    n = int(lines[count_header + 1])
    a = np.loadtxt(lines[h + 1:h + 1 + n], ndmin=2)
    return a, {name: i for i, name in enumerate(names)}


def nrmse(y, p):
    return float(np.sqrt(np.mean((p-y)**2)) / max(np.sqrt(np.mean(y**2)), 1e-30))


def corr(y, p):
    return float(np.corrcoef(y, p)[0, 1])


def analyze_case(case: Path, label: str = "half_range_long"):
    meta = json.loads((case / "metadata.json").read_text())
    schema = meta["column_schema"]["collision"]
    ci = {x: i for i, x in enumerate(schema)}
    wfiles = sorted((case / "output" / label).glob("wall.*.dat"))
    if len(wfiles) < 2:
        raise ValueError(f"{case.name}: expected at least two wall blocks")
    wall_steps = [int(path.name.split(".")[1]) for path in wfiles]
    block_steps = wall_steps[0]
    if wall_steps != [block_steps*(i+1) for i in range(len(wfiles))]:
        raise ValueError(f"{case.name}: wall-block timesteps are not contiguous")
    nblocks = len(wfiles)
    wb, wc = zip(*(wall_file(p) for p in wfiles))
    if any(c != wc[0] for c in wc):
        raise ValueError("Wall schemas differ")
    wc, wall = wc[0], np.stack(wb)
    order = np.argsort(wall[0, :, wc["id"]])
    wall = wall[:, order]
    ids = wall[0, :, wc["id"]].astype(int)
    if len(ids) != 60 or not np.array_equal(ids, np.arange(981, 1041)):
        raise ValueError(f"{case.name}: expected surface IDs 981..1040")
    v1 = wall[0, :, [wc["v1x"], wc["v1y"]]].T
    v2 = wall[0, :, [wc["v2x"], wc["v2y"]]].T
    tangent = v2-v1
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    # Positive signed shear is the streamwise (positive-x) component on both
    # faces.  This convention remains meaningful for the overhanging FWD/BWD
    # profiles, where surface-ID direction is not everywhere streamwise.
    tangent[tangent[:, 0] < 0] *= -1
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    length = wall[0, :, wc["area"]]
    midpoint = (v1+v2)/2
    s = (np.cumsum(length)-0.5*length)/np.sum(length)
    apex_s = float(np.sum(length[:30])/np.sum(length))

    raw = snapshots(case / "output" / label / "collisions.dat", len(schema))
    # Step 1 is the deliberately empty initialization snapshot.  A final dump
    # one step beyond the last ave/surf block can also be present and is
    # excluded.  The block size is recovered from the wall-dump timesteps.
    by_block = [[] for _ in range(nblocks)]
    ignored = []
    for step, a in raw:
        if step == 1 or step > wall_steps[-1]:
            ignored.append((step, len(a)))
            continue
        b = (step-1)//block_steps
        if not 0 <= b < nblocks:
            raise ValueError(f"Unexpected step {step}")
        by_block[b].append(a)

    # A globally upward normal is wrong for the overhanging FWD/BWD faces.
    # Orient each element from solid into the locally sampled gas using only
    # incident (pre-collision) velocities.  Incoming normal velocity must be
    # negative.  This is leakage-safe: neither post-collision velocity nor a
    # wall target enters the orientation decision.
    orientation_rows = np.vstack([rows for block in by_block for rows in block])
    orientation_ix = orientation_rows[:, ci["surface_id"]].astype(int)-981
    orientation_pre = orientation_rows[:, [ci["vx_pre"], ci["vy_pre"]]]
    candidate_vn = np.sum(orientation_pre*normal[orientation_ix], axis=1)
    vn_sum = np.bincount(orientation_ix, weights=candidate_vn, minlength=60)
    vn_count = np.bincount(orientation_ix, minlength=60)
    if np.any(vn_count == 0):
        raise ValueError(f"{case.name}: an element has no incident records")
    normal[vn_sum/vn_count > 0] *= -1

    phys = meta["physics"]
    fnum, dt = phys["fnum_particles_per_simulator"], phys["timestep_s"]
    pinf = phys["number_density_m_minus_3"]*K_B*phys["temperature_K"]
    qinf = 0.5*phys["number_density_m_minus_3"]*M_AR*phys["stream_speed_m_per_s"]**2
    escale = pinf*phys["stream_speed_m_per_s"]
    norms = np.array([qinf, qinf, escale])
    offsets = np.array([pinf, 0, 0])
    evn = math.sqrt(math.pi*K_B*meta["wall_temperature_K"]/(2*M_AR))
    epost = 2*K_B*meta["wall_temperature_K"]

    incident, full, counts = [], [], []
    for block in by_block:
        a = np.vstack(block)
        ix = a[:, ci["surface_id"]].astype(int)-981
        pre = a[:, [ci["vx_pre"], ci["vy_pre"], ci["vz_pre"]]]
        post = a[:, [ci["vx_post"], ci["vy_post"], ci["vz_post"]]]
        vn0 = np.sum(pre[:, :2]*normal[ix], axis=1)
        vn1 = np.sum(post[:, :2]*normal[ix], axis=1)
        vt0 = np.sum(pre[:, :2]*tangent[ix], axis=1)
        vt1 = np.sum(post[:, :2]*tangent[ix], axis=1)
        inc = np.column_stack((M_AR*(evn-vn0), M_AR*vt0,
                               0.5*M_AR*np.sum(pre**2, axis=1)-epost))
        ful = np.column_stack((M_AR*(vn1-vn0), M_AR*(vt0-vt1),
                               0.5*M_AR*(np.sum(pre**2, axis=1)-np.sum(post**2, axis=1))))
        factor = fnum/(length*block_steps*dt)
        incident.append(np.column_stack([np.bincount(ix, weights=inc[:, j], minlength=60)*factor for j in range(3)]))
        full.append(np.column_stack([np.bincount(ix, weights=ful[:, j], minlength=60)*factor for j in range(3)]))
        counts.append(np.bincount(ix, minlength=60))
    incident, full, counts = map(np.asarray, (incident, full, counts))

    shear = wall[:, :, wc["f_fWallLong[3]"]]*tangent[:, 0] + wall[:, :, wc["f_fWallLong[4]"]]*tangent[:, 1]
    target_flux = np.stack((wall[:, :, wc["f_fWallLong[2]"]], shear,
                            wall[:, :, wc["f_fWallLong[9]"]]), axis=2)
    target = (target_flux-offsets)/norms
    incident = (incident-offsets)/norms
    full = (full-offsets)/norms
    mean = {"target": target.mean(0), "incident": incident.mean(0), "full": full.mean(0)}
    sem = {"target": target.std(0, ddof=1)/math.sqrt(nblocks),
           "incident": incident.std(0, ddof=1)/math.sqrt(nblocks)}
    metrics = {}
    for j, name in enumerate(("cp", "cf", "cq")):
        metrics[name] = {
            "incident_nrmse": nrmse(mean["target"][:, j], mean["incident"][:, j]),
            "incident_correlation": corr(mean["target"][:, j], mean["incident"][:, j]),
            "full_control_nrmse": nrmse(mean["target"][:, j], mean["full"][:, j]),
            "windward_incident_nrmse": nrmse(mean["target"][:30, j], mean["incident"][:30, j]),
            "leeward_incident_nrmse": nrmse(mean["target"][30:, j], mean["incident"][30:, j]),
            "target_rms_sem_percent": 100*np.sqrt(np.mean(sem["target"][:, j]**2))/max(np.sqrt(np.mean(mean["target"][:, j]**2)), 1e-30),
        }
    return {"case_id": case.name, "geometry": meta["geometry"], "kn": phys["knudsen"],
            "s": s, "apex_s": apex_s, "midpoint": midpoint, "v1": v1, "v2": v2,
            "ids": ids, "mean": mean, "sem": sem, "counts": counts,
            "metrics": metrics, "ignored_snapshots": ignored,
            "nblocks": nblocks, "block_steps": block_steps,
            "t95": float(student_t.ppf(.975, nblocks-1)),
            "used_records": int(counts.sum()), "minimum_element_records": int(counts.sum(0).min())}


def add_independent_reference(case: dict, npz_path: Path, metadata_path: Path) -> None:
    """Attach the independent four-block production wall target to a case."""
    meta = json.loads(metadata_path.read_text())
    with np.load(npz_path, allow_pickle=False) as data:
        wall = np.asarray(data["wall"])
        columns = {name: i for i, name in enumerate(data["wall_columns"].tolist())}
    ids = wall[0, :, columns["id"]].astype(int)
    selected = np.flatnonzero((ids >= 981) & (ids <= 1040))
    wall = wall[:, selected]
    order = np.argsort(wall[0, :, columns["id"]])
    wall = wall[:, order]
    if not np.array_equal(wall[0, :, columns["id"]].astype(int), case["ids"]):
        raise ValueError(f"{case['case_id']}: independent-reference surface IDs differ")
    v1 = wall[0, :, [columns["v1x"], columns["v1y"]]].T
    v2 = wall[0, :, [columns["v2x"], columns["v2y"]]].T
    tangent = v2-v1
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    tangent[tangent[:, 0] < 0] *= -1
    shear = (wall[:, :, columns["shx"]]*tangent[:, 0]
             + wall[:, :, columns["shy"]]*tangent[:, 1])
    phys = meta["physics"]
    pinf = phys["number_density_m_minus_3"]*K_B*phys["temperature_K"]
    qinf = 0.5*phys["number_density_m_minus_3"]*M_AR*phys["stream_speed_m_per_s"]**2
    escale = pinf*phys["stream_speed_m_per_s"]
    target = np.stack(((wall[:, :, columns["press"]]-pinf)/qinf,
                       shear/qinf, wall[:, :, columns["etot"]]/escale), axis=2)
    case["reference"] = target.mean(0)
    case["reference_sem"] = target.std(0, ddof=1)/math.sqrt(target.shape[0])
    case["reference_t95"] = float(student_t.ppf(.975, target.shape[0]-1))
    for j, name in enumerate(("cp", "cf", "cq")):
        metric = case["metrics"][name]
        metric["independent_incident_nrmse"] = nrmse(
            case["reference"][:, j], case["mean"]["incident"][:, j])
        metric["independent_incident_correlation"] = corr(
            case["reference"][:, j], case["mean"]["incident"][:, j])
        metric["independent_target_rms_sem_percent"] = (
            100*np.sqrt(np.mean(case["reference_sem"][:, j]**2))
            / max(np.sqrt(np.mean(case["reference"][:, j]**2)), 1e-30))
        metric["incident_rms_sem_percent"] = (
            100*np.sqrt(np.mean(case["sem"]["incident"][:, j]**2))
            / max(np.sqrt(np.mean(case["mean"]["incident"][:, j]**2)), 1e-30))


def _surface_collection(case: dict, values: np.ndarray, norm, cmap: str) -> LineCollection:
    hp = 0.03
    segments = np.stack(((case["v1"]-np.array([0.22, 0.0]))/hp,
                         (case["v2"]-np.array([0.22, 0.0]))/hp), axis=1)
    collection = LineCollection(segments, cmap=cmap, norm=norm, linewidths=7.0,
                                capstyle="round")
    collection.set_array(values)
    return collection


def plot_physical(cases, out: Path, quantity: str, independent: bool = False):
    """Plot DSMC and half-range reconstruction on each wall and profile."""
    j = {"cp": 0, "cf": 1}[quantity]
    label = {"cp": r"$C_p=(p_w-p_\infty)/(\frac{1}{2}\rho_\infty U_\infty^2)$",
             "cf": r"$C_f=\tau_w/(\frac{1}{2}\rho_\infty U_\infty^2)$"}[quantity]
    target_key = "reference" if independent else "target"
    sem_key = "reference_sem" if independent else None
    shown = [c for c in cases if not independent or "reference" in c]
    if not shown:
        return
    all_values = np.concatenate([
        np.r_[c[target_key][:, j] if independent else c["mean"][target_key][:, j],
              c["mean"]["incident"][:, j]] for c in shown])
    if quantity == "cf" and np.min(all_values) < 0 < np.max(all_values):
        bound = float(np.max(np.abs(all_values)))
        norm, cmap = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound), "coolwarm"
    else:
        lo, hi = float(np.min(all_values)), float(np.max(all_values))
        pad = 0.02*max(hi-lo, 1e-12)
        norm, cmap = Normalize(lo-pad, hi+pad), "viridis"

    fig, axes = plt.subplots(len(shown), 3, figsize=(14.2, 3.0*len(shown)+1.8),
                             squeeze=False)
    for row, (ax_dsmc, ax_s3, ax_profile, case) in enumerate(zip(
            axes[:, 0], axes[:, 1], axes[:, 2], shown)):
        truth = case[target_key][:, j] if independent else case["mean"][target_key][:, j]
        truth_sem = case[sem_key][:, j] if independent else case["sem"][target_key][:, j]
        incident = case["mean"]["incident"][:, j]
        for ax, values in ((ax_dsmc, truth), (ax_s3, incident)):
            polygon = np.vstack((case["v1"][0], case["v2"][:30][-1], case["v2"][-1]))
            polygon = (polygon-np.array([0.22, 0.0]))/0.03
            ax.fill(polygon[:, 0], polygon[:, 1], color="0.92", zorder=0)
            ax.add_collection(_surface_collection(case, values, norm, cmap))
            ax.set_xlim(-0.45, 1.12)
            ax.set_ylim(-0.08, 1.10)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=.16)
            ax.set_xlabel(r"$(x-0.22)/h_p$")
            ax.set_ylabel(r"$y/h_p$")
        critical = case["reference_t95"] if independent else case["t95"]
        ci = critical*truth_sem
        ax_profile.fill_between(case["s"], truth-ci, truth+ci, color="0.78",
                                alpha=.55, linewidth=0, label="DSMC 95% CI")
        ax_profile.plot(case["s"], incident, "s-", color="#2f74b5", ms=2.8,
                        lw=1.8, label=HALF_RANGE_LABEL, zorder=3)
        ax_profile.plot(case["s"], truth, "o", color="black", mfc="white",
                        mew=.9, ms=3.2, label="DSMC", zorder=4)
        ax_profile.axvline(case["apex_s"], color="0.42", ls=":", lw=1.1)
        ax_profile.axhline(0, color="0.6", lw=.8)
        ax_profile.grid(alpha=.22)
        ax_profile.set_xlabel(r"physical wall arclength, $s/L_w$")
        ax_profile.set_ylabel(label)
        metric = case["metrics"][quantity]
        if independent:
            error = metric["independent_incident_nrmse"]
            correlation = metric["independent_incident_correlation"]
        else:
            error = metric["incident_nrmse"]
            correlation = metric["incident_correlation"]
        case["_plot_row_label"] = (
            f"{case['geometry']}\n$Kn={case['kn']:g}$\n"
            f"NRMSE={100*error:.2f}%\n$r={correlation:.4f}$")

    axes[0, 0].set_title("DSMC surface tally", fontsize=12, weight="bold", pad=12)
    axes[0, 1].set_title(r"Half-range reconstruction, $S_{\mathrm{HR}}$",
                         fontsize=12, weight="bold", pad=12)
    axes[0, 2].set_title("Physical wall profile", fontsize=12, weight="bold", pad=12)
    handles, labels = axes[0, 2].get_legend_handles_labels()
    order = [labels.index("DSMC 95% CI"), labels.index("DSMC"),
             labels.index(HALF_RANGE_LABEL)]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="upper center", bbox_to_anchor=(0.72, 0.965),
               ncol=3, frameon=False, fontsize=10)
    source = "independent 40,000-step DSMC reference" if independent else "concurrent five-block DSMC target"
    fig.suptitle(f"{label}: DSMC versus half-range reconstruction\n({source})",
                 y=0.997, fontsize=14, weight="bold")
    fig.subplots_adjust(left=.15, right=.91, top=.91, bottom=.055, hspace=.58, wspace=.34)
    for ax, case in zip(axes[:, 0], shown):
        box = ax.get_position()
        fig.text(.048, .5*(box.y0+box.y1), case.pop("_plot_row_label"),
                 ha="center", va="center", fontsize=10.5, weight="bold")
    colorbar = fig.colorbar(_surface_collection(shown[0], shown[0]["mean"]["incident"][:, j], norm, cmap),
                            ax=axes[:, :2], location="right", fraction=.024, pad=.025)
    colorbar.set_label(label)
    stem = f"cross_{'window' if independent else 'geometry'}_{quantity}_physical_{'independent' if independent else 'concurrent'}"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out/f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _quantity_style(quantity: str):
    j = {"cp": 0, "cf": 1}[quantity]
    label = {
        "cp": r"$C_p=(p_w-p_\infty)/(\frac{1}{2}\rho_\infty U_\infty^2)$",
        "cf": r"$C_f=\tau_w/(\frac{1}{2}\rho_\infty U_\infty^2)$",
    }[quantity]
    return j, label


def _save_article_figure(fig, out: Path, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        destination = out/f"{stem}.{ext}"
        temporary = out/f"{stem}.{ext}.tmp"
        fig.savefig(temporary, format=ext, dpi=400, bbox_inches="tight",
                    facecolor="white")
        temporary.replace(destination)
    plt.close(fig)


def _facewise_display_smooth(values: np.ndarray) -> np.ndarray:
    """Smooth each 30-element face only; never smooth across the apex jump."""
    values = np.asarray(values, dtype=float)
    smoothed = values.copy()
    for face in (slice(0, 30), slice(30, 60)):
        smoothed[face] = savgol_filter(values[face], 7, 2, mode="interp")
    return smoothed


def plot_paper_profiles(cases, out: Path, quantity: str,
                        independent: bool = False) -> None:
    """Publication landscape: six physical wall profiles in a 2-by-3 grid."""
    j, label = _quantity_style(quantity)
    target_key = "reference" if independent else "target"
    sem_key = "reference_sem" if independent else None
    shown = [case for case in cases if not independent or "reference" in case]
    if not shown:
        return
    by_geometry = {
        geometry: sorted((case for case in shown if case["geometry"] == geometry),
                         key=lambda case: case["kn"])
        for geometry in ("ISO", "BWD", "FWD")
    }
    if any(len(group) != 2 for group in by_geometry.values()):
        return

    fig, axes = plt.subplots(2, 3, figsize=(14.8, 6.15), sharey=True,
                             constrained_layout=False)
    for col, geometry in enumerate(("ISO", "BWD", "FWD")):
        for row, case in enumerate(by_geometry[geometry]):
            ax = axes[row, col]
            truth = (case[target_key][:, j] if independent
                     else case["mean"][target_key][:, j])
            sem = (case[sem_key][:, j] if independent
                   else case["sem"][target_key][:, j])
            critical = case["reference_t95"] if independent else case["t95"]
            reconstruction = case["mean"]["incident"][:, j]
            truth_display = _facewise_display_smooth(truth)
            reconstruction_display = _facewise_display_smooth(reconstruction)
            ci = critical*sem
            ax.axvspan(0.0, case["apex_s"], color="#f3ead8", alpha=.55,
                       zorder=0)
            ax.axvspan(case["apex_s"], 1.0, color="#e8eef5", alpha=.55,
                       zorder=0)
            ax.fill_between(case["s"], truth-ci, truth+ci, color="0.70",
                            alpha=.30, linewidth=0, label="DSMC 95% CI")
            ax.plot(case["s"], reconstruction_display, color="#1874b4",
                    lw=2.0, label=HALF_RANGE_LABEL, zorder=3)
            ax.plot(case["s"], reconstruction, ls="none", marker="s",
                    color="#1874b4", ms=3.0, markevery=2, zorder=4)
            ax.plot(case["s"], truth_display, color="black", lw=1.25,
                    label="DSMC", zorder=4)
            ax.plot(case["s"], truth, ls="none", marker="o", color="black",
                    mfc="white", mec="black", mew=.8, ms=3.1,
                    markevery=2, zorder=5)
            ax.axvline(case["apex_s"], color="0.35", ls=":", lw=1.1)
            ax.axhline(0, color="0.55", lw=.7)
            ax.grid(color="0.82", lw=.6, alpha=.55)
            ax.set_xlim(0, 1)
            metric = case["metrics"][quantity]
            error_key = ("independent_incident_nrmse" if independent
                         else "incident_nrmse")
            corr_key = ("independent_incident_correlation" if independent
                        else "incident_correlation")
            ax.set_title(fr"{geometry}, $Kn={case['kn']:g}$", fontsize=12,
                         weight="bold", pad=6)
            panel = row*3 + col
            ax.text(.025, .95, f"({chr(97+panel)})", transform=ax.transAxes,
                    ha="left", va="top", fontsize=11, weight="bold")
            ax.text(.975, .95,
                    fr"NRMSE = {100*metric[error_key]:.2f}%" + "\n"
                    + fr"$r = {metric[corr_key]:.4f}$",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9.5,
                    bbox=dict(facecolor="white", edgecolor="0.78",
                              boxstyle="round,pad=0.24", alpha=.92))

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    order = [legend_labels.index("DSMC"), legend_labels.index(HALF_RANGE_LABEL),
             legend_labels.index("DSMC 95% CI")]
    fig.legend([handles[i] for i in order], [legend_labels[i] for i in order],
               loc="upper center", bbox_to_anchor=(.5, .985), ncol=3,
               frameon=False, fontsize=11, columnspacing=2.0)
    fig.supylabel(label, fontsize=12, x=.018)
    fig.supxlabel(r"Physical wall arclength, $s/L_w$", fontsize=12, y=.018)
    source = ("independent 40,000-step DSMC reference" if independent
              else "co-temporal block-averaged DSMC reference")
    fig.text(.995, .012, source, ha="right", va="bottom", fontsize=8.5,
             color="0.35")
    fig.text(.005, .012,
             "Markers: raw binned values; lines: facewise display smoothing; metrics use unsmoothed data",
             ha="left", va="bottom", fontsize=8.0, color="0.35")
    fig.subplots_adjust(left=.075, right=.99, bottom=.11, top=.88,
                        wspace=.12, hspace=.28)
    stem = f"paper_{quantity}_profiles_{'independent' if independent else 'concurrent'}_landscape"
    _save_article_figure(fig, out, stem)


def plot_paper_surface_maps(cases, out: Path, quantity: str) -> None:
    """Publication landscape: DSMC and half-range values on physical walls."""
    j, label = _quantity_style(quantity)
    by_geometry = {
        geometry: sorted((case for case in cases if case["geometry"] == geometry),
                         key=lambda case: case["kn"])
        for geometry in ("ISO", "BWD", "FWD")
    }
    if any(len(group) != 2 for group in by_geometry.values()):
        return
    ordered = [case for geometry in ("ISO", "BWD", "FWD")
               for case in by_geometry[geometry]]
    values = np.concatenate([
        np.r_[case["mean"]["target"][:, j], case["mean"]["incident"][:, j]]
        for case in ordered])
    if quantity == "cf" and values.min() < 0 < values.max():
        bound = float(np.max(np.abs(values)))
        norm, cmap = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), "coolwarm"
    else:
        lo, hi = float(values.min()), float(values.max())
        norm, cmap = Normalize(lo, hi), "viridis"

    fig, axes = plt.subplots(2, 6, figsize=(15.5, 4.35), squeeze=False)
    for group_col, geometry in enumerate(("ISO", "BWD", "FWD")):
        for row, case in enumerate(by_geometry[geometry]):
            for offset, (key, title) in enumerate((
                    ("target", "DSMC"),
                    ("incident", r"$S_{\mathrm{HR}}$"))):
                col = 2*group_col+offset
                ax = axes[row, col]
                polygon = np.vstack((case["v1"][0], case["v2"][:30][-1],
                                     case["v2"][-1]))
                polygon = (polygon-np.array([0.22, 0.0]))/0.03
                ax.fill(polygon[:, 0], polygon[:, 1], color="0.92", zorder=0)
                ax.add_collection(_surface_collection(case, case["mean"][key][:, j],
                                                      norm, cmap))
                ax.set_xlim(-.42, 1.10)
                ax.set_ylim(-.08, 1.08)
                ax.set_aspect("equal", adjustable="box")
                ax.grid(color="0.85", lw=.5, alpha=.45)
                ax.set_title(fr"{geometry}, $Kn={case['kn']:g}$: {title}",
                             fontsize=9.6, weight="bold", pad=5)
                panel = row*6+col
                ax.text(.02, .98, f"({chr(97+panel)})", transform=ax.transAxes,
                        ha="left", va="top", fontsize=9.5, weight="bold")
                if row == 1:
                    ax.set_xlabel(r"$(x-0.22)/h_p$", fontsize=9)
                else:
                    ax.set_xticklabels([])
                if col == 0:
                    ax.set_ylabel(r"$y/h_p$", fontsize=9)
                else:
                    ax.set_yticklabels([])
    fig.subplots_adjust(left=.045, right=.925, bottom=.135, top=.91,
                        wspace=.12, hspace=.10)
    colorbar = fig.colorbar(
        _surface_collection(ordered[0], ordered[0]["mean"]["incident"][:, j],
                            norm, cmap),
        ax=axes, location="right", fraction=.022, pad=.018)
    colorbar.set_label(label, fontsize=11)
    fig.text(.5, .025,
             r"Physical protrusion coordinates; $S_{\mathrm{HR}}$ uses the incoming molecular distribution",
             ha="center", va="bottom", fontsize=10)
    _save_article_figure(fig, out,
                         f"paper_{quantity}_surface_maps_concurrent_landscape")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive_root", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--additional-root", type=Path, action="append", default=[])
    ap.add_argument("--reference-root", type=Path, action="append", default=[],
                    help="Root containing CASE/output/moment_blocks.npz")
    ap.add_argument("--label", default="half_range_long")
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        raise SystemExit("label must contain only letters, digits, underscores, and hyphens")
    args.output.mkdir(parents=True, exist_ok=True)
    roots = [args.archive_root, *args.additional_root]
    case_paths = []
    for root in roots:
        case_list = root/f"{args.label}_case_list.txt"
        if case_list.exists():
            names = [x.strip() for x in case_list.read_text().splitlines() if x.strip()]
        else:
            manifest = json.loads((root/f"{args.label}_manifest.json").read_text())
            names = [entry["case_id"] for entry in manifest]
        case_paths.extend(root/name for name in names)
    seen = set()
    case_paths = [path for path in case_paths
                  if not (path.name in seen or seen.add(path.name))]
    cases = [analyze_case(path, args.label) for path in case_paths]
    cases.sort(key=lambda case: ({"ISO": 0, "BWD": 1, "FWD": 2}.get(case["geometry"], 9),
                                 case["kn"]))

    references = {}
    for root in args.reference_root:
        for path in root.rglob("moment_blocks.npz"):
            references[path.parent.parent.name] = path
    metadata = {path.name: path/"metadata.json" for path in case_paths}
    for case in cases:
        if case["case_id"] in references:
            add_independent_reference(case, references[case["case_id"]],
                                      metadata[case["case_id"]])

    plot_physical(cases, args.output, "cp")
    plot_physical(cases, args.output, "cf")
    plot_physical(cases, args.output, "cp", independent=True)
    plot_physical(cases, args.output, "cf", independent=True)
    plot_paper_profiles(cases, args.output, "cp")
    plot_paper_profiles(cases, args.output, "cf")
    plot_paper_profiles(cases, args.output, "cp", independent=True)
    plot_paper_profiles(cases, args.output, "cf", independent=True)
    plot_paper_surface_maps(cases, args.output, "cp")
    plot_paper_surface_maps(cases, args.output, "cf")
    fields = ["case_id", "geometry", "kn", "surface_id", "s_over_Lw", "x_m", "y_m", "records",
              "cp_dsmc", "cp_dsmc_sem", "cp_s3", "cp_full_control",
              "cf_dsmc", "cf_dsmc_sem", "cf_s3", "cf_full_control",
              "cp_independent", "cp_independent_sem",
              "cf_independent", "cf_independent_sem"]
    with (args.output/"half_range_long_elementwise.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for c in cases:
            for i in range(60):
                row = dict(case_id=c["case_id"], geometry=c["geometry"], kn=c["kn"],
                    surface_id=c["ids"][i],
                    s_over_Lw=c["s"][i], x_m=c["midpoint"][i,0], y_m=c["midpoint"][i,1],
                    records=c["counts"][:,i].sum(), cp_dsmc=c["mean"]["target"][i,0],
                    cp_dsmc_sem=c["sem"]["target"][i,0], cp_s3=c["mean"]["incident"][i,0],
                    cp_full_control=c["mean"]["full"][i,0], cf_dsmc=c["mean"]["target"][i,1],
                    cf_dsmc_sem=c["sem"]["target"][i,1], cf_s3=c["mean"]["incident"][i,1],
                    cf_full_control=c["mean"]["full"][i,1], cp_independent="",
                    cp_independent_sem="", cf_independent="", cf_independent_sem="")
                if "reference" in c:
                    row.update(cp_independent=c["reference"][i,0],
                               cp_independent_sem=c["reference_sem"][i,0],
                               cf_independent=c["reference"][i,1],
                               cf_independent_sem=c["reference_sem"][i,1])
                w.writerow(row)
    serial = []
    for c in cases:
        serial.append({k: v for k,v in c.items() if k not in (
            "s", "midpoint", "v1", "v2", "ids", "mean", "sem", "counts",
            "reference", "reference_sem", "reference_t95")})
    (args.output/"half_range_long_metrics.json").write_text(json.dumps(serial, indent=2)+"\n")
    lines = ["# Cross-geometry half-range reconstruction gate", "",
             "The distribution-level model is denoted S_HR, not S3: it uses pre-collision molecular velocities and the known 300 K diffuse-wall kernel. ",
             "The full-impulse control alone uses post-collision velocity. Surface-normal orientation is inferred ",
             "elementwise from incident velocities, which is required for the overhanging FWD/BWD faces.", ""]
    for c in serial:
        lines += [f"## {c['case_id']}", "",
                  f"Records used: {c['used_records']:,}; minimum per element: {c['minimum_element_records']:,}.", "",
                  "| Quantity | Concurrent S_HR NRMSE | r | Full-impulse control | DSMC RMS SEM | Independent S_HR NRMSE | Independent DSMC SEM |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for q in ("cp","cf","cq"):
            m=c["metrics"][q]
            independent_error = (f"{100*m['independent_incident_nrmse']:.2f}%"
                                 if "independent_incident_nrmse" in m else "—")
            independent_sem = (f"{m['independent_target_rms_sem_percent']:.2f}%"
                               if "independent_target_rms_sem_percent" in m else "—")
            lines.append(f"| {q.upper()} | {100*m['incident_nrmse']:.2f}% | {m['incident_correlation']:.4f} | "
                         f"{100*m['full_control_nrmse']:.3f}% | {m['target_rms_sem_percent']:.2f}% | "
                         f"{independent_error} | {independent_sem} |")
        lines.append("")
    cp_pass = all(c["metrics"]["cp"]["incident_nrmse"] < .02 for c in cases)
    cf_pass = all(c["metrics"]["cf"]["incident_nrmse"] < .15 for c in cases)
    lines += ["## Decision", "",
              f"- Cross-geometry concurrent Cp gate (<2%): {'PASS' if cp_pass else 'FAIL'}.",
              f"- Cross-geometry concurrent signed-Cf gate (<15%): {'PASS' if cf_pass else 'FAIL'}.",
              "- Independent-window FWD shear differences must be interpreted against its 9–21% DSMC block uncertainty.", ""]
    (args.output/"REPORT.md").write_text("\n".join(lines)+"\n")
    if any(c["metrics"]["cp"]["full_control_nrmse"] > .001 for c in cases):
        raise SystemExit("Full-impulse reconstruction control failed")
    print(json.dumps(serial, indent=2))

if __name__ == "__main__":
    main()
