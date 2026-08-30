#!/usr/bin/env python3
"""Elementwise analysis of the long SPARTA incident half-range pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

K_B = 1.380649e-23
M_AR = 6.6335209e-26
T95_4DOF = 2.7764451052


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


def analyze_case(case: Path):
    meta = json.loads((case / "metadata.json").read_text())
    schema = meta["column_schema"]["collision"]
    ci = {x: i for i, x in enumerate(schema)}
    wfiles = sorted((case / "output/half_range_long").glob("wall.*.dat"))
    if len(wfiles) != 5:
        raise ValueError(f"{case.name}: expected five wall blocks")
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
    tangent[tangent[:, 0] < 0] *= -1
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    length = wall[0, :, wc["area"]]
    midpoint = (v1+v2)/2
    ds = np.linalg.norm(np.diff(midpoint, axis=0), axis=1)
    s = np.r_[0, np.cumsum(ds)]
    s /= s[-1]

    raw = snapshots(case / "output/half_range_long/collisions.dat", len(schema))
    # Step 1 is the deliberately empty initialization snapshot. The five
    # SPARTA ave/surf blocks cover 1..1000, ..., 4001..5000. Step 5001 is an
    # extra collision dump after the final wall block and is excluded.
    by_block = [[] for _ in range(5)]
    ignored = []
    for step, a in raw:
        if step == 1 or step == 5001:
            ignored.append((step, len(a)))
            continue
        b = (step-1)//1000
        if not 0 <= b < 5:
            raise ValueError(f"Unexpected step {step}")
        by_block[b].append(a)

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
        factor = fnum/(length*1000*dt)
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
    sem = {"target": target.std(0, ddof=1)/math.sqrt(5),
           "incident": incident.std(0, ddof=1)/math.sqrt(5)}
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
    return {"case_id": case.name, "kn": phys["knudsen"], "s": s, "midpoint": midpoint,
            "ids": ids, "mean": mean, "sem": sem, "counts": counts,
            "metrics": metrics, "ignored_snapshots": ignored,
            "used_records": int(counts.sum()), "minimum_element_records": int(counts.sum(0).min())}


def plot(cases, out: Path, quantity: str):
    j = {"cp": 0, "cf": 1}[quantity]
    label = {"cp": r"$C_p=(p_w-p_\infty)/(\frac{1}{2}\rho_\infty U_\infty^2)$",
             "cf": r"$C_f=\tau_w/(\frac{1}{2}\rho_\infty U_\infty^2)$"}[quantity]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.3), sharex=True)
    for ax, c in zip(axes, cases):
        s, y, p = c["s"], c["mean"]["target"][:, j], c["mean"]["incident"][:, j]
        ci = T95_4DOF*c["sem"]["target"][:, j]
        ax.fill_between(s, y-ci, y+ci, color="0.78", alpha=.55, linewidth=0, label="DSMC 95% CI")
        ax.plot(s, y, "o-", color="black", ms=3.6, lw=1.45, label="DSMC")
        ax.plot(s, p, "s-", color="#2f74b5", ms=3.2, lw=1.35, label=r"$S_3$ incident")
        ax.axvline(.5, color="0.45", ls=":", lw=1.2)
        ax.axhline(0, color="0.6", lw=.8)
        m = c["metrics"][quantity]
        ax.set_title(fr"ISO, $Ma=6$, $Kn={c['kn']:g}$   NRMSE={100*m['incident_nrmse']:.1f}%, $r$={m['incident_correlation']:.3f}", fontsize=11, weight="bold")
        ax.set_ylabel(label)
        ax.text(.21, .94, "windward face", transform=ax.transAxes, ha="center", va="top", fontsize=9)
        ax.text(.77, .94, "leeward face", transform=ax.transAxes, ha="center", va="top", fontsize=9)
        ax.grid(alpha=.22)
        ax.legend(frameon=False, fontsize=9, loc="best")
    axes[-1].set_xlabel(r"normalized physical arclength along protrusion, $s/L_w$")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out/f"half_range_long_{quantity}_profiles.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive_root", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [analyze_case(args.archive_root/x) for x in ("ISO_Ma6_Kn0p1", "ISO_Ma6_Kn0p8")]
    plot(cases, args.output, "cp")
    plot(cases, args.output, "cf")
    fields = ["case_id", "kn", "surface_id", "s_over_Lw", "x_m", "y_m", "records",
              "cp_dsmc", "cp_dsmc_sem", "cp_s3", "cp_full_control",
              "cf_dsmc", "cf_dsmc_sem", "cf_s3", "cf_full_control"]
    with (args.output/"half_range_long_elementwise.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for c in cases:
            for i in range(60):
                w.writerow(dict(case_id=c["case_id"], kn=c["kn"], surface_id=c["ids"][i],
                    s_over_Lw=c["s"][i], x_m=c["midpoint"][i,0], y_m=c["midpoint"][i,1],
                    records=c["counts"][:,i].sum(), cp_dsmc=c["mean"]["target"][i,0],
                    cp_dsmc_sem=c["sem"]["target"][i,0], cp_s3=c["mean"]["incident"][i,0],
                    cp_full_control=c["mean"]["full"][i,0], cf_dsmc=c["mean"]["target"][i,1],
                    cf_dsmc_sem=c["sem"]["target"][i,1], cf_s3=c["mean"]["incident"][i,1],
                    cf_full_control=c["mean"]["full"][i,1]))
    serial = []
    for c in cases:
        serial.append({k: v for k,v in c.items() if k not in ("s","midpoint","ids","mean","sem","counts")})
    (args.output/"half_range_long_metrics.json").write_text(json.dumps(serial, indent=2)+"\n")
    lines = ["# Long half-range gate — job 63786902", ""]
    for c in serial:
        lines += [f"## {c['case_id']}", "", f"Records used: {c['used_records']:,}; minimum per element: {c['minimum_element_records']:,}.", "",
                  "| Quantity | S3 NRMSE | r | Full-impulse control NRMSE | DSMC RMS SEM |", "|---|---:|---:|---:|---:|"]
        for q in ("cp","cf","cq"):
            m=c["metrics"][q]; lines.append(f"| {q.upper()} | {100*m['incident_nrmse']:.2f}% | {m['incident_correlation']:.4f} | {100*m['full_control_nrmse']:.3f}% | {m['target_rms_sem_percent']:.2f}% |")
        lines.append("")
    (args.output/"REPORT.md").write_text("\n".join(lines)+"\n")
    if any(c["metrics"][q]["full_control_nrmse"] > .01 for c in cases for q in ("cp","cf")):
        raise SystemExit("Full-impulse reconstruction control failed")
    print(json.dumps(serial, indent=2))

if __name__ == "__main__":
    main()
