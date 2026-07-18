#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter

DSMC = {
    "f_AF_active": 0.343,
    "mean_IAF_AF": 0.342,
    "PDelta_over_PR": 0.063,
    "mean_chiDelta": 0.097,
}


def read_case(root: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, int, float]:
    with h5py.File(root / "outdat" / "flowfield.h5", "r") as h:
        fields = {
            key: np.asarray(h[key][0], dtype=float)
            for key in ["ro", "p", "t", "u1", "u2", "qx", "qy", "Rxx", "Rxy", "Ryy", "Delta"]
        }
        nstep = int(h["nstep"][0])
        time = float(h["time"][0])
    with h5py.File(root / "datin" / "grid.h5", "r") as h:
        X = np.asarray(h["x"][0], dtype=float)
        Y = np.asarray(h["y"][0], dtype=float)
    return fields, X, Y, nstep, time


def analyze(fields: dict[str, np.ndarray], Xnative: np.ndarray, Ynative: np.ndarray) -> dict[str, object]:
    x = Xnative[0, :]
    y = Ynative[:, 0]
    n = 160
    xc = (np.arange(n) + 0.5) / n
    yc = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xc, yc)
    points = np.column_stack([Y.ravel(), X.ravel()])

    interp: dict[str, np.ndarray] = {}
    for key, array in fields.items():
        fn = RegularGridInterpolator((y, x), array, bounds_error=False, fill_value=np.nan)
        interp[key] = fn(points).reshape(n, n)

    smooth = lambda a: uniform_filter(a, size=7, mode="nearest")
    T = smooth(interp["t"])
    qx = smooth(interp["qx"])
    qy = smooth(interp["qy"])
    dTdy, dTdx = np.gradient(T, yc, xc, edge_order=2)
    qmag = np.hypot(qx, qy)
    gmag = np.hypot(dTdx, dTdy)
    den = qmag * gmag
    iaf = np.full_like(T, np.nan)
    valid = den > 1.0e-14
    iaf[valid] = (qx[valid] * dTdx[valid] + qy[valid] * dTdy[valid]) / den[valid]
    active = valid & (qmag >= 0.05 * np.nanmax(qmag)) & (gmag >= 0.05 * np.nanmax(gmag))
    af = active & (iaf > 0.0)

    Rxx = smooth(interp["Rxx"])
    Rxy = smooth(interp["Rxy"])
    Ryy = smooth(interp["Ryy"])
    Delta = smooth(interp["Delta"])
    dRxxdy, dRxxdx = np.gradient(Rxx, yc, xc, edge_order=2)
    dRxydy, dRxydx = np.gradient(Rxy, yc, xc, edge_order=2)
    dRyydy, _ = np.gradient(Ryy, yc, xc, edge_order=2)
    dDdy, dDdx = np.gradient(Delta, yc, xc, edge_order=2)
    divRx = dRxxdx + dRxydy
    divRy = dRxydx + dRyydy
    PR = np.full_like(T, np.nan)
    PD = np.full_like(T, np.nan)
    vq = qmag > 1.0e-14
    PR[vq] = (qx[vq] * divRx[vq] + qy[vq] * divRy[vq]) / qmag[vq]
    PD[vq] = (qx[vq] * dDdx[vq] + qy[vq] * dDdy[vq]) / (3.0 * qmag[vq])
    mask = af & np.isfinite(PR) & np.isfinite(PD)
    rmsPR = float(np.sqrt(np.mean(PR[mask] ** 2)))
    rmsPD = float(np.sqrt(np.mean(PD[mask] ** 2)))
    chi = np.abs(PD[mask]) / (np.abs(PR[mask]) + np.abs(PD[mask]) + 1.0e-30)

    return {
        "interp": interp,
        "X": X,
        "Y": Y,
        "iaf": iaf,
        "active": active,
        "af": af,
        "f_AF_active": float(af.sum() / active.sum()),
        "mean_IAF_AF": float(np.nanmean(iaf[af])),
        "PDelta_over_PR": float(rmsPD / rmsPR),
        "mean_chiDelta": float(np.mean(chi)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare R26, R13, and DSMC cavity metrics.")
    parser.add_argument("--r26-case", type=Path, required=True)
    parser.add_argument("--r13-case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    f26, X26, Y26, n26, t26 = read_case(args.r26_case)
    f13, X13, Y13, n13, t13 = read_case(args.r13_case)
    m26 = analyze(f26, X26, Y26)
    m13 = analyze(f13, X13, Y13)

    table = pd.DataFrame([
        {"metric": "f_AF|active", "DSMC": DSMC["f_AF_active"], "R13": m13["f_AF_active"], "R26_stage1": m26["f_AF_active"]},
        {"metric": "mean I_AF in AF set", "DSMC": DSMC["mean_IAF_AF"], "R13": m13["mean_IAF_AF"], "R26_stage1": m26["mean_IAF_AF"]},
        {"metric": "RMS P_Delta / P_R", "DSMC": DSMC["PDelta_over_PR"], "R13": m13["PDelta_over_PR"], "R26_stage1": m26["PDelta_over_PR"]},
        {"metric": "mean chi_Delta", "DSMC": DSMC["mean_chiDelta"], "R13": m13["mean_chiDelta"], "R26_stage1": m26["mean_chiDelta"]},
    ])
    table["R26_error_percent"] = (table["R26_stage1"] / table["DSMC"] - 1.0) * 100.0
    table.to_csv(args.output / "r26_r13_dsmc_metrics.csv", index=False)

    summary = {
        "R26": {"nstep": n26, "time": t26, **{key: m26[key] for key in DSMC}},
        "R13": {"nstep": n13, "time": t13, **{key: m13[key] for key in DSMC}},
        "DSMC": DSMC,
        "model_label": "Audited Maxwell/semi-linear R26 Stage 1",
    }
    (args.output / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    table.set_index("metric")[["DSMC", "R13", "R26_stage1"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("Metric value")
    ax.set_xlabel("")
    ax.set_title("DSMC versus ASTR R13 and Stage-1 R26")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(args.output / "r26_r13_dsmc_metrics.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.3))
    masked = np.ma.masked_where(~m26["active"], m26["iaf"])
    contour = ax.contourf(m26["X"], m26["Y"], masked, levels=np.linspace(-1.0, 1.0, 41), extend="both")
    ax.contour(m26["X"], m26["Y"], m26["af"].astype(float), levels=[0.5], linewidths=1.0)
    fig.colorbar(contour, ax=ax, label=r"$I_{\rm AF}$")
    ax.set_xlabel("x/L")
    ax.set_ylabel("y/L")
    ax.set_title(f"Stage-1 R26: $f_{{AF|active}}$={m26['f_AF_active']:.3f}")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(args.output / "r26_stage1_anti_fourier.png", dpi=220)
    plt.close(fig)

    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
