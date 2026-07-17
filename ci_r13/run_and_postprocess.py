import gc
import json
import math
from pathlib import Path

import dolfin as df
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.ndimage import uniform_filter

from fenicsR13.input import Input
from fenicsR13.meshes import H5Mesh
from fenicsR13.solver import Solver

KB = 1.380649e-23
M_AR = 6.6335e-26
T0 = 300.0
VREF = math.sqrt(KB * T0 / M_AR)
USTAR = 100.0 / VREF
MACH = 100.0 / math.sqrt((5.0 / 3.0) * KB * T0 / M_AR)
OUT = Path("/work/output")
OUT.mkdir(parents=True, exist_ok=True)

df.set_log_level(30)
df.parameters["ghost_mode"] = "shared_vertex"
params = Input("/work/input_jfm_kn005_u100.yml").dict
summaries = []


def vertex_values(fun, mesh, ncomp=1):
    vals = fun.compute_vertex_values(mesh)
    if ncomp == 1:
        return vals
    return vals.reshape((ncomp, -1)).T


def interp_grid(coords, cells, vals, n=160):
    tri = mtri.Triangulation(coords[:, 0], coords[:, 1], cells)
    x = (np.arange(n) + 0.5) / n
    y = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, y)
    vals2 = vals[:, None] if vals.ndim == 1 else vals
    fields = []
    for j in range(vals2.shape[1]):
        z = mtri.LinearTriInterpolator(tri, vals2[:, j])(X, Y)
        fields.append(np.asarray(np.ma.filled(z, np.nan), dtype=float))
    arr = np.stack(fields, axis=-1)
    return x, y, X, Y, arr[..., 0] if vals.ndim == 1 else arr


for mesh_index, mesh_name in enumerate(params["meshes"]):
    print("RUNNING", mesh_name, flush=True)
    h5mesh = H5Mesh(mesh_name)
    solver = Solver(params, h5mesh, mesh_index)
    solver.assemble()
    solver.solve()
    solver.write()

    mesh = solver.mesh
    coords = mesh.coordinates()
    cells = mesh.cells()
    theta_v = vertex_values(solver.sol["theta"], mesh)
    q_v = vertex_values(solver.sol["s"], mesh, 2)
    u_v = vertex_values(solver.sol["u"], mesh, 2)
    p_v = vertex_values(solver.sol["p"], mesh)

    x, y, X, Y, theta = interp_grid(coords, cells, theta_v)
    _, _, _, _, q = interp_grid(coords, cells, q_v)
    _, _, _, _, u = interp_grid(coords, cells, u_v)
    _, _, _, _, p = interp_grid(coords, cells, p_v)

    theta_s = uniform_filter(theta, size=7, mode="nearest")
    qx_s = uniform_filter(q[..., 0], size=7, mode="nearest")
    qy_s = uniform_filter(q[..., 1], size=7, mode="nearest")
    dtheta_dy, dtheta_dx = np.gradient(theta_s, y, x, edge_order=2)
    qmag = np.hypot(qx_s, qy_s)
    gmag = np.hypot(dtheta_dx, dtheta_dy)
    denom = qmag * gmag
    iaf = np.full_like(theta, np.nan)
    valid = denom > 1e-14
    iaf[valid] = (
        qx_s[valid] * dtheta_dx[valid] + qy_s[valid] * dtheta_dy[valid]
    ) / denom[valid]
    active = (
        valid
        & (qmag >= 0.05 * np.nanmax(qmag))
        & (gmag >= 0.05 * np.nanmax(gmag))
    )
    anti_fourier = active & (iaf > 0.0)
    f_af = float(anti_fourier.sum() / active.sum()) if active.sum() else float("nan")
    mean_iaf_af = (
        float(np.nanmean(iaf[anti_fourier])) if anti_fourier.sum() else float("nan")
    )
    umag = np.hypot(u[..., 0], u[..., 1])

    case_dir = OUT / ("mesh_" + Path(mesh_name).stem)
    case_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        case_dir / "grid_160.npz",
        x=x,
        y=y,
        theta=theta,
        q=q,
        u=u,
        p=p,
        iaf=iaf,
        active=active,
    )
    np.savetxt(
        case_dir / "centerline_vertical.csv",
        np.c_[y, u[:, len(x) // 2, 0], u[:, len(x) // 2, 1], theta[:, len(x) // 2]],
        delimiter=",",
        header="y,u_x,u_y,theta",
        comments="",
    )
    np.savetxt(
        case_dir / "centerline_horizontal.csv",
        np.c_[x, u[len(y) // 2, :, 0], u[len(y) // 2, :, 1], theta[len(y) // 2, :]],
        delimiter=",",
        header="x,u_x,u_y,theta",
        comments="",
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    cf = ax.contourf(X, Y, umag, 40)
    ax.streamplot(x, y, u[..., 0], u[..., 1], density=1.4, linewidth=0.7)
    fig.colorbar(cf, ax=ax, label="|u| / sqrt(kT0/m)")
    ax.set(xlabel="x/L", ylabel="y/L", title="R13 velocity, " + mesh_name)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(case_dir / "velocity_streamlines.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    delta_T = (theta - 1.0) * T0
    cf = ax.contourf(X, Y, delta_T, 40)
    fig.colorbar(cf, ax=ax, label="T - 300 K (linear R13 interpretation)")
    ax.set(xlabel="x/L", ylabel="y/L", title="R13 temperature perturbation, " + mesh_name)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(case_dir / "temperature_perturbation.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    masked = np.ma.masked_where(~active, iaf)
    cf = ax.contourf(X, Y, masked, np.linspace(-1, 1, 41), extend="both")
    if np.any(anti_fourier):
        ax.contour(X, Y, anti_fourier.astype(float), levels=[0.5], colors="k", linewidths=1)
    fig.colorbar(cf, ax=ax, label="q dot grad(T) / (|q||grad(T)|)")
    ax.set(
        xlabel="x/L",
        ylabel="y/L",
        title="Anti-Fourier indicator, %s\nf_AF|active=%.3f" % (mesh_name, f_af),
    )
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(case_dir / "anti_fourier_indicator.png", dpi=180)
    plt.close(fig)

    summary = {
        "mesh": mesh_name,
        "hmax": float(h5mesh.hmax),
        "n_vertices": int(mesh.num_vertices()),
        "n_cells": int(mesh.num_cells()),
        "Kn_code": 0.05,
        "T0_K": T0,
        "lid_speed_m_s": 100.0,
        "velocity_reference_m_s": VREF,
        "lid_speed_dimensionless": USTAR,
        "Mach_gamma_5_3": MACH,
        "max_u_dimensionless_grid": float(np.nanmax(umag)),
        "max_u_m_s_grid": float(np.nanmax(umag) * VREF),
        "theta_min": float(np.nanmin(theta)),
        "theta_max": float(np.nanmax(theta)),
        "temperature_min_K": float(np.nanmin(theta) * T0),
        "temperature_max_K": float(np.nanmax(theta) * T0),
        "active_cells": int(active.sum()),
        "anti_fourier_cells": int(anti_fourier.sum()),
        "f_AF_active": f_af,
        "mean_IAF_in_AF": mean_iaf_af,
        "note": "Formal steady linear R13 solution; U=100 m/s gives Mach about 0.31, outside a conservative low-Mach quantitative range.",
    }
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    summaries.append(summary)
    solver = None
    gc.collect()

(OUT / "summary_all_meshes.json").write_text(json.dumps(summaries, indent=2))
(OUT / "RUN_NOTES.txt").write_text(
    "Nominal match to the uploaded DSMC cavity: square cavity, Ar, T0=300 K, Kn(code)=0.05, U_lid=100 m/s.\n"
    "The code velocity scale is sqrt(k_B T0/m_Ar), giving u_lid*=%.12f and Mach=%.6f.\n"
    "Walls use the repository-standard chi_tilde=1.0 and theta_w=1.0.\n"
    "This is a steady linear R13 calculation, not a quantitatively valid compressible replacement for DSMC at Mach about 0.31.\n"
    % (USTAR, MACH)
)
