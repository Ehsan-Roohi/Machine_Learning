#!/usr/bin/env python3
"""Paper-equation linear R13 reference solver for the 2-D lid cavity.

This is deliberately independent of ASTR's explicit pseudo-time integrator.
It discretizes the linearization of Rana, Torrilhon & Struchtrup (2013),
Eqs. (1)--(4), transformed closures (13), and wall conditions (7), on the
17-state ordering used in their Eq. (11).  Boundary values are eliminated by
the paper's linear extrapolation (15)--(16).  At corners, the x- and y-wall
ghost states are evaluated separately in the same residual, so neither wall
overwrites the other.

The density null mode is removed by replacing one redundant conservation row
with the discrete total-mass condition.  For a consistent linear system this
has the same physical solution as the bordered null-space construction in
Eqs. (23)--(25), while avoiding an externally transcribed left-null vector.

The module is a verification solver, not a claim that the nonlinear R13
calculation is complete.  It provides the required linear first iterate and a
test oracle before the nonlinear fixed-point/QMR path is enabled.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, qmr, spsolve

try:
    from .discrete_operator_oracle import NVAR, STATE_ORDER
except ImportError:  # Direct script execution.
    from discrete_operator_oracle import NVAR, STATE_ORDER


IDX = {name: index for index, name in enumerate(STATE_ORDER)}


@dataclass(frozen=True)
class LinearR13Config:
    nx: int = 8
    ny: int = 8
    kn: float = 0.01
    lid_velocity: float = 0.2096
    accommodation: float = 1.0
    qmr_rtol: float = 1.0e-10
    qmr_maxiter: int = 12000

    def validate(self) -> None:
        if self.nx < 3 or self.ny < 3:
            raise ValueError("nx and ny must be at least 3")
        if self.kn <= 0.0:
            raise ValueError("kn must be positive")
        if not 0.0 < self.accommodation <= 1.0:
            raise ValueError("accommodation must be in (0, 1]")
        if self.qmr_rtol <= 0.0 or self.qmr_maxiter < 1:
            raise ValueError("invalid QMR controls")

    @property
    def dx(self) -> float:
        return 1.0 / (self.nx + 1)

    @property
    def dy(self) -> float:
        return 1.0 / (self.ny + 1)


def equilibrium_perturbation(config: LinearR13Config) -> np.ndarray:
    """The Eq. (4.4 Step 1) equilibrium, expressed as perturbations."""
    config.validate()
    return np.zeros((config.ny, config.nx, NVAR), dtype=float)


def _tensor2(state: np.ndarray, xx: str, xy: str, yy: str) -> np.ndarray:
    result = np.zeros(state.shape[:-1] + (3, 3), dtype=state.dtype)
    result[..., 0, 0] = state[..., IDX[xx]]
    result[..., 0, 1] = result[..., 1, 0] = state[..., IDX[xy]]
    result[..., 1, 1] = state[..., IDX[yy]]
    result[..., 2, 2] = -result[..., 0, 0] - result[..., 1, 1]
    return result


def _tensor3(state: np.ndarray) -> np.ndarray:
    result = np.zeros(state.shape[:-1] + (3, 3, 3), dtype=state.dtype)

    def assign(value: np.ndarray, indices: tuple[int, int, int]) -> None:
        from itertools import permutations

        for permuted in set(permutations(indices)):
            result[..., permuted[0], permuted[1], permuted[2]] = value

    mxxx = state[..., IDX["m_xxx"]]
    mxyy = state[..., IDX["m_xyy"]]
    mxxy = state[..., IDX["m_xxy"]]
    myyy = state[..., IDX["m_yyy"]]
    assign(mxxx, (0, 0, 0))
    assign(mxxy, (0, 0, 1))
    assign(mxyy, (0, 1, 1))
    assign(myyy, (1, 1, 1))
    assign(-mxxx - mxyy, (0, 2, 2))
    assign(-mxxy - myyy, (1, 2, 2))
    return result


def _stf2(raw: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (raw + np.swapaxes(raw, -1, -2))
    trace = np.trace(symmetric, axis1=-2, axis2=-1) / 3.0
    result = symmetric.copy()
    for axis in range(3):
        result[..., axis, axis] -= trace
    return result


def _stf3_from_symmetric2_gradient(raw: np.ndarray) -> np.ndarray:
    """STF over all three indices of d_k S_ij, with S_ij symmetric."""
    symmetric = (
        raw
        + np.transpose(raw, (*range(raw.ndim - 3), -3, -1, -2))
        + np.transpose(raw, (*range(raw.ndim - 3), -1, -2, -3))
    ) / 3.0
    trace = np.einsum("...llk->...k", symmetric)
    result = symmetric.copy()
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i == j:
                    result[..., i, j, k] -= trace[..., k] / 5.0
                if i == k:
                    result[..., i, j, k] -= trace[..., j] / 5.0
                if j == k:
                    result[..., i, j, k] -= trace[..., i] / 5.0
    return result


def linear_wall_state(
    extrapolated: np.ndarray,
    *,
    axis: str,
    normal_sign: int,
    wall_tangential_velocity: float,
    accommodation: float,
) -> np.ndarray:
    """Linearized Rana Eqs. (7a)--(7f) for one oriented wall.

    ``extrapolated`` is ``2 U_1 - U_2`` from Eqs. (15)--(16).  The six
    incoming/controlled moments are replaced; the other eleven rows are the
    identity rows described immediately after Eq. (16).
    """
    if axis not in ("x", "y"):
        raise ValueError("axis must be x or y")
    if normal_sign not in (-1, 1):
        raise ValueError("normal_sign must be -1 or +1")
    if not 0.0 < accommodation <= 1.0:
        raise ValueError("accommodation must be in (0, 1]")
    state = np.asarray(extrapolated, dtype=float).copy()
    if state.shape[-1] != NVAR:
        raise ValueError(f"last dimension must be {NVAR}")
    coefficient = (
        accommodation / (2.0 - accommodation) * np.sqrt(2.0 / np.pi)
    )
    sign = float(normal_sign)

    if axis == "x":
        vs = state[..., IDX["vy"]] - wall_tangential_velocity
        qs = state[..., IDX["qy"]]
        snn = state[..., IDX["sigma_xx"]]
        sss = state[..., IDX["sigma_yy"]]
        rnn = state[..., IDX["R_xx"]]
        rss = state[..., IDX["R_yy"]]
        msnn = state[..., IDX["m_xxy"]]
    else:
        vs = state[..., IDX["vx"]] - wall_tangential_velocity
        qs = state[..., IDX["qx"]]
        snn = state[..., IDX["sigma_yy"]]
        sss = state[..., IDX["sigma_xx"]]
        rnn = state[..., IDX["R_yy"]]
        rss = state[..., IDX["R_xx"]]
        msnn = state[..., IDX["m_xyy"]]

    temperature_jump = state[..., IDX["theta"]]
    delta = state[..., IDX["Delta"]]
    sigma_sn = -coefficient * (vs + qs / 5.0 + msnn / 2.0)
    qn = -coefficient * (
        2.0 * temperature_jump
        + snn / 2.0
        + delta / 15.0
        + 5.0 * rnn / 28.0
    )
    r_sn = coefficient * (vs - 11.0 * qs / 5.0 - msnn / 2.0)
    m_nnn = coefficient * (
        2.0 * temperature_jump / 5.0
        - 7.0 * snn / 5.0
        + delta / 75.0
        - rnn / 14.0
    )
    m_ssn = -coefficient * (
        temperature_jump / 5.0
        + rss / 14.0
        + sss
        - snn / 5.0
        + delta / 150.0
    )

    if axis == "x":
        state[..., IDX["vx"]] = 0.0
        state[..., IDX["qx"]] = sign * qn
        state[..., IDX["sigma_xy"]] = sign * sigma_sn
        state[..., IDX["R_xy"]] = sign * r_sn
        state[..., IDX["m_xxx"]] = sign * m_nnn
        state[..., IDX["m_xyy"]] = sign * m_ssn
    else:
        state[..., IDX["vy"]] = 0.0
        state[..., IDX["qy"]] = sign * qn
        state[..., IDX["sigma_xy"]] = sign * sigma_sn
        state[..., IDX["R_xy"]] = sign * r_sn
        state[..., IDX["m_yyy"]] = sign * m_nnn
        state[..., IDX["m_xxy"]] = sign * m_ssn
    return state


def ghosted_state(state: np.ndarray, config: LinearR13Config) -> np.ndarray:
    """Eliminate all four walls without a sequential corner overwrite."""
    config.validate()
    state = np.asarray(state, dtype=float)
    if state.shape != (config.ny, config.nx, NVAR):
        raise ValueError("state shape does not match configuration")
    extended = np.zeros((config.ny + 2, config.nx + 2, NVAR), dtype=float)
    extended[1:-1, 1:-1] = state
    extended[1:-1, 0] = linear_wall_state(
        2.0 * state[:, 0] - state[:, 1],
        axis="x",
        normal_sign=1,
        wall_tangential_velocity=0.0,
        accommodation=config.accommodation,
    )
    extended[1:-1, -1] = linear_wall_state(
        2.0 * state[:, -1] - state[:, -2],
        axis="x",
        normal_sign=-1,
        wall_tangential_velocity=0.0,
        accommodation=config.accommodation,
    )
    extended[0, 1:-1] = linear_wall_state(
        2.0 * state[0] - state[1],
        axis="y",
        normal_sign=1,
        wall_tangential_velocity=0.0,
        accommodation=config.accommodation,
    )
    extended[-1, 1:-1] = linear_wall_state(
        2.0 * state[-1] - state[-2],
        axis="y",
        normal_sign=-1,
        wall_tangential_velocity=config.lid_velocity,
        accommodation=config.accommodation,
    )
    return extended


def linear_residual(state: np.ndarray, config: LinearR13Config) -> np.ndarray:
    """Central-difference residual of the complete linear 17-state system."""
    extended = ghosted_state(state, config)
    center = extended[1:-1, 1:-1]
    dx = (extended[1:-1, 2:] - extended[1:-1, :-2]) / (2.0 * config.dx)
    dy = (extended[2:, 1:-1] - extended[:-2, 1:-1]) / (2.0 * config.dy)
    derivative = np.zeros(center.shape[:-1] + (NVAR, 3), dtype=float)
    derivative[..., :, 0] = dx
    derivative[..., :, 1] = dy

    velocity_gradient = np.zeros(center.shape[:-1] + (3, 3), dtype=float)
    heat_gradient = np.zeros_like(velocity_gradient)
    velocity_gradient[..., 0, :2] = derivative[..., IDX["vx"], :2]
    velocity_gradient[..., 1, :2] = derivative[..., IDX["vy"], :2]
    heat_gradient[..., 0, :2] = derivative[..., IDX["qx"], :2]
    heat_gradient[..., 1, :2] = derivative[..., IDX["qy"], :2]
    stf_velocity_gradient = _stf2(velocity_gradient)
    stf_heat_gradient = _stf2(heat_gradient)

    stress = _tensor2(center, "sigma_xx", "sigma_xy", "sigma_yy")
    regularized_stress = _tensor2(center, "R_xx", "R_xy", "R_yy")
    third_moment = _tensor3(center)
    stress_gradient = np.zeros(center.shape[:-1] + (3, 3, 3), dtype=float)
    for derivative_axis, values in ((0, dx), (1, dy)):
        stress_gradient[..., 0, 0, derivative_axis] = values[..., IDX["sigma_xx"]]
        stress_gradient[..., 0, 1, derivative_axis] = values[..., IDX["sigma_xy"]]
        stress_gradient[..., 1, 0, derivative_axis] = values[..., IDX["sigma_xy"]]
        stress_gradient[..., 1, 1, derivative_axis] = values[..., IDX["sigma_yy"]]
        stress_gradient[..., 2, 2, derivative_axis] = -(
            values[..., IDX["sigma_xx"]] + values[..., IDX["sigma_yy"]]
        )
    stf_stress_gradient = _stf3_from_symmetric2_gradient(stress_gradient)

    divergence_velocity = (
        derivative[..., IDX["vx"], 0] + derivative[..., IDX["vy"], 1]
    )
    divergence_heat = (
        derivative[..., IDX["qx"], 0] + derivative[..., IDX["qy"], 1]
    )
    divergence_stress = np.zeros(center.shape[:-1] + (3,), dtype=float)
    divergence_r = np.zeros_like(divergence_stress)
    divergence_m = np.zeros(center.shape[:-1] + (3, 3), dtype=float)
    for i in range(3):
        divergence_stress[..., i] = (
            stress_gradient[..., i, 0, 0] + stress_gradient[..., i, 1, 1]
        )
        divergence_r[..., i] = (
            derivative[..., IDX["R_xx"], 0]
            if i == 0
            else derivative[..., IDX["R_xy"], 0]
            if i == 1
            else 0.0
        )
        if i == 0:
            divergence_r[..., i] += derivative[..., IDX["R_xy"], 1]
        elif i == 1:
            divergence_r[..., i] += derivative[..., IDX["R_yy"], 1]
        for j in range(3):
            divergence_m[..., i, j] = (
                (third_moment[..., i, j, 0] * 0.0)
            )
    # Derivatives of the four independent m-components.
    divergence_m[..., 0, 0] = (
        derivative[..., IDX["m_xxx"], 0] + derivative[..., IDX["m_xxy"], 1]
    )
    divergence_m[..., 0, 1] = divergence_m[..., 1, 0] = (
        derivative[..., IDX["m_xxy"], 0] + derivative[..., IDX["m_xyy"], 1]
    )
    divergence_m[..., 1, 1] = (
        derivative[..., IDX["m_xyy"], 0] + derivative[..., IDX["m_yyy"], 1]
    )

    residual = np.zeros_like(center)
    pressure_perturbation = center[..., IDX["rho"]] + center[..., IDX["theta"]]
    residual[..., IDX["rho"]] = divergence_velocity
    residual[..., IDX["vx"]] = (
        derivative[..., IDX["rho"], 0]
        + derivative[..., IDX["theta"], 0]
        + divergence_stress[..., 0]
    )
    residual[..., IDX["vy"]] = (
        derivative[..., IDX["rho"], 1]
        + derivative[..., IDX["theta"], 1]
        + divergence_stress[..., 1]
    )
    residual[..., IDX["theta"]] = 2.5 * divergence_velocity + divergence_heat

    heat = np.stack(
        (center[..., IDX["qx"]], center[..., IDX["qy"]], np.zeros_like(pressure_perturbation)),
        axis=-1,
    )
    delta_gradient = np.zeros_like(heat)
    delta_gradient[..., :2] = derivative[..., IDX["Delta"], :2]
    temperature_gradient = np.zeros_like(heat)
    temperature_gradient[..., :2] = derivative[..., IDX["theta"], :2]
    heat_balance = (
        divergence_stress
        + 0.5 * divergence_r
        + delta_gradient / 6.0
        + 2.5 * temperature_gradient
        + (2.0 / (3.0 * config.kn)) * heat
    )
    residual[..., IDX["qx"]] = heat_balance[..., 0]
    residual[..., IDX["qy"]] = heat_balance[..., 1]

    stress_balance = (
        divergence_m
        + 0.8 * stf_heat_gradient
        + 2.0 * stf_velocity_gradient
        + stress / config.kn
    )
    residual[..., IDX["sigma_xx"]] = stress_balance[..., 0, 0]
    residual[..., IDX["sigma_xy"]] = stress_balance[..., 0, 1]
    residual[..., IDX["sigma_yy"]] = stress_balance[..., 1, 1]

    r_closure = regularized_stress + (24.0 / 5.0) * config.kn * stf_heat_gradient
    residual[..., IDX["R_xx"]] = r_closure[..., 0, 0]
    residual[..., IDX["R_xy"]] = r_closure[..., 0, 1]
    residual[..., IDX["R_yy"]] = r_closure[..., 1, 1]

    m_closure = third_moment + 2.0 * config.kn * stf_stress_gradient
    residual[..., IDX["m_xxx"]] = m_closure[..., 0, 0, 0]
    residual[..., IDX["m_xyy"]] = m_closure[..., 0, 1, 1]
    residual[..., IDX["m_xxy"]] = m_closure[..., 0, 0, 1]
    residual[..., IDX["m_yyy"]] = m_closure[..., 1, 1, 1]
    residual[..., IDX["Delta"]] = (
        center[..., IDX["Delta"]] + 12.0 * config.kn * divergence_heat
    )
    return residual


def _flat_index(config: LinearR13Config, j: int, i: int, variable: int) -> int:
    return (j * config.nx + i) * NVAR + variable


def assemble_colored_operator(
    config: LinearR13Config, *, drop_tolerance: float = 1.0e-13
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Assemble the exact local linear operator in 9 graph-colored batches."""
    config.validate()
    zero = equilibrium_perturbation(config)
    affine = linear_residual(zero, config).reshape(-1)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for variable in range(NVAR):
        for color_j in range(3):
            for color_i in range(3):
                probe = np.zeros_like(zero)
                nodes: list[tuple[int, int]] = []
                for j in range(color_j, config.ny, 3):
                    for i in range(color_i, config.nx, 3):
                        probe[j, i, variable] = 1.0
                        nodes.append((j, i))
                if not nodes:
                    continue
                response = (linear_residual(probe, config).reshape(-1) - affine)
                response = response.reshape(config.ny, config.nx, NVAR)
                for j, i in nodes:
                    column = _flat_index(config, j, i, variable)
                    affected = {(j, i)}
                    if i > 0:
                        affected.add((j, i - 1))
                    if i + 1 < config.nx:
                        affected.add((j, i + 1))
                    if j > 0:
                        affected.add((j - 1, i))
                    if j + 1 < config.ny:
                        affected.add((j + 1, i))
                    for out_j, out_i in affected:
                        values = response[out_j, out_i]
                        for out_variable in np.flatnonzero(np.abs(values) > drop_tolerance):
                            rows.append(_flat_index(config, out_j, out_i, int(out_variable)))
                            cols.append(column)
                            data.append(float(values[out_variable]))
    size = config.nx * config.ny * NVAR
    operator = sparse.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()
    operator.sum_duplicates()
    return operator, affine


def impose_total_mass(
    operator: sparse.csr_matrix, rhs: np.ndarray, config: LinearR13Config
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Replace one redundant mass row with mean(rho-rho0)=0."""
    modified = operator.tolil(copy=True)
    modified[0, :] = 0.0
    rho_columns = [
        _flat_index(config, j, i, IDX["rho"])
        for j in range(config.ny)
        for i in range(config.nx)
    ]
    modified[0, rho_columns] = 1.0 / (config.nx * config.ny)
    modified_rhs = np.asarray(rhs, dtype=float).copy()
    modified_rhs[0] = 0.0
    return modified.tocsr(), modified_rhs


def solve_linear_r13(
    config: LinearR13Config,
    *,
    direct: bool = False,
    callback: Callable[[float], None] | None = None,
) -> tuple[np.ndarray, dict[str, float | int | str | bool]]:
    """Assemble and solve the mass-constrained linear R13 cavity system."""
    operator, affine = assemble_colored_operator(config)
    rhs = -affine
    operator, rhs = impose_total_mass(operator, rhs, config)
    method = "spsolve" if direct else "qmr"
    iterations = 0
    history: list[float] = []

    if direct:
        solution = spsolve(operator, rhs)
        info = 0
    else:
        identity = LinearOperator(
            operator.shape,
            matvec=lambda value: value,
            rmatvec=lambda value: value,
            dtype=float,
        )

        def record(vector: np.ndarray) -> None:
            nonlocal iterations
            iterations += 1
            relative = float(
                np.linalg.norm(operator @ vector - rhs)
                / max(np.linalg.norm(rhs), np.finfo(float).tiny)
            )
            history.append(relative)
            if callback is not None:
                callback(relative)

        solution, info = qmr(
            operator,
            rhs,
            rtol=config.qmr_rtol,
            atol=0.0,
            maxiter=config.qmr_maxiter,
            M1=identity,
            M2=identity,
            callback=record,
        )
    residual = operator @ solution - rhs
    relative_residual = float(
        np.linalg.norm(residual) / max(np.linalg.norm(rhs), np.finfo(float).tiny)
    )
    if info != 0 or not np.isfinite(solution).all():
        raise RuntimeError(f"{method} failed: info={info}, residual={relative_residual:.3e}")
    state = solution.reshape(config.ny, config.nx, NVAR)
    diagnostics: dict[str, float | int | str | bool] = {
        "method": method,
        "iterations": iterations,
        "qmr_info": int(info),
        "relative_linear_residual": relative_residual,
        "operator_rows": int(operator.shape[0]),
        "operator_nnz": int(operator.nnz),
        "finite": bool(np.isfinite(state).all()),
        "mass_perturbation": float(np.mean(state[..., IDX["rho"]])),
    }
    if history:
        diagnostics["last_callback_residual"] = history[-1]
    return state, diagnostics


def cavity_metrics(state: np.ndarray, config: LinearR13Config) -> dict[str, float]:
    """Compute paper Eq. (30) diagnostics on the eliminated wall states."""
    extended = ghosted_state(state, config)
    top = extended[-1, 1:-1]
    # The state stress is sigma_xy/p0.  Rana/Sharipov's low-speed drag
    # coefficient is sqrt(2)*(sigma_xy/p0)/U*, with
    # U*=U_lid/sqrt(theta0).  Keep the
    # pre-reduction integral explicit so a normalization error cannot masquerade
    # as agreement with the paper.
    sigma_over_p0_signed = float(
        config.dx * np.sum(top[..., IDX["sigma_xy"]])
    )
    reduction_factor = np.sqrt(2.0) / abs(config.lid_velocity)
    signed_d = reduction_factor * sigma_over_p0_signed
    x_coordinates = np.arange(1, config.nx + 1, dtype=float) * config.dx
    center_x = 0.5
    vx = state[..., IDX["vx"]]
    center_velocity = np.asarray(
        [np.interp(center_x, x_coordinates, row) for row in vx], dtype=float
    )
    g_value = float(
        config.dy * np.sum(np.abs(center_velocity)) / abs(config.lid_velocity)
    )
    return {
        "D_signed": signed_d,
        "D": abs(signed_d),
        "D_sigma_over_p0_signed": sigma_over_p0_signed,
        "D_reduced_stress_factor": reduction_factor,
        "G": g_value,
        "rho_min": float(np.min(1.0 + state[..., IDX["rho"]])),
        "theta_min": float(np.min(1.0 + state[..., IDX["theta"]])),
        "state_max_abs": float(np.max(np.abs(state))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--kn", type=float, default=0.01)
    parser.add_argument("--lid-velocity", type=float, default=0.2096)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = LinearR13Config(
        nx=args.nx,
        ny=args.ny,
        kn=args.kn,
        lid_velocity=args.lid_velocity,
    )
    state, solver = solve_linear_r13(config, direct=args.direct)
    report = {
        "model": "linearized transformed R13",
        "paper": "Rana, Torrilhon & Struchtrup, JCP 236 (2013), Eqs. 1-4, 7, 13-25",
        "configuration": config.__dict__,
        "solver": solver,
        "metrics": cavity_metrics(state, config),
        "scientific_status": "verification-only; nonlinear and grid-convergence gates remain",
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
        np.save(args.output.with_suffix(".npy"), state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
