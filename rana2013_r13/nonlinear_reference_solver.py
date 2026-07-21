#!/usr/bin/env python3
"""Nonlinear transformed-R13 residual and Newton--QMR verification solver.

The equations are evaluated directly in tensor form from Rana, Torrilhon &
Struchtrup (2013), Eqs. (1)--(4), (7), and transformed closures (13).  This
avoids transcription of the image-only Appendix-A matrices while producing
the same quasilinear first-order system.  The discretization follows
Eqs. (15)--(20): centered differences, linearly extrapolated boundary states,
and simultaneous x/y ghost contributions at corners.

The nonlinear solve uses a graph-colored finite-difference Jacobian and QMR
steps with a residual-decreasing line search.  A directional-derivative test
against ``linear_reference_solver`` guards every coefficient and wall sign at
global equilibrium before production-sized runs may be enabled.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, qmr

try:
    from .discrete_operator_oracle import NVAR, STATE_ORDER, relative_l1_change
    from .linear_reference_solver import (
        IDX,
        LinearR13Config,
        _flat_index,
        _stf2,
        _stf3_from_symmetric2_gradient,
        _tensor2,
        _tensor3,
    )
except ImportError:
    from discrete_operator_oracle import NVAR, STATE_ORDER, relative_l1_change
    from linear_reference_solver import (
        IDX,
        LinearR13Config,
        _flat_index,
        _stf2,
        _stf3_from_symmetric2_gradient,
        _tensor2,
        _tensor3,
    )


@dataclass(frozen=True)
class NonlinearR13Config(LinearR13Config):
    nonlinear_rtol: float = 1.0e-8
    nonlinear_update_tol: float = 1.0e-7
    nonlinear_maxiter: int = 20
    jacobian_step: float = 2.0e-7
    minimum_line_step: float = 1.0 / 128.0

    def validate(self) -> None:
        super().validate()
        if self.nonlinear_rtol <= 0.0 or self.nonlinear_update_tol <= 0.0:
            raise ValueError("nonlinear tolerances must be positive")
        if self.nonlinear_maxiter < 1 or self.jacobian_step <= 0.0:
            raise ValueError("invalid nonlinear iteration controls")
        if not 0.0 < self.minimum_line_step <= 1.0:
            raise ValueError("minimum_line_step must be in (0, 1]")


def _stf3(raw: np.ndarray) -> np.ndarray:
    """Symmetric trace-free part of an arbitrary rank-three tensor."""
    from itertools import permutations

    symmetric = np.zeros_like(raw)
    for permutation in permutations((0, 1, 2)):
        axes = (*range(raw.ndim - 3), *(raw.ndim - 3 + p for p in permutation))
        symmetric += np.transpose(raw, axes)
    symmetric /= 6.0
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


def nonlinear_wall_state(
    extrapolated: np.ndarray,
    *,
    axis: str,
    normal_sign: int,
    wall_tangential_velocity: float,
    accommodation: float,
) -> np.ndarray:
    """Exact nonlinear Eq. (7) map for the six controlled wall moments."""
    if axis not in ("x", "y"):
        raise ValueError("axis must be x or y")
    if normal_sign not in (-1, 1):
        raise ValueError("normal_sign must be -1 or +1")
    if not 0.0 < accommodation <= 1.0:
        raise ValueError("accommodation must be in (0, 1]")
    state = np.asarray(extrapolated, dtype=float).copy()
    rho = 1.0 + state[..., IDX["rho"]]
    theta = 1.0 + state[..., IDX["theta"]]
    if np.any(rho <= 0.0) or np.any(theta <= 0.0):
        raise FloatingPointError("non-positive extrapolated rho/theta at wall")
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
    effective_pressure = (
        rho * theta
        + 0.5 * sss
        - delta / (120.0 * theta)
        - rss / (28.0 * theta)
    )
    if np.any(~np.isfinite(effective_pressure)) or np.any(effective_pressure <= 0.0):
        raise FloatingPointError("non-positive Rana Eq. (7) effective wall pressure")
    coefficient = (
        accommodation
        / (2.0 - accommodation)
        * np.sqrt(2.0 / (np.pi * theta))
    )
    sigma_sn = -coefficient * (
        effective_pressure * vs + qs / 5.0 + msnn / 2.0
    )
    qn = -coefficient * (
        2.0 * effective_pressure * temperature_jump
        - 0.5 * effective_pressure * vs**2
        + 0.5 * theta * snn
        + delta / 15.0
        + 5.0 * rnn / 28.0
    )
    r_sn = coefficient * (
        6.0 * effective_pressure * temperature_jump * vs
        + effective_pressure * theta * vs
        - effective_pressure * vs**3
        - 11.0 * theta * qs / 5.0
        - 0.5 * theta * msnn
    )
    m_nnn = coefficient * (
        2.0 * effective_pressure * temperature_jump / 5.0
        - 3.0 * effective_pressure * vs**2 / 5.0
        - 7.0 * theta * snn / 5.0
        + delta / 75.0
        - rnn / 14.0
    )
    m_ssn = -coefficient * (
        effective_pressure * temperature_jump / 5.0
        - 4.0 * effective_pressure * vs**2 / 5.0
        + rss / 14.0
        + theta * sss
        - theta * snn / 5.0
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


def nonlinear_ghosted_state(
    state: np.ndarray, config: NonlinearR13Config
) -> np.ndarray:
    config.validate()
    state = np.asarray(state, dtype=float)
    if state.shape != (config.ny, config.nx, NVAR):
        raise ValueError("state shape does not match configuration")
    extended = np.zeros((config.ny + 2, config.nx + 2, NVAR), dtype=float)
    extended[1:-1, 1:-1] = state
    extended[1:-1, 0] = nonlinear_wall_state(
        2.0 * state[:, 0] - state[:, 1],
        axis="x",
        normal_sign=1,
        wall_tangential_velocity=0.0,
        accommodation=config.accommodation,
    )
    extended[1:-1, -1] = nonlinear_wall_state(
        2.0 * state[:, -1] - state[:, -2],
        axis="x",
        normal_sign=-1,
        wall_tangential_velocity=0.0,
        accommodation=config.accommodation,
    )
    extended[0, 1:-1] = nonlinear_wall_state(
        2.0 * state[0] - state[1],
        axis="y",
        normal_sign=1,
        wall_tangential_velocity=0.0,
        accommodation=config.accommodation,
    )
    extended[-1, 1:-1] = nonlinear_wall_state(
        2.0 * state[-1] - state[-2],
        axis="y",
        normal_sign=-1,
        wall_tangential_velocity=config.lid_velocity,
        accommodation=config.accommodation,
    )
    return extended


def nonlinear_residual(state: np.ndarray, config: NonlinearR13Config) -> np.ndarray:
    """Full steady transformed-R13 residual in primitive 17-state form."""
    extended = nonlinear_ghosted_state(state, config)
    center = extended[1:-1, 1:-1]
    dx = (extended[1:-1, 2:] - extended[1:-1, :-2]) / (2.0 * config.dx)
    dy = (extended[2:, 1:-1] - extended[:-2, 1:-1]) / (2.0 * config.dy)
    derivative = np.zeros(center.shape[:-1] + (NVAR, 3), dtype=float)
    derivative[..., :, 0] = dx
    derivative[..., :, 1] = dy

    rho = 1.0 + center[..., IDX["rho"]]
    theta = 1.0 + center[..., IDX["theta"]]
    if np.any(rho <= 0.0) or np.any(theta <= 0.0):
        raise FloatingPointError("non-positive interior rho/theta")
    pressure = rho * theta
    mu = config.kn * theta  # Maxwell molecules: mu/mu0 = theta/theta0.
    velocity = np.stack(
        (center[..., IDX["vx"]], center[..., IDX["vy"]], np.zeros_like(rho)),
        axis=-1,
    )
    heat = np.stack(
        (center[..., IDX["qx"]], center[..., IDX["qy"]], np.zeros_like(rho)),
        axis=-1,
    )
    velocity_gradient = np.zeros(center.shape[:-1] + (3, 3), dtype=float)
    heat_gradient = np.zeros_like(velocity_gradient)
    velocity_gradient[..., 0, :2] = derivative[..., IDX["vx"], :2]
    velocity_gradient[..., 1, :2] = derivative[..., IDX["vy"], :2]
    heat_gradient[..., 0, :2] = derivative[..., IDX["qx"], :2]
    heat_gradient[..., 1, :2] = derivative[..., IDX["qy"], :2]
    stf_velocity_gradient = _stf2(velocity_gradient)
    stf_heat_gradient = _stf2(heat_gradient)
    divergence_velocity = np.trace(velocity_gradient, axis1=-2, axis2=-1)
    divergence_heat = np.trace(heat_gradient, axis1=-2, axis2=-1)

    stress = _tensor2(center, "sigma_xx", "sigma_xy", "sigma_yy")
    regularized_stress = _tensor2(center, "R_xx", "R_xy", "R_yy")
    third_moment = _tensor3(center)
    stress_gradient = np.zeros(center.shape[:-1] + (3, 3, 3), dtype=float)
    r_gradient = np.zeros_like(stress_gradient)
    for derivative_axis, values in ((0, dx), (1, dy)):
        for tensor_gradient, xx, xy, yy in (
            (stress_gradient, "sigma_xx", "sigma_xy", "sigma_yy"),
            (r_gradient, "R_xx", "R_xy", "R_yy"),
        ):
            tensor_gradient[..., 0, 0, derivative_axis] = values[..., IDX[xx]]
            tensor_gradient[..., 0, 1, derivative_axis] = values[..., IDX[xy]]
            tensor_gradient[..., 1, 0, derivative_axis] = values[..., IDX[xy]]
            tensor_gradient[..., 1, 1, derivative_axis] = values[..., IDX[yy]]
            tensor_gradient[..., 2, 2, derivative_axis] = -(
                values[..., IDX[xx]] + values[..., IDX[yy]]
            )
    divergence_stress = np.einsum("...ikk->...i", stress_gradient)
    divergence_r = np.einsum("...ikk->...i", r_gradient)
    stf_stress_gradient = _stf3_from_symmetric2_gradient(stress_gradient)
    divergence_m = np.zeros(center.shape[:-1] + (3, 3), dtype=float)
    divergence_m[..., 0, 0] = (
        derivative[..., IDX["m_xxx"], 0] + derivative[..., IDX["m_xxy"], 1]
    )
    divergence_m[..., 0, 1] = divergence_m[..., 1, 0] = (
        derivative[..., IDX["m_xxy"], 0] + derivative[..., IDX["m_xyy"], 1]
    )
    divergence_m[..., 1, 1] = (
        derivative[..., IDX["m_xyy"], 0] + derivative[..., IDX["m_yyy"], 1]
    )
    divergence_m[..., 2, 2] = -divergence_m[..., 0, 0] - divergence_m[..., 1, 1]

    rho_gradient = np.zeros_like(velocity)
    theta_gradient = np.zeros_like(velocity)
    delta_gradient = np.zeros_like(velocity)
    rho_gradient[..., :2] = derivative[..., IDX["rho"], :2]
    theta_gradient[..., :2] = derivative[..., IDX["theta"], :2]
    delta_gradient[..., :2] = derivative[..., IDX["Delta"], :2]
    pressure_gradient = theta[..., None] * rho_gradient + rho[..., None] * theta_gradient
    grad_log_pressure = pressure_gradient / pressure[..., None]
    convective_derivative = np.einsum("...k,...ak->...a", velocity, derivative)

    residual = np.zeros_like(center)
    residual[..., IDX["rho"]] = (
        np.einsum("...k,...k->...", velocity, rho_gradient)
        + rho * divergence_velocity
    )
    momentum = (
        pressure_gradient
        + divergence_stress
        + velocity * np.einsum("...k,...k->...", velocity, rho_gradient)[..., None]
        + rho[..., None] * np.einsum("...k,...ik->...i", velocity, velocity_gradient)
        + rho[..., None] * velocity * divergence_velocity[..., None]
    )
    residual[..., IDX["vx"]] = momentum[..., 0]
    residual[..., IDX["vy"]] = momentum[..., 1]

    speed2 = np.einsum("...i,...i->...", velocity, velocity)
    energy = 1.5 * theta + 0.5 * speed2
    energy_gradient = (
        energy[..., None] * rho_gradient
        + rho[..., None]
        * (
            1.5 * theta_gradient
            + np.einsum("...i,...ik->...k", velocity, velocity_gradient)
        )
    )
    total_energy_balance = (
        np.einsum("...k,...k->...", velocity, energy_gradient)
        + rho * energy * divergence_velocity
        + np.einsum("...k,...k->...", velocity, pressure_gradient)
        + pressure * divergence_velocity
        + np.einsum("...i,...i->...", velocity, divergence_stress)
        + np.einsum("...ik,...ik->...", stress, velocity_gradient)
        + divergence_heat
    )
    residual[..., IDX["theta"]] = total_energy_balance

    heat_balance = (
        np.einsum("...k,...ik->...i", velocity, heat_gradient)
        + (7.0 / 5.0) * np.einsum("...k,...ik->...i", heat, velocity_gradient)
        - (theta / rho)[..., None] * np.einsum("...ik,...k->...i", stress, rho_gradient)
        - np.einsum("...kl,...ikl->...i", stress, stress_gradient) / rho[..., None]
        + 2.5 * np.einsum("...ik,...k->...i", stress, theta_gradient)
        + theta[..., None] * divergence_stress
        + (7.0 / 5.0) * heat * divergence_velocity[..., None]
        + (2.0 / 5.0) * np.einsum("...k,...ki->...i", heat, velocity_gradient)
        + 0.5 * divergence_r
        + delta_gradient / 6.0
        + np.einsum("...ikl,...kl->...i", third_moment, velocity_gradient)
        + 2.5 * pressure[..., None] * theta_gradient
        + (2.0 * pressure / (3.0 * mu))[..., None] * heat
    )
    residual[..., IDX["qx"]] = heat_balance[..., 0]
    residual[..., IDX["qy"]] = heat_balance[..., 1]

    # Rana Eq. (4): sigma_{k<i} d_k v_{j>}.  The derivative index is k;
    # using d_j v_k here is a different tensor and destabilizes the branch.
    stress_velocity_raw = np.einsum("...ki,...jk->...ij", stress, velocity_gradient)
    stress_balance = (
        np.einsum("...k,...ijk->...ij", velocity, stress_gradient)
        + divergence_m
        + 0.8 * stf_heat_gradient
        + 2.0 * _stf2(stress_velocity_raw)
        + stress * divergence_velocity[..., None, None]
        + 2.0 * pressure[..., None, None] * stf_velocity_gradient
        + (pressure / mu)[..., None, None] * stress
    )
    residual[..., IDX["sigma_xx"]] = stress_balance[..., 0, 0]
    residual[..., IDX["sigma_xy"]] = stress_balance[..., 0, 1]
    residual[..., IDX["sigma_yy"]] = stress_balance[..., 1, 1]

    sigma_squared_stf = _stf2(np.einsum("...ki,...jk->...ij", stress, stress))
    heat_squared_stf = _stf2(heat[..., :, None] * heat[..., None, :])
    quotient_heat_gradient = _stf2(
        heat_gradient - heat[..., :, None] * grad_log_pressure[..., None, :]
    )
    r_target = (
        (20.0 / 7.0) * sigma_squared_stf / rho[..., None, None]
        + (64.0 / 25.0) * heat_squared_stf / pressure[..., None, None]
        - (24.0 / 5.0)
        * (mu * theta / pressure)[..., None, None]
        * quotient_heat_gradient
    )
    r_closure = regularized_stress - r_target
    residual[..., IDX["R_xx"]] = r_closure[..., 0, 0]
    residual[..., IDX["R_xy"]] = r_closure[..., 0, 1]
    residual[..., IDX["R_yy"]] = r_closure[..., 1, 1]

    q_sigma_stf = _stf3(heat[..., :, None, None] * stress[..., None, :, :])
    sigma_logp_stf = _stf3_from_symmetric2_gradient(
        stress[..., :, :, None] * grad_log_pressure[..., None, None, :]
    )
    m_target = (
        (4.0 / 3.0) * q_sigma_stf / pressure[..., None, None, None]
        - 2.0
        * (mu * theta / pressure)[..., None, None, None]
        * (stf_stress_gradient - sigma_logp_stf)
    )
    m_closure = third_moment - m_target
    residual[..., IDX["m_xxx"]] = m_closure[..., 0, 0, 0]
    residual[..., IDX["m_xyy"]] = m_closure[..., 0, 1, 1]
    residual[..., IDX["m_xxy"]] = m_closure[..., 0, 0, 1]
    residual[..., IDX["m_yyy"]] = m_closure[..., 1, 1, 1]

    sigma2 = np.einsum("...ij,...ij->...", stress, stress)
    heat2 = np.einsum("...i,...i->...", heat, heat)
    delta_target = (
        5.0 * sigma2 / rho
        + (56.0 / 5.0) * heat2 / pressure
        - 12.0
        * (mu * theta / pressure)
        * (divergence_heat - np.einsum("...i,...i->...", heat, grad_log_pressure))
    )
    residual[..., IDX["Delta"]] = center[..., IDX["Delta"]] - delta_target
    return residual


def constrained_residual(state: np.ndarray, config: NonlinearR13Config) -> np.ndarray:
    residual = nonlinear_residual(state, config).reshape(-1).copy()
    residual[0] = float(np.mean(state[..., IDX["rho"]]))
    return residual


def assemble_colored_jacobian(
    state: np.ndarray,
    config: NonlinearR13Config,
    *,
    base_residual: np.ndarray | None = None,
    drop_tolerance: float = 1.0e-12,
) -> sparse.csr_matrix:
    """Local finite-difference Jacobian in 9 color batches per variable."""
    if base_residual is None:
        base_residual = constrained_residual(state, config)
    base_grid = base_residual.reshape(config.ny, config.nx, NVAR)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for variable in range(NVAR):
        scale = max(1.0, float(np.max(np.abs(state[..., variable]))))
        step = config.jacobian_step * scale
        for color_j in range(3):
            for color_i in range(3):
                probe = state.copy()
                nodes: list[tuple[int, int]] = []
                for j in range(color_j, config.ny, 3):
                    for i in range(color_i, config.nx, 3):
                        probe[j, i, variable] += step
                        nodes.append((j, i))
                if not nodes:
                    continue
                response = (
                    constrained_residual(probe, config).reshape(
                        config.ny, config.nx, NVAR
                    )
                    - base_grid
                ) / step
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
    jacobian = sparse.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()
    jacobian.sum_duplicates()
    # The constraint row is globally dense and cannot be recovered by local
    # support extraction.  Install it explicitly after colored assembly.
    jacobian = jacobian.tolil()
    jacobian[0, :] = 0.0
    rho_columns = [
        _flat_index(config, j, i, IDX["rho"])
        for j in range(config.ny)
        for i in range(config.nx)
    ]
    jacobian[0, rho_columns] = 1.0 / (config.nx * config.ny)
    return jacobian.tocsr()


def _qmr_step(
    jacobian: sparse.csr_matrix,
    rhs: np.ndarray,
    config: NonlinearR13Config,
) -> tuple[np.ndarray, int, float]:
    identity = LinearOperator(
        jacobian.shape,
        matvec=lambda value: value,
        rmatvec=lambda value: value,
        dtype=float,
    )
    iterations = 0

    def record(_: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    step, info = qmr(
        jacobian,
        rhs,
        rtol=max(config.qmr_rtol, 1.0e-9),
        atol=0.0,
        maxiter=config.qmr_maxiter,
        M1=identity,
        M2=identity,
        callback=record,
    )
    relative = float(
        np.linalg.norm(jacobian @ step - rhs)
        / max(np.linalg.norm(rhs), np.finfo(float).tiny)
    )
    if info != 0 or not np.isfinite(step).all():
        raise RuntimeError(f"Newton QMR failed: info={info}, residual={relative:.3e}")
    return step, iterations, relative


def solve_nonlinear_r13(
    config: NonlinearR13Config,
    initial: np.ndarray,
    *,
    callback: Callable[[dict[str, float | int]], None] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    config.validate()
    state = np.asarray(initial, dtype=float).copy()
    if state.shape != (config.ny, config.nx, NVAR):
        raise ValueError("initial state shape does not match configuration")
    history: list[dict[str, float | int]] = []
    initial_norm = max(
        float(np.linalg.norm(constrained_residual(state, config))),
        np.finfo(float).tiny,
    )
    converged = False
    for iteration in range(1, config.nonlinear_maxiter + 1):
        residual = constrained_residual(state, config)
        residual_norm = float(np.linalg.norm(residual))
        relative_residual = residual_norm / initial_norm
        if relative_residual <= config.nonlinear_rtol:
            converged = True
            break
        jacobian = assemble_colored_jacobian(
            state, config, base_residual=residual
        )
        newton_step, qmr_iterations, qmr_residual = _qmr_step(
            jacobian, -residual, config
        )
        step_grid = newton_step.reshape(state.shape)
        line_step = 1.0
        accepted = False
        candidate = state
        candidate_norm = residual_norm
        while line_step >= config.minimum_line_step:
            trial = state + line_step * step_grid
            if (
                np.min(1.0 + trial[..., IDX["rho"]]) <= 0.0
                or np.min(1.0 + trial[..., IDX["theta"]]) <= 0.0
            ):
                line_step *= 0.5
                continue
            try:
                trial_residual = constrained_residual(trial, config)
            except FloatingPointError:
                line_step *= 0.5
                continue
            trial_norm = float(np.linalg.norm(trial_residual))
            if trial_norm < residual_norm:
                candidate = trial
                candidate_norm = trial_norm
                accepted = True
                break
            line_step *= 0.5
        if not accepted:
            raise RuntimeError(
                f"Newton line search failed at iteration {iteration}; "
                f"residual={residual_norm:.3e}"
            )
        update = relative_l1_change(candidate, state)
        state = candidate
        record: dict[str, float | int] = {
            "iteration": iteration,
            "residual_norm": candidate_norm,
            "relative_residual": candidate_norm / initial_norm,
            "relative_l1_update": update,
            "line_step": line_step,
            "qmr_iterations": qmr_iterations,
            "qmr_relative_residual": qmr_residual,
            "jacobian_nnz": int(jacobian.nnz),
        }
        history.append(record)
        if callback is not None:
            callback(record)
        if (
            candidate_norm / initial_norm <= config.nonlinear_rtol
            and update <= config.nonlinear_update_tol
        ):
            converged = True
            break
    final_residual = constrained_residual(state, config)
    report: dict[str, object] = {
        "converged": converged,
        "iterations": len(history),
        "initial_residual_norm": initial_norm,
        "final_residual_norm": float(np.linalg.norm(final_residual)),
        "relative_residual": float(np.linalg.norm(final_residual)) / initial_norm,
        "mass_perturbation": float(np.mean(state[..., IDX["rho"]])),
        "rho_min": float(np.min(1.0 + state[..., IDX["rho"]])),
        "theta_min": float(np.min(1.0 + state[..., IDX["theta"]])),
        "finite": bool(np.isfinite(state).all()),
        "history": history,
    }
    if not converged:
        raise RuntimeError(
            "nonlinear solver did not satisfy both residual and update gates: "
            + json.dumps(report, sort_keys=True)
        )
    return state, report


def nonlinear_cavity_metrics(
    state: np.ndarray, config: NonlinearR13Config
) -> dict[str, float]:
    extended = nonlinear_ghosted_state(state, config)
    top = extended[-1, 1:-1]
    sigma_over_p0_signed = float(
        config.dx * np.sum(top[..., IDX["sigma_xy"]])
    )
    reduction_factor = np.sqrt(2.0) / abs(config.lid_velocity)
    signed_d = reduction_factor * sigma_over_p0_signed
    x_coordinates = np.arange(1, config.nx + 1, dtype=float) * config.dx
    center_velocity = np.asarray(
        [
            np.interp(0.5, x_coordinates, row)
            for row in state[..., IDX["vx"]]
        ]
    )
    return {
        "D_signed": signed_d,
        "D": abs(signed_d),
        "D_sigma_over_p0_signed": sigma_over_p0_signed,
        "D_reduced_stress_factor": reduction_factor,
        "G": float(
            config.dy
            * np.sum(np.abs(center_velocity))
            / abs(config.lid_velocity)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=6)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--kn", type=float, default=0.01)
    parser.add_argument("--lid-velocity", type=float, default=0.05)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = NonlinearR13Config(
        nx=args.nx,
        ny=args.ny,
        kn=args.kn,
        lid_velocity=args.lid_velocity,
        qmr_maxiter=12000,
    )
    if args.initial is None:
        initial = np.zeros((config.ny, config.nx, NVAR))
    else:
        initial = np.load(args.initial)
    state, solver = solve_nonlinear_r13(
        config,
        initial,
        callback=lambda item: print(json.dumps(item, sort_keys=True)),
    )
    report = {
        "model": "nonlinear transformed R13",
        "configuration": config.__dict__,
        "solver": solver,
        "metrics": nonlinear_cavity_metrics(state, config),
        "scientific_status": "verification-only; grid convergence still required",
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
