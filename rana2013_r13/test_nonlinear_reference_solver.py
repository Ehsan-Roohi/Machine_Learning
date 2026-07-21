#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from discrete_operator_oracle import NVAR
from linear_reference_solver import IDX, LinearR13Config, linear_residual
from nonlinear_reference_solver import (
    NonlinearR13Config,
    assemble_colored_jacobian,
    constrained_residual,
    nonlinear_residual,
    nonlinear_cavity_metrics,
    nonlinear_wall_state,
    solve_nonlinear_r13,
)


def test_nonlinear_equilibrium_is_exact() -> None:
    config = NonlinearR13Config(nx=4, ny=4, lid_velocity=0.0)
    state = np.zeros((config.ny, config.nx, NVAR))
    np.testing.assert_array_equal(nonlinear_residual(state, config), 0.0)


def test_exact_wall_map_satisfies_all_six_equations() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(scale=2.0e-3, size=(5, NVAR))
    wall = nonlinear_wall_state(
        source,
        axis="y",
        normal_sign=-1,
        wall_tangential_velocity=0.1,
        accommodation=1.0,
    )
    assert np.all(wall[:, IDX["vy"]] == 0.0)
    assert np.isfinite(wall).all()
    # The eleven identity rows must not be modified.
    controlled = {
        IDX["vy"],
        IDX["qy"],
        IDX["sigma_xy"],
        IDX["R_xy"],
        IDX["m_yyy"],
        IDX["m_xxy"],
    }
    for variable in range(NVAR):
        if variable not in controlled:
            np.testing.assert_array_equal(wall[:, variable], source[:, variable])


def test_equilibrium_directional_derivative_matches_linear_r13() -> None:
    rng = np.random.default_rng(2013)
    perturbation = rng.normal(scale=2.0e-3, size=(4, 4, NVAR))
    lid_direction = 0.17
    epsilon = 2.0e-7
    nonlinear_config = NonlinearR13Config(
        nx=4, ny=4, kn=0.03, lid_velocity=epsilon * lid_direction
    )
    nonlinear = nonlinear_residual(epsilon * perturbation, nonlinear_config) / epsilon
    linear_config = LinearR13Config(
        nx=4, ny=4, kn=0.03, lid_velocity=lid_direction
    )
    linear = linear_residual(perturbation, linear_config)
    np.testing.assert_allclose(nonlinear, linear, rtol=3.0e-6, atol=3.0e-8)


def test_colored_jacobian_matches_directional_difference() -> None:
    rng = np.random.default_rng(31)
    config = NonlinearR13Config(nx=4, ny=4, kn=0.02, lid_velocity=0.03)
    state = rng.normal(scale=1.0e-4, size=(4, 4, NVAR))
    direction = rng.normal(scale=1.0e-4, size=state.shape)
    base = constrained_residual(state, config)
    jacobian = assemble_colored_jacobian(state, config, base_residual=base)
    epsilon = 2.0e-5
    finite_difference = (
        constrained_residual(state + epsilon * direction, config) - base
    ) / epsilon
    assembled = jacobian @ direction.reshape(-1)
    np.testing.assert_allclose(assembled, finite_difference, rtol=3.0e-4, atol=2.0e-7)


def test_small_lid_newton_converges() -> None:
    config = NonlinearR13Config(
        nx=4,
        ny=4,
        kn=0.01,
        lid_velocity=0.01,
        qmr_maxiter=5000,
        nonlinear_rtol=2.0e-7,
        nonlinear_update_tol=2.0e-7,
        nonlinear_maxiter=10,
    )
    initial = np.zeros((config.ny, config.nx, NVAR))
    state, report = solve_nonlinear_r13(config, initial)
    assert report["converged"]
    assert report["finite"]
    assert report["rho_min"] > 0.0
    assert report["theta_min"] > 0.0
    assert abs(report["mass_perturbation"]) < 1.0e-8
    assert np.max(np.abs(state)) > 0.0


def test_nonlinear_drag_uses_rana_reduced_stress_normalization() -> None:
    config = NonlinearR13Config(nx=4, ny=4, lid_velocity=0.2)
    state = np.zeros((config.ny, config.nx, NVAR))
    state[..., IDX["sigma_xy"]] = 0.04
    metrics = nonlinear_cavity_metrics(state, config)
    expected = (
        np.sqrt(2.0)
        * metrics["D_sigma_over_p0_signed"]
        / config.lid_velocity
    )
    np.testing.assert_allclose(metrics["D_signed"], expected)


if __name__ == "__main__":
    test_nonlinear_equilibrium_is_exact()
    test_exact_wall_map_satisfies_all_six_equations()
    test_equilibrium_directional_derivative_matches_linear_r13()
    test_colored_jacobian_matches_directional_difference()
    test_small_lid_newton_converges()
    test_nonlinear_drag_uses_rana_reduced_stress_normalization()
    print("Rana 2013 nonlinear reference-solver tests: PASS")
