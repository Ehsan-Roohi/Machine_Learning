#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from discrete_operator_oracle import NVAR
from linear_reference_solver import (
    IDX,
    LinearR13Config,
    assemble_colored_operator,
    cavity_metrics,
    equilibrium_perturbation,
    ghosted_state,
    linear_residual,
    linear_wall_state,
    solve_linear_r13,
)


def test_equilibrium_is_exact_for_stationary_walls() -> None:
    config = LinearR13Config(nx=4, ny=5, lid_velocity=0.0)
    state = equilibrium_perturbation(config)
    np.testing.assert_array_equal(linear_residual(state, config), 0.0)


def test_wall_orientation_and_controlled_rows() -> None:
    state = np.zeros((2, NVAR))
    state[:, IDX["vy"]] = 0.3
    plus = linear_wall_state(
        state,
        axis="x",
        normal_sign=1,
        wall_tangential_velocity=0.0,
        accommodation=1.0,
    )
    minus = linear_wall_state(
        state,
        axis="x",
        normal_sign=-1,
        wall_tangential_velocity=0.0,
        accommodation=1.0,
    )
    assert np.all(plus[:, IDX["vx"]] == 0.0)
    np.testing.assert_allclose(
        plus[:, IDX["sigma_xy"]], -minus[:, IDX["sigma_xy"]]
    )
    np.testing.assert_allclose(plus[:, IDX["vy"]], state[:, IDX["vy"]])


def test_top_and_right_ghosts_coexist_at_corner() -> None:
    config = LinearR13Config(nx=4, ny=4, lid_velocity=0.2)
    extended = ghosted_state(equilibrium_perturbation(config), config)
    # Top wall is driven, right wall is stationary.  The two boundary arrays
    # are separate inputs to the top-right residual and cannot overwrite.
    assert np.max(np.abs(extended[-1, 1:-1])) > 0.0
    np.testing.assert_array_equal(extended[1:-1, -1], 0.0)


def test_colored_assembly_matches_matrix_free_residual() -> None:
    config = LinearR13Config(nx=4, ny=4, lid_velocity=0.2)
    operator, affine = assemble_colored_operator(config)
    rng = np.random.default_rng(2013)
    state = rng.normal(scale=1.0e-3, size=(config.ny, config.nx, NVAR))
    expected = linear_residual(state, config).reshape(-1)
    actual = operator @ state.reshape(-1) + affine
    np.testing.assert_allclose(actual, expected, rtol=2.0e-12, atol=2.0e-12)


def test_direct_solution_is_finite_mass_constrained_and_small_residual() -> None:
    config = LinearR13Config(nx=4, ny=4, kn=0.01, lid_velocity=0.05)
    state, diagnostics = solve_linear_r13(config, direct=True)
    metrics = cavity_metrics(state, config)
    assert diagnostics["finite"]
    assert abs(float(diagnostics["mass_perturbation"])) < 1.0e-11
    assert float(diagnostics["relative_linear_residual"]) < 1.0e-10
    assert metrics["rho_min"] > 0.0
    assert metrics["theta_min"] > 0.0
    assert metrics["D"] > 0.0
    assert metrics["G"] > 0.0


def test_drag_uses_rana_reduced_stress_normalization() -> None:
    config = LinearR13Config(nx=4, ny=4, lid_velocity=0.2)
    state = equilibrium_perturbation(config)
    # The wall map may alter the extrapolated stress; the metric must still
    # expose and reduce exactly the same sigma/p0 integral.
    state[..., IDX["sigma_xy"]] = 0.04
    metrics = cavity_metrics(state, config)
    expected = (
        np.sqrt(2.0)
        * metrics["D_sigma_over_p0_signed"]
        / config.lid_velocity
    )
    np.testing.assert_allclose(metrics["D_signed"], expected)


def test_qmr_agrees_with_direct_small_grid() -> None:
    config = LinearR13Config(
        nx=4,
        ny=4,
        kn=0.01,
        lid_velocity=0.05,
        qmr_rtol=1.0e-10,
        qmr_maxiter=2000,
    )
    direct, _ = solve_linear_r13(config, direct=True)
    iterative, diagnostics = solve_linear_r13(config, direct=False)
    assert diagnostics["qmr_info"] == 0
    np.testing.assert_allclose(iterative, direct, rtol=2.0e-7, atol=2.0e-9)


if __name__ == "__main__":
    test_equilibrium_is_exact_for_stationary_walls()
    test_wall_orientation_and_controlled_rows()
    test_top_and_right_ghosts_coexist_at_corner()
    test_colored_assembly_matches_matrix_free_residual()
    test_direct_solution_is_finite_mass_constrained_and_small_residual()
    test_drag_uses_rana_reduced_stress_normalization()
    test_qmr_agrees_with_direct_small_grid()
    print("Rana 2013 linear reference-solver tests: PASS")
