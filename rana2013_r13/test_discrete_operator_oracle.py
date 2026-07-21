#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from discrete_operator_oracle import (
    NVAR,
    STATE_ORDER,
    bordered_mass_system,
    boundary_row,
    corner_row,
    interior_row,
    relative_l1_change,
    trapezoidal_mass_weights,
)


def matrices(seed: int = 2013):
    rng = np.random.default_rng(seed)
    shape = (NVAR, NVAR)
    return tuple(rng.normal(size=shape) for _ in range(5)) + tuple(
        rng.normal(size=NVAR) for _ in range(2)
    )


def test_state_order() -> None:
    assert NVAR == 17
    assert STATE_ORDER[0] == "rho"
    assert STATE_ORDER[-1] == "Delta"


def test_interior_eq17() -> None:
    A, B, P, _, _, _, _ = matrices()
    row = interior_row(A, B, P, dx=0.2, dy=0.3, kn=0.1)
    np.testing.assert_allclose(row.center, P / 0.1)
    np.testing.assert_allclose(row.east, A / 0.4)
    np.testing.assert_allclose(row.west, -A / 0.4)
    np.testing.assert_allclose(row.north, B / 0.6)
    np.testing.assert_allclose(row.south, -B / 0.6)


def test_lower_left_corner_is_exact_eq20_sum() -> None:
    A, B, P, X, Y, Xd, Yd = matrices()
    dx, dy, kn = 0.2, 0.3, 0.1
    row = corner_row(
        A,
        B,
        P,
        X,
        Xd,
        Y,
        Yd,
        x_side="left",
        y_side="bottom",
        dx=dx,
        dy=dy,
        kn=kn,
    )
    I = np.eye(NVAR)
    np.testing.assert_allclose(row.center, P / kn - A @ X / dx - B @ Y / dy)
    np.testing.assert_allclose(row.east, A @ (I + X) / (2.0 * dx))
    np.testing.assert_allclose(row.north, B @ (I + Y) / (2.0 * dy))
    np.testing.assert_allclose(row.rhs, A @ Xd / (2.0 * dx) + B @ Yd / (2.0 * dy))
    assert row.west is None and row.south is None


def test_sequential_face_row_is_not_corner_row() -> None:
    A, B, P, X, Y, Xd, Yd = matrices()
    corner = corner_row(
        A,
        B,
        P,
        X,
        Xd,
        Y,
        Yd,
        x_side="left",
        y_side="bottom",
        dx=0.2,
        dy=0.3,
        kn=0.1,
    )
    last_face_only = boundary_row(
        A, B, P, Y, Yd, side="bottom", dx=0.2, dy=0.3, kn=0.1
    )
    assert not np.allclose(last_face_only.center, corner.center)
    assert not np.allclose(last_face_only.rhs, corner.rhs)


def test_bordered_mass_row() -> None:
    n = 34
    M = np.eye(n)
    b = np.arange(n, dtype=float)
    w = np.linspace(1.0, 2.0, n)
    xl = np.linspace(2.0, 3.0, n)
    augmented, rhs = bordered_mass_system(M, b, w, xl, total_mass=1.0)
    assert augmented.shape == (n + 1, n + 1)
    np.testing.assert_allclose(augmented[0, 1:], w)
    np.testing.assert_allclose(augmented[1:, 0], xl)
    np.testing.assert_allclose(augmented[1:, 1:], M)
    assert rhs[0] == 1.0


def test_trapezoidal_mass_and_fixed_point_norm() -> None:
    weights = trapezoidal_mass_weights(3, 3, dx=0.5, dy=0.5)
    assert np.isclose(weights.sum(), 1.0)
    assert np.isclose(relative_l1_change(np.ones(4), np.full(4, 0.5)), 0.5)


if __name__ == "__main__":
    test_state_order()
    test_interior_eq17()
    test_lower_left_corner_is_exact_eq20_sum()
    test_sequential_face_row_is_not_corner_row()
    test_bordered_mass_row()
    test_trapezoidal_mass_and_fixed_point_norm()
    print("Rana 2013 discrete-operator oracle tests: PASS")
