#!/usr/bin/env python3
"""Independent block-row oracle for Rana et al. (2013), Eqs. (17)--(25).

This module does not invent the Appendix-A physics matrices.  It takes the
reviewed 17x17 A/B/P and oriented X/Y boundary matrices as inputs, then applies
the paper's discrete algebra exactly.  It is intentionally independent of the
ASTR pseudo-time boundary routines and is suitable as an oracle for a future
Fortran implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


STATE_ORDER = (
    "rho",
    "vx",
    "vy",
    "theta",
    "qx",
    "qy",
    "sigma_xx",
    "sigma_xy",
    "sigma_yy",
    "R_xx",
    "R_xy",
    "R_yy",
    "m_xxx",
    "m_xyy",
    "m_xxy",
    "m_yyy",
    "Delta",
)
NVAR = len(STATE_ORDER)


@dataclass(frozen=True)
class BlockRow:
    center: np.ndarray
    rhs: np.ndarray
    west: np.ndarray | None = None
    east: np.ndarray | None = None
    south: np.ndarray | None = None
    north: np.ndarray | None = None


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (NVAR, NVAR):
        raise ValueError(f"{name} must have shape {(NVAR, NVAR)}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite entries")
    return result


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (NVAR,):
        raise ValueError(f"{name} must have shape {(NVAR,)}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite entries")
    return result


def _scales(dx: float, dy: float, kn: float) -> tuple[float, float, float]:
    dx, dy, kn = float(dx), float(dy), float(kn)
    if dx <= 0.0 or dy <= 0.0 or kn <= 0.0:
        raise ValueError("dx, dy, and kn must be positive")
    return dx, dy, kn


def interior_row(
    A: np.ndarray,
    B: np.ndarray,
    P: np.ndarray,
    *,
    dx: float,
    dy: float,
    kn: float,
) -> BlockRow:
    """Rana Eq. (17), central difference at an interior node."""
    A, B, P = _matrix(A, "A"), _matrix(B, "B"), _matrix(P, "P")
    dx, dy, kn = _scales(dx, dy, kn)
    return BlockRow(
        center=P / kn,
        west=-A / (2.0 * dx),
        east=A / (2.0 * dx),
        south=-B / (2.0 * dy),
        north=B / (2.0 * dy),
        rhs=np.zeros(NVAR),
    )


def boundary_row(
    A: np.ndarray,
    B: np.ndarray,
    P: np.ndarray,
    boundary_matrix: np.ndarray,
    boundary_data: np.ndarray,
    *,
    side: str,
    dx: float,
    dy: float,
    kn: float,
) -> BlockRow:
    """Rana Eqs. (18)--(19) and their y-wall counterparts."""
    A, B, P = _matrix(A, "A"), _matrix(B, "B"), _matrix(P, "P")
    X = _matrix(boundary_matrix, "boundary_matrix")
    d = _vector(boundary_data, "boundary_data")
    dx, dy, kn = _scales(dx, dy, kn)
    I = np.eye(NVAR)
    base = dict(
        center=P / kn,
        west=None,
        east=None,
        south=None,
        north=None,
        rhs=np.zeros(NVAR),
    )
    if side == "left":
        base.update(
            center=P / kn - A @ X / dx,
            east=A @ (I + X) / (2.0 * dx),
            south=-B / (2.0 * dy),
            north=B / (2.0 * dy),
            rhs=A @ d / (2.0 * dx),
        )
    elif side == "right":
        base.update(
            center=P / kn + A @ X / dx,
            west=-A @ (I + X) / (2.0 * dx),
            south=-B / (2.0 * dy),
            north=B / (2.0 * dy),
            rhs=-A @ d / (2.0 * dx),
        )
    elif side == "bottom":
        base.update(
            center=P / kn - B @ X / dy,
            north=B @ (I + X) / (2.0 * dy),
            west=-A / (2.0 * dx),
            east=A / (2.0 * dx),
            rhs=B @ d / (2.0 * dy),
        )
    elif side == "top":
        base.update(
            center=P / kn + B @ X / dy,
            south=-B @ (I + X) / (2.0 * dy),
            west=-A / (2.0 * dx),
            east=A / (2.0 * dx),
            rhs=-B @ d / (2.0 * dy),
        )
    else:
        raise ValueError("side must be left, right, bottom, or top")
    return BlockRow(**base)


def corner_row(
    A: np.ndarray,
    B: np.ndarray,
    P: np.ndarray,
    X: np.ndarray,
    Xd: np.ndarray,
    Y: np.ndarray,
    Yd: np.ndarray,
    *,
    x_side: str,
    y_side: str,
    dx: float,
    dy: float,
    kn: float,
) -> BlockRow:
    """Coupled two-wall corner row; lower-left is paper Eq. (20)."""
    A, B, P = _matrix(A, "A"), _matrix(B, "B"), _matrix(P, "P")
    X, Y = _matrix(X, "X"), _matrix(Y, "Y")
    Xd, Yd = _vector(Xd, "Xd"), _vector(Yd, "Yd")
    dx, dy, kn = _scales(dx, dy, kn)
    if x_side not in ("left", "right"):
        raise ValueError("x_side must be left or right")
    if y_side not in ("bottom", "top"):
        raise ValueError("y_side must be bottom or top")
    I = np.eye(NVAR)
    sx = 1.0 if x_side == "left" else -1.0
    sy = 1.0 if y_side == "bottom" else -1.0
    return BlockRow(
        center=P / kn - sx * A @ X / dx - sy * B @ Y / dy,
        west=None if x_side == "left" else -A @ (I + X) / (2.0 * dx),
        east=A @ (I + X) / (2.0 * dx) if x_side == "left" else None,
        south=None if y_side == "bottom" else -B @ (I + Y) / (2.0 * dy),
        north=B @ (I + Y) / (2.0 * dy) if y_side == "bottom" else None,
        rhs=sx * A @ Xd / (2.0 * dx) + sy * B @ Yd / (2.0 * dy),
    )


def trapezoidal_mass_weights(nx: int, ny: int, *, dx: float, dy: float) -> np.ndarray:
    """Weights for paper Eq. (23), flattened with x varying fastest."""
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must each be at least 2")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be positive")
    wx = np.ones(nx)
    wy = np.ones(ny)
    wx[[0, -1]] = 0.5
    wy[[0, -1]] = 0.5
    return (dy * dx * np.outer(wy, wx)).reshape(-1)


def bordered_mass_system(
    M: np.ndarray,
    b: np.ndarray,
    weights: np.ndarray,
    left_null: np.ndarray,
    *,
    total_mass: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Paper Eqs. (24)--(25) for an already assembled global operator."""
    M = np.asarray(M, dtype=float)
    b = np.asarray(b, dtype=float)
    w = np.asarray(weights, dtype=float)
    xl = np.asarray(left_null, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be square")
    n = M.shape[0]
    if b.shape != (n,) or w.shape != (n,) or xl.shape != (n,):
        raise ValueError("b, weights, and left_null must match M")
    augmented = np.zeros((n + 1, n + 1))
    augmented[0, 1:] = w
    augmented[1:, 0] = xl
    augmented[1:, 1:] = M
    rhs = np.concatenate(([float(total_mass)], b))
    return augmented, rhs


def relative_l1_change(current: np.ndarray, previous: np.ndarray) -> float:
    """Nonlinear fixed-point change used to enforce a separate convergence gate."""
    current = np.asarray(current, dtype=float)
    previous = np.asarray(previous, dtype=float)
    if current.shape != previous.shape:
        raise ValueError("current and previous must have identical shapes")
    scale = max(float(np.sum(np.abs(current))), np.finfo(float).tiny)
    return float(np.sum(np.abs(current - previous)) / scale)
