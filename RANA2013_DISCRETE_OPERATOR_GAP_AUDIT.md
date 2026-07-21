# Rana 2013 discrete-operator gap audit

Reference: A. Rana, M. Torrilhon and H. Struchtrup, *A robust numerical
method for the R13 equations of rarefied gas dynamics: Application to lid
driven cavity*, JCP 236 (2013) 169--186.

## Decision

**The ASTR production path remains stopped, but an independent reference
path is now implemented and locally verified.** The current ASTR branch contains
important equation-level pieces of Rana's transformed nonlinear R13 model,
including Eq. (13) coefficient fingerprints and local rearrangements of
Eqs. (7a--f). It does not implement the paper's discrete boundary-value
operator in Eqs. (14--25), so it cannot yet be called paper-exact.

This is not a stylistic difference between two equivalent solvers. The
discrete equations at smooth walls and corners are different.

## Direct source evidence

1. `bc.F90::boucon` loops over faces `n=1..6` and calls
   `NSslip_wall_boundary(n)` separately.
2. `methodmoment.F90::rk3mom` does the same for `MOM_wall_boundary(n)`.
3. Each x-wall loop includes `j=0..jm` and each y-wall loop includes
   `i=0..im`. Consequently a corner belongs to two calls and the later call
   writes the state produced by the earlier call.
4. `R13wbc` and `R13wbc_slip` are damped, local fixed-point maps. The bulk
   and moment states are advanced by explicit RK pseudo-time steps.
5. There is no explicit 17-state global `A/B/P` assembly, `X+/X-/Y+/Y-`
   boundary matrix, coupled Eq. (20) corner row, bordered null-space mass
   row, or QMR solve.

The companion executable audit is
`diagnostics/audit_rana2013_discrete_operator.py`.

The independent implementation is in
`rana2013_r13/linear_reference_solver.py` and
`rana2013_r13/nonlinear_reference_solver.py`.  Unlike ASTR, it evaluates both
wall contributions in the same corner residual, imposes a global mass row,
and solves the steady algebraic system with QMR.  Its N75 Kn=0.010 result is
documented in `RANA2013_REFERENCE_SOLVER_REPORT.md`.  This does not remove the
hard holds on the older ASTR `paper-core` and `paper-grid-study` jobs.

## Why this explains the D/G pattern

`G` is a bulk velocity integral. The legacy 120k trajectory produced a value
near the paper even though it was dirty and not converged. `D` is an integral
of the moving-wall shear stress, so it is directly controlled by the wall and
corner stress equations. The observed combination--reasonable `G`, strongly
low `D`, order-dependent corner failure, and large `Rxy/Delta/qx` changes--is
therefore consistent with a boundary-operator mismatch.

This is a causal diagnosis of the implementation gap, not a claim that the
gap is the only source of numerical error. Normalization, grid convergence,
and final residuals remain independent gates.

## Paper method that is currently missing

- Eq. (11): one steady first-order system for the 17-component state
  `U=(rho,vx,vy,theta,qx,qy,sxx,sxy,syy,Rxx,Rxy,Ryy,mxxx,mxyy,mxxy,myyy,Delta)`.
- Eqs. (15--16): Eq. (7) is represented by `X` and `Y` boundary matrices and
  inhomogeneous wall vectors, with linear extrapolation at boundary nodes.
- Eqs. (18--19): those matrices are inserted into the discrete PDE rows;
  boundary values are not imposed afterward as an independent projection.
- Eq. (20): a corner row contains both wall contributions simultaneously.
- Eqs. (23--25): total mass is imposed in the bordered null-space system.
- Section 4.4: each nonlinear fixed-point step solves the linear system with
  QMR and stops on the paper's norm tolerance.

## Required implementation order

1. Build an independent Python oracle for the Appendix-A `A(U)`, `B(U)`, and
   `P(U)` matrices and the four oriented `X/Y` boundary matrices.
2. Add manufactured residual tests for all four smooth walls and all four
   corners. At a corner, verify the single Eq. (20) row, not two sequential
   wall assignments.
3. Add the trapezoidal mass row and left/right-null-vector checks from
   Eqs. (23--25).
4. Implement the same blocks in a dedicated solver path; do not retrofit them
   into `R13wbc` as another local corner convention.
5. Require static audit, matrix-oracle tests, strict build, and a 10-step
   smoke before one fresh `Kn=0.010, N=75` diagnostic.
6. Only after clean completion and residual convergence compare `D` and `G`.

The existing `paper-core` and `paper-grid-study` hard holds must remain in
place throughout steps 1--4.
