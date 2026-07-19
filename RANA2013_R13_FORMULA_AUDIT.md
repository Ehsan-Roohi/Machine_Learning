# Rana–Torrilhon–Struchtrup (2013) exact R13 benchmark

Reference: A. Rana, M. Torrilhon and H. Struchtrup, *A robust numerical method for the R13 equations of rarefied gas dynamics: Application to lid driven cavity*, JCP 236 (2013) 169–186.

## Source changes

The dedicated branch modifies only the R13 path. Existing R26 workflows and source branches are not changed.

### Maxwell R13 balance equations

The existing ASTR stress and heat-flux balance implementation is retained. The audit checks the Maxwell relaxation coefficients and fingerprints every published group in Eqs. (3) and (4): stress relaxation, heat-flux relaxation, STF velocity-gradient term, STF heat-flux-gradient term, higher-moment divergence channels, Delta gradient, and nonlinear stress/heat-flux couplings.

### Transformed nonlinear closures: Eq. (13)

The old gradient-only closures are replaced by the complete transformed nonlinear relations:

- `m_ijk`: `4/3 q_<i sigma_jk>/p` and the quotient gradient `-2 mu theta d_<i(sigma_jk>/p)`;
- `R_ij`: `20/7 sigma_k<i sigma_j>k/rho`, `64/25 q_<i q_j>/p`, and `-24/5 mu theta d_<i(q_j>/p)`;
- `Delta`: `5 sigma:sigma/rho`, `56/5 q.q/p`, and `-12 mu theta div(q/p)`.

The implementation is tensor-generic and a randomized independent NumPy oracle checks symmetry, trace-free projection, quotient-gradient identities and equilibrium limits to `1e-11`.

### Wall conditions: Eqs. (7a–f)

The previous Gu/ASTR wall rearrangement is replaced on the dedicated 2D Rana branch. In particular, the effective wall pressure uses the paper's tangential components,

`P = p + sigma_tt/2 - Delta/(120 theta) - R_tt/(28 theta)`.

All six conditions are implemented algebraically with full accommodation `chi=1`. The solver uses a damped, checkpoint-invariant fixed-point iteration of the two exact maps used by ASTR's split boundary update: Eqs. (7a–c) set normal velocity, tangential slip and temperature jump; Eqs. (7b–f) set stress and heat-flux boundary values. The damping changes only the nonlinear iteration path, not the Eq. (7) fixed point. No uncheckpointed wall-memory arrays are used, so restart changes cannot alter the fixed point.

## Paper conditions reproduced

- square cavity, homogeneous `z` direction;
- `T0 = 273 K` on all walls;
- top-lid final speed `50 m/s`, reached through a 1000-iteration numerical homotopy; the final boundary condition is exactly the paper value;
- full Maxwell accommodation `chi=1`;
- paper Knudsen definition `Kn = mu/(rho sqrt(theta) L)`;
- principal/profile and table values: `0.010, 0.071, 0.0798, 0.141, 0.354, 0.3989, 0.707`;
- mesh ladder at `Kn = 0.05, 0.10, 0.50` for `N = 40, 50, 75, 100, 200, 400` grid points.

The workflow exports centerline profiles, Eq. (30) metrics `D` and `G`, paper errors where tabulated, all six wall-equation residuals, last-checkpoint changes, logs and restartable cases.

## Validation gates

1. patch count and coefficient audit;
2. randomized tensor and wall-equation oracle;
3. full Fortran build;
4. fresh 40×40 smoke run from equilibrium, through the complete lid homotopy and at the final paper speed;
5. positive/finite thermodynamic fields and positive effective wall pressure;
6. matrix benchmark runs with labeled partial artifacts if the GitHub runner wall-time limit is reached.
