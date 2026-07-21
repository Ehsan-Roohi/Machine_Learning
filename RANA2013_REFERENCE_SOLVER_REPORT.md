# Rana 2013 independent reference solver

Reference: A. Rana, M. Torrilhon and H. Struchtrup, *A robust numerical
method for the R13 equations of rarefied gas dynamics: Application to lid
driven cavity*, Journal of Computational Physics 236 (2013) 169--186,
doi:10.1016/j.jcp.2012.11.023.

## Outcome

The repository now has an independent steady 17-variable reference solver in
`rana2013_r13/`.  It does not use the ASTR pseudo-time solver or its local
wall update.  It implements the paper's centered finite-difference operator,
simultaneous two-wall corner contribution, total-mass constraint, linear first
iterate, and nonlinear Newton--QMR solve.

For `Kn=0.010`, `N=75`, and `U*=0.2096`, a fresh local verification gave:

| quantity | computed | paper | relative error |
|---|---:|---:|---:|
| D | 0.1600351 | 0.1585 | +0.97% |
| G | 0.1798914 | 0.1893 | -4.97% |

The nonlinear algebraic relative residual was `2.52e-12`; the mass
perturbation was `6.63e-18`; all state values were finite; and the minimum
density and temperature were `0.9041` and `0.9961`, respectively.  This is a
verification result, not a publication-grade claim: an independent CI rerun
and a nonlinear grid study remain required.

## Root cause of the historical D discrepancy

The former analyzer used the multiplier

```text
1 / (sqrt(2) * U*)
```

for the low-speed drag coefficient.  The Rana/Sharipov convention requires

```text
sqrt(2) / U*
```

and the old result was therefore exactly a factor of two too small.  The N75
linear solve makes the error independently observable:

```text
integral(sigma_xy/p0 dx) = -0.0234899994
sqrt(2)/U*               =  6.74720211
abs(D)                   =  0.1584917734
```

This also explains why mass rescaling, corner zeroing, longer pseudo-time
integration, and wall-order variants could not repair D: they were being
judged with the wrong post-processing scale.

## Grid trend

All entries below are converged algebraic solutions of the same independent
operator; they are not pseudo-time checkpoints.

| N | linear D | linear G | nonlinear D | nonlinear G |
|---:|---:|---:|---:|---:|
| 20 | 0.18591 | 0.16262 | 0.19048 | 0.14587 |
| 30 | 0.17147 | 0.17194 | 0.17418 | 0.16291 |
| 50 | 0.16193 | 0.17820 | 0.16379 | 0.17360 |
| 75 | 0.15849 | 0.18144 | 0.16004 | 0.17989 |

The monotone trend is why the coarse nonlinear N12/N20 values must not be
used as model verdicts.

## Reproduction

Fast tests:

```bash
python3 rana2013_r13/test_discrete_operator_oracle.py
python3 rana2013_r13/test_linear_reference_solver.py
python3 rana2013_r13/test_nonlinear_reference_solver.py
python3 rana2013_r13/test_reference_benchmark.py
python3 diagnostics/test_kn0010_recovery_metrics.py
```

Fresh N75 verification:

```bash
python3 rana2013_r13/run_reference_benchmark.py \
  --nx 75 --ny 75 --kn 0.01 --lid-velocity 0.2096 \
  --nonlinear --qmr-maxiter 40000 --nonlinear-maxiter 12 \
  --minimum-line-step 0.0009765625 --require-paper-comparison \
  --output-dir rana2013-reference-N75
```

The driver writes the linear and nonlinear states, SHA-256 hashes, complete
iteration history, metrics, independent gates, and an explicit
`publication_grade=false` verdict.

## Safety gates

- The existing `paper-core` and `paper-grid-study` jobs remain under literal
  `if: ${{ false }}` hard holds.
- The new workflow runs only unit/operator tests and a small nonlinear smoke.
- No ASTR continuation or old dirty checkpoint is used by the reference
  solver.
- Production-sized or multi-Kn runs must not be enabled until the smoke
  artifact is reviewed and the metric convention is independently confirmed.
