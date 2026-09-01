# Moment-gate sensitivity audit

**Verdict: FAIL**

This audit changes estimator, kinetic sampling horizon, and holdout protocol without using the held-out target to tune a model.

| Protocol | Estimator | Depths/lambda | Target | S0 NRMSE | S1 NRMSE | S2 NRMSE | S1 gain |
|---|---|---:|---:|---:|---:|---:|---:|
| LOCO | extra_trees | 0.5 | cp | 89.33% | 92.08% | 92.18% | -3.09% |
| LOCO | extra_trees | 0.5 | cf | 39.45% | 39.50% | 39.57% | -0.12% |
| LOCO | extra_trees | 0.5+1+2 | cp | 91.62% | 94.04% | 93.34% | -2.63% |
| LOCO | extra_trees | 0.5+1+2 | cf | 39.33% | 39.51% | 38.82% | -0.45% |
| LOCO | ridge_100 | 0.5 | cp | 76.35% | 78.35% | 79.77% | -2.62% |
| LOCO | ridge_100 | 0.5 | cf | 39.98% | 33.51% | 33.63% | 16.19% |
| LOCO | ridge_100 | 0.5+1+2 | cp | 77.55% | 81.09% | 80.25% | -4.56% |
| LOCO | ridge_100 | 0.5+1+2 | cf | 39.13% | 34.69% | 33.56% | 11.35% |
| LOO-case | extra_trees | 0.5 | cp | 68.60% | 72.33% | 75.03% | -5.44% |
| LOO-case | extra_trees | 0.5 | cf | 38.49% | 38.37% | 38.89% | 0.32% |
| LOO-case | extra_trees | 0.5+1+2 | cp | 71.33% | 79.29% | 80.84% | -11.16% |
| LOO-case | extra_trees | 0.5+1+2 | cf | 38.52% | 39.23% | 38.24% | -1.83% |
| LOO-case | ridge_100 | 0.5 | cp | 69.42% | 72.53% | 75.63% | -4.48% |
| LOO-case | ridge_100 | 0.5 | cf | 39.30% | 33.18% | 33.43% | 15.58% |
| LOO-case | ridge_100 | 0.5+1+2 | cp | 72.45% | 76.25% | 75.40% | -5.24% |
| LOO-case | ridge_100 | 0.5+1+2 | cf | 39.22% | 34.51% | 33.31% | 12.02% |

## Interpretation

The largest aggregate signed-shear gain is 16.19% (LOCO, ridge_100, depths 0.5 lambda).
Its held-out gains are not all positive: BWD=31.2%, FWD=11.8%, ISO=-11.5%.
Pressure does not improve when the full momentum-flux tensor is added.
The conclusion is therefore insensitive to the tested tree/linear models and sampling horizons: full-range moments are not sufficient for a transferable operator on this six-case pilot.
The next diagnostic is the already-generated ISO collision tally, which can test incident half-range moments without launching more DSMC cases.
