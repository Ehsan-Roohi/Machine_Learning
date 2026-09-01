# Cross-geometry incident half-range gate

S3 uses pre-collision molecular velocities and the known 300 K diffuse-wall kernel. 
The full-impulse control alone uses post-collision velocity. Surface-normal orientation is inferred 
elementwise from incident velocities, which is required for the overhanging FWD/BWD faces.

## ISO_Ma6_Kn0p1

Records used: 188,120; minimum per element: 109.

| Quantity | Concurrent S3 NRMSE | r | Full-impulse control | DSMC RMS SEM | Independent S3 NRMSE | Independent DSMC SEM |
|---|---:|---:|---:|---:|---:|---:|
| CP | 0.28% | 1.0000 | 0.057% | 1.82% | — | — |
| CF | 7.35% | 0.9975 | 0.649% | 11.63% | — | — |
| CQ | 0.67% | 1.0000 | 0.053% | 2.63% | — | — |

## ISO_Ma6_Kn0p8

Records used: 334,469; minimum per element: 220.

| Quantity | Concurrent S3 NRMSE | r | Full-impulse control | DSMC RMS SEM | Independent S3 NRMSE | Independent DSMC SEM |
|---|---:|---:|---:|---:|---:|---:|
| CP | 0.16% | 1.0000 | 0.056% | 1.17% | — | — |
| CF | 2.08% | 0.9998 | 0.375% | 3.47% | — | — |
| CQ | 0.14% | 1.0000 | 0.053% | 1.46% | — | — |

## BWD_Ma6_Kn0p2

Records used: 295,672; minimum per element: 189.

| Quantity | Concurrent S3 NRMSE | r | Full-impulse control | DSMC RMS SEM | Independent S3 NRMSE | Independent DSMC SEM |
|---|---:|---:|---:|---:|---:|---:|
| CP | 0.25% | 1.0000 | 0.057% | 1.19% | 2.41% | 1.83% |
| CF | 1.95% | 0.9997 | 0.054% | 3.85% | 6.77% | 6.23% |
| CQ | 0.30% | 1.0000 | 0.053% | 1.74% | 2.80% | 2.60% |

## BWD_Ma6_Kn0p4

Records used: 317,452; minimum per element: 212.

| Quantity | Concurrent S3 NRMSE | r | Full-impulse control | DSMC RMS SEM | Independent S3 NRMSE | Independent DSMC SEM |
|---|---:|---:|---:|---:|---:|---:|
| CP | 0.15% | 1.0000 | 0.056% | 1.14% | 1.70% | 1.71% |
| CF | 1.30% | 0.9999 | 0.054% | 2.23% | 5.52% | 4.24% |
| CQ | 0.16% | 1.0000 | 0.053% | 1.34% | 2.80% | 2.30% |

## FWD_Ma6_Kn0p2

Records used: 353,417; minimum per element: 381.

| Quantity | Concurrent S3 NRMSE | r | Full-impulse control | DSMC RMS SEM | Independent S3 NRMSE | Independent DSMC SEM |
|---|---:|---:|---:|---:|---:|---:|
| CP | 0.17% | 1.0000 | 0.056% | 1.12% | 2.04% | 1.84% |
| CF | 7.32% | 0.9937 | 1.064% | 15.72% | 25.09% | 20.70% |
| CQ | 0.36% | 1.0000 | 0.053% | 1.67% | 3.42% | 2.60% |

## FWD_Ma6_Kn0p4

Records used: 404,401; minimum per element: 385.

| Quantity | Concurrent S3 NRMSE | r | Full-impulse control | DSMC RMS SEM | Independent S3 NRMSE | Independent DSMC SEM |
|---|---:|---:|---:|---:|---:|---:|
| CP | 0.16% | 1.0000 | 0.056% | 0.98% | 1.68% | 1.53% |
| CF | 5.71% | 0.9963 | 0.803% | 8.92% | 16.65% | 13.93% |
| CQ | 0.30% | 1.0000 | 0.053% | 1.57% | 2.89% | 1.96% |

## Decision

- Cross-geometry concurrent Cp gate (<2%): PASS.
- Cross-geometry concurrent signed-Cf gate (<15%): PASS.
- Independent-window FWD shear differences must be interpreted against its 9–21% DSMC block uncertainty.

