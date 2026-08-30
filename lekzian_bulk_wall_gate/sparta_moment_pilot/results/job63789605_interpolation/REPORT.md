# Blind intermediate-Kn moment gate

**Verdict: FAIL**

This supersedes the first analysis committed for job 63789605.  The earlier
loader forced the wall normal to follow a positive-x shear tangent.  That is
not gas-facing on the overhanging FWD/BWD faces.  The corrected loader keeps
the left normal of the stored clockwise SPARTA surface element for bulk
sampling and uses the positive-x tangent only for signed shear.  With this
correction, S1/S2 help pressure but do not solve the shear-transfer problem.

Training: Kn=0.1 and 0.8. Blind testing: Kn=0.2 and 0.4 for BWD/FWD/ISO.

| Target | P | S0 | S1 | S2 | S1 gain | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| CP | 3.20% | 8.87% | 6.11% | 5.84% | 31.15% | [3.83, 43.68]% |
| CF | 26.37% | 35.66% | 43.38% | 45.76% | -21.64% | [-38.09, -5.10]% |

## Checks

- FAIL: `S1_cp_nrmse_below_5_percent`
- FAIL: `S1_cf_nrmse_below_15_percent`
- FAIL: `S1_gain_at_least_20_percent_for_cf`
- FAIL: `cf_gain_ci_excludes_zero`
- FAIL: `S1_beats_parameter_baseline`
