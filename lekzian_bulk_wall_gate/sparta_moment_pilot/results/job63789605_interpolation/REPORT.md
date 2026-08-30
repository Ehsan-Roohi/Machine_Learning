# Blind intermediate-Kn moment gate

**Verdict: FAIL**

Training: Kn=0.1 and 0.8. Blind testing: Kn=0.2 and 0.4 for BWD/FWD/ISO.

| Target | P | S0 | S1 | S2 | S1 gain | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| CP | 2.99% | 13.58% | 17.15% | 18.98% | -26.25% | [-32.63, -13.67]% |
| CF | 25.91% | 37.03% | 44.31% | 49.26% | -19.66% | [-36.98, -4.59]% |

## Checks

- FAIL: `S1_cp_nrmse_below_5_percent`
- FAIL: `S1_cf_nrmse_below_15_percent`
- FAIL: `S1_gain_at_least_20_percent_for_cf`
- FAIL: `cf_gain_ci_excludes_zero`
- FAIL: `S1_beats_parameter_baseline`
