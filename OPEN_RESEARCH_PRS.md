# Open R13/R26 research pull requests

Inventory date: 2026-08-18.  These 39 PRs are preserved article-development
records.  Listing does not mean acceptance or merge readiness.  Their code and
evidence remain protected by `JFM_R13_R26_PROTECTION.md`.

## Current audits and independent references

- [#45](https://github.com/Ehsan-Roohi/Machine_Learning/pull/45) — staged no-new-DSMC audits for a PoF redesign
- [#44](https://github.com/Ehsan-Roohi/Machine_Learning/pull/44) — independent steady Rana-2013 R13 reference solver
- [#43](https://github.com/Ehsan-Roohi/Machine_Learning/pull/43) — R13 Kn=0.010 owner/no-mass wall-face extrapolation
- [#42](https://github.com/Ehsan-Roohi/Machine_Learning/pull/42) — Rana reduced-shear D-oracle correction
- [#41](https://github.com/Ehsan-Roohi/Machine_Learning/pull/41) — legacy R13 Rana Kn=0.010 trajectory to 120k
- [#40](https://github.com/Ehsan-Roohi/Machine_Learning/pull/40) — critical R13 fixes on Rana Kn=0.010
- [#39](https://github.com/Ehsan-Roohi/Machine_Learning/pull/39) — untouched initial R13 code against Rana-2013 cases
- [#38](https://github.com/Ehsan-Roohi/Machine_Learning/pull/38) — remaining Rana R13 Kn=0.707 instability

## R13/R26 validation and continuation sequence

- [#37](https://github.com/Ehsan-Roohi/Machine_Learning/pull/37) — clean R26-v3 checkpoint from 11k to 20k
- [#36](https://github.com/Ehsan-Roohi/Machine_Learning/pull/36) — Rana R13 numerical wall path
- [#35](https://github.com/Ehsan-Roohi/Machine_Learning/pull/35) — Maxwell transport and R26-v3 MPI layout benchmark
- [#34](https://github.com/Ehsan-Roohi/Machine_Learning/pull/34) — Maxwell R13 and Rana pseudo-time pilots
- [#33](https://github.com/Ehsan-Roohi/Machine_Learning/pull/33) — Rana-2013 R13 core cases to 100k
- [#32](https://github.com/Ehsan-Roohi/Machine_Learning/pull/32) — nonlinear R26-v3 continuation to 100k milestones
- [#31](https://github.com/Ehsan-Roohi/Machine_Learning/pull/31) — nonlinear R13 continuation from 20k to 100k
- [#30](https://github.com/Ehsan-Roohi/Machine_Learning/pull/30) — nonlinear R13 to matched pseudo-time 0.5
- [#29](https://github.com/Ehsan-Roohi/Machine_Learning/pull/29) — LR13 completeness audit and mislabeled-run block
- [#28](https://github.com/Ehsan-Roohi/Machine_Learning/pull/28) — matched nonlinear R13 20k diagnostic
- [#27](https://github.com/Ehsan-Roohi/Machine_Learning/pull/27) — fresh nonlinear R26-v3 matched 20k diagnostic
- [#26](https://github.com/Ehsan-Roohi/Machine_Learning/pull/26) — first strict-FPE invalid in fresh R26-v3 smoke
- [#25](https://github.com/Ehsan-Roohi/Machine_Learning/pull/25) — paper-exact Rana-2013 nonlinear R13 benchmark
- [#24](https://github.com/Ehsan-Roohi/Machine_Learning/pull/24) — corrected R13 from equilibrium to 100k
- [#23](https://github.com/Ehsan-Roohi/Machine_Learning/pull/23) — exact Rana-2013 nonlinear R13 cavity benchmarks
- [#22](https://github.com/Ehsan-Roohi/Machine_Learning/pull/22) — term-by-term R26 full-bulk closure v3
- [#21](https://github.com/Ehsan-Roohi/Machine_Learning/pull/21) — R13 Delta closure and continuation to 100k
- [#20](https://github.com/Ehsan-Roohi/Machine_Learning/pull/20) — full-bulk R26 formula audit and continuation
- [#19](https://github.com/Ehsan-Roohi/Machine_Learning/pull/19) — nonlinear-source R26 to 40k/50k
- [#18](https://github.com/Ehsan-Roohi/Machine_Learning/pull/18) — semi-linear R26 to 40k/50k
- [#17](https://github.com/Ehsan-Roohi/Machine_Learning/pull/17) — nonlinear-source R26-v1 cavity pilot
- [#16](https://github.com/Ehsan-Roohi/Machine_Learning/pull/16) — semi-linear R26 continuation to convergence
- [#15](https://github.com/Ehsan-Roohi/Machine_Learning/pull/15) — ASTR R26 cavity 10k-to-20k ladder
- [#14](https://github.com/Ehsan-Roohi/Machine_Learning/pull/14) — ASTR R26 2k-step continuation
- [#13](https://github.com/Ehsan-Roohi/Machine_Learning/pull/13) — corrected ASTR R26 short stability gates
- [#11](https://github.com/Ehsan-Roohi/Machine_Learning/pull/11) — ASTR nonlinear R13 cavity to 40k
- [#9](https://github.com/Ehsan-Roohi/Machine_Learning/pull/9) — ASTR nonlinear R13 from scratch to 20k/30k
- [#8](https://github.com/Ehsan-Roohi/Machine_Learning/pull/8) — ASTR nonlinear R13 retry to 30k
- [#7](https://github.com/Ehsan-Roohi/Machine_Learning/pull/7) — ASTR nonlinear R13 continuation to step 30k
- [#6](https://github.com/Ehsan-Roohi/Machine_Learning/pull/6) — ASTR nonlinear R13 JFM cavity U100 Kn0.05
- [#5](https://github.com/Ehsan-Roohi/Machine_Learning/pull/5) — initial ASTR nonlinear R13 cavity build/inspection

## Triage rule

Do not collapse competing formulations or continuation histories into one
branch for cosmetic cleanup.  Close a PR only after a documented successor
preserves its evidence; merge only with the declared numerical gate and status
language intact.
