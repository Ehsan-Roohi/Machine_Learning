# ASTR R26 Stage-1 cavity audit

This package implements the conservative first stage of an R26 extension for the
argon lid-driven cavity at `Kn=0.05`, `Tw=300 K`, and lid speeds 10 and 100 m/s.

Stage 1 intentionally does **not** enable the nonlinear source blocks currently
wrapped in `if (.false.) then`. It:

1. Corrects `Apsi1=1.698d9` to `Apsi1=1.698d0`.
2. Activates the existing `moment='r26'` path.
3. Runs an equilibrium residual test.
4. Runs a 32x32, 10 m/s sanity cavity.
5. Runs a 32x32, 100 m/s stability pilot with `dt=2.5e-5`.
6. If all tests pass, runs the 64x64, 100 m/s production pilot to 60,000 steps.
7. Compares Stage-1 R26 against the validated 40k R13 result and manuscript DSMC metrics.

The result must be labelled **audited Maxwell/semi-linear R26 Stage 1**, not a
fully nonlinear VHS-argon R26 solution.
