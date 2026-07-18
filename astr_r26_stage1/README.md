# ASTR R26 Stage-1 short-gate audit

This package implements a conservative first executable audit of the existing
ASTR `mom_merge` R26 path for the lid-driven cavity at `Kn=0.05`, `Tw=300 K`,
and lid speeds 10 and 100 m/s.

## Important corrections

1. `Apsi1=1.698d9` is changed to `Apsi1=1.698d0`.
2. All generated real values use Fortran `D` exponents because ASTR's legacy
   string parser silently misreads `E` notation.
3. The top wall is set to full speed from the first step so a short run is a
   genuine U=10 or U=100 m/s stability test rather than a low-speed ramp test.
4. Existing R26 nonlinear source blocks wrapped in `if (.false.) then` remain
   disabled and are explicitly reported.

## Short-gate ladder

- 16x16, U=10 m/s, 100 steps
- 16x16, U=100 m/s, 200 steps
- 32x32, U=100 m/s, 2000 steps

Each gate checks compilation, HDF5 completeness, all R26 fields, finite values,
positive density/pressure/temperature, conservative bounds, and nonzero wall-
driven motion. No long 64x64 run is launched until these gates pass.

Results must be labelled **audited Maxwell/semi-linear R26 Stage 1**, not fully
nonlinear VHS-argon R26.
