# Conditional Stage-2 with the existing Lekzian--Roohi data

## Why Stage-2 is necessary

The completed Stage-1 V2 run was scientifically useful but did not support a
radial-locality claim. `Cp` met the absolute error bounds, `Cq` improved strongly
in LOCO validation, and `Cp` improved strongly in `(Ma,Kn)` pair-out validation.
However, near, far, interleaved, and fixed-permutation controls were statistically
indistinguishable.

The full Stage-1 audit identified two specific causes:

- 36--59% of squared prediction error was concentrated around the protrusion
  apex (`s01` approximately 0.4--0.5);
- full-domain ring-attention entropy was approximately 0.98--1.00, meaning the
  six annuli received almost uniform weight.

This makes a small selected radius insufficient evidence of locality: raw
annular descriptors can be redundant proxies for the same global flow state.

## Prospective Stage-2 question

> After freezing a strong parameter/surface wall surrogate and removing the
> component of every physical bulk descriptor predictable from its inputs, do
> near-wall disturbances provide a significant error reduction that cannot be
> reproduced by far rings or incorrect surface alignment?

No higher-order SPARTA moments, particles, VDFs, or new DSMC cases are required.

## Methodological changes

1. A target-specific structured wall baseline (`B0`) is trained independently.
2. The baseline is frozen before any bulk correction is learned.
3. Five nonphysical sampling/geometry channels (`npts` and ring-coordinate
   statistics) are removed from every annulus.
4. A train-fold-only ridge model predicts each remaining bulk channel from the
   parameter/surface representation. Only the standardized residual remains.
5. Every ring contains three views: conditional residual, excess over the outer
   ring, and adjacent-ring jump.
6. `C0` has the same correction-network capacity but no visible rings. Therefore
   `C0-C_R` measures incremental bulk value rather than added neural capacity.
7. The Stage-1 apex diagnosis is converted into a fixed, prospective Gaussian
   weighting rule centered at `s01=0.5`.

## Strong negative controls

- `C_far_K3`: three farthest rings, matched to the three nearest rings.
- `C_interleaved_K3`: rings 1, 3, and 5.
- `C_shift_R0p5`: the real trained model evaluated after coherent cyclic shifts
  of each held-out case's bulk profile along the surface; predictions are
  averaged over 12 nonzero shifts.
- `C_radialperm_R0p5`: the near-three ring order is permuted at inference and
  averaged over the five nonidentity permutations.

The surface-shift control replaces the fixed train-time permutation used in
Stage-1. It preserves the held-out case's bulk distribution while destroying
the local wall/bulk alignment and cannot be learned as a stable remapping.

## Validation and gate

- 27 LOCO and 9 `(Ma,Kn)` pair-out outer groups;
- one shared physical inner group across all five seeds;
- five-seed ensembles are formed before radius selection and profile error;
- bootstrap units are physical cases for LOCO and `(Ma,Kn)` clusters for
  pair-out;
- 540 target-specific, resumable Slurm tasks.

For primary `Cp` and `Cq` claims, the default gate requires:

- the same absolute accuracy bounds used in Stage-1;
- at least 1 percentage point `C0-C_selected` improvement with a positive lower
  95% paired bound in both validation schemes;
- a significantly positive near-versus-far gain in LOCO and a positive pair-out
  point estimate;
- a significantly positive real-versus-surface-shift gain in LOCO and a
  positive pair-out point estimate;
- at most 20% right-censored selections.

If conditional gain exists but locality/alignment fails, the automatic verdict
is `BULK_INCREMENTAL_BUT_NOT_SPATIALLY_IDENTIFIED`. In that case the paper must
pivot to bulk-state augmentation and may not claim spatial support.

## One-line Unity run

```bash
bash <(curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/Machine_Learning/agent/lekzian-gate-test/lekzian_bulk_wall_gate/run_unity_stage2_conditional.sh")
```

The working directory is
`/project/pi_roohie_umass_edu/Combustion/LEKZIAN_BULK_WALL_STAGE2_CONDITIONAL`.

## Main outputs

- `stage2_decision.txt` and `.json`;
- `stage2_ensemble_summary.csv`;
- `stage2_paired_gains.csv`;
- `stage2_radius_distribution.csv` and `stage2_radius_censoring.csv`;
- `stage2_redundancy_summary.csv`;
- `stage2_ensemble_case_metrics.csv` and surface predictions;
- `stage2_error_summary.pdf` and `.png`.

The permitted term after a full pass is **conditional predictive support**. A
physical or causal horizon is outside what these observability tests establish.
