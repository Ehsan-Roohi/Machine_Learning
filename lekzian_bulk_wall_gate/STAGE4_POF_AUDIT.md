# Stage 4 PoF representation audit

## Status and venue

This stage supports a possible **Physics of Fluids** redesign. It is explicitly
exploratory and cannot replace, weaken, or reinterpret the locked prospective
Stage-3 verdict `NO_ACTIONABLE_SPATIAL_BULK_SIGNAL`.

No new DSMC/SPARTA simulation and no higher-order moment are used.

## Scientific question

Stage 3 found accurate `Cp` and `Cq` profiles, but only a small full-field gain
for `Cp`, no field gain for `Cq`, and an inaccurate yet field-sensitive shear
target. Near did not beat far. Stage 4 asks the narrower representation
question needed for a defensible PoF paper:

> Does the remaining gain require the two-dimensional wall-aligned bulk field,
> or can patch statistics, case-wide information, or one channel explain it?

The same model, parameter ceiling, grouped folds, nested stopping, training
schedule, seeds, target scaling, and dataset are retained. No hyperparameter is
tuned after observing Stage 3.

The repeated `A0/A_full` means must reproduce frozen Stage-3 `P0/P_full` means
within 0.50 percentage points. Failure yields `STAGE4_REPLICATION_FAILURE` and
blocks every scientific interpretation of the new controls.

## Controls

- `A0`: context only.
- `A_full`: intact wall-aligned patch.
- `A_patchmean`: retains the four local patch means but destroys their 2-D
  arrangement.
- `A_cellperm`: preserves every physical-channel marginal while jointly
  permuting cells.
- `A_casepool`: broadcasts the case-average patch to every wall point.
- `A_surfaceperm`: preserves the within-case patch multiset but destroys wall
  alignment.
- `A_no_u`, `A_no_v`, `A_no_T`, `A_no_logP`: mean-fill one standardized
  physical channel at a time.

Distance and validity channels are held intact in the representation controls,
so an error change cannot be attributed merely to a different interpolation
support pattern.

## Validation and decision

The audit repeats 27 LOCO and 9 `(Ma,Kn)` pair-out folds with seeds
`101,202,303`. It reports physical-case bootstrap intervals and subgroup gains
for `Ma`, `Kn`, and geometry.

For each target, a control is called robust only when the LOCO paired lower 95%
bound is positive and the pair-out mean has the same sign. The classifications
are:

1. `STRUCTURED_WALL_ALIGNED_SUPPORT`: full field beats context, patch-mean,
   cell-permuted, and surface-permuted controls.
2. `PATCH_STATISTICS_SUFFICIENT`: field information helps, but destroying the
   2-D arrangement does not hurt.
3. `PARTIAL_OR_UNSTABLE_SPATIAL_SUPPORT`: the controls disagree.
4. `NO_INCREMENTAL_FIELD_SUPPORT`: full does not robustly beat context.

`POF_REFRAME_CANDIDATE` additionally requires accuracy-qualified structured
support for `Cp`. This is a manuscript triage rule, not a retroactive Stage-3
pass. Allowed language is *quantity-dependent predictive representation
support*. A causal information horizon or physical observability radius remains
forbidden.

## Unity run

```bash
bash <(curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/Machine_Learning/agent/lekzian-gate-test/lekzian_bulk_wall_gate/run_unity_stage4_pof_audit.sh")
```

The launcher requires the completed Stage-3 dataset and decision, runs the new
tests, and submits 108 resumable GPU tasks. Final outputs include:

- `stage4_pof_decision.txt` and `.json`;
- `stage4_ensemble_summary.csv`;
- `stage4_paired_gains.csv`;
- `stage4_regime_gains.csv`;
- `stage4_stage3_replication_audit.csv`;
- individual- and ensemble-level surface/case predictions;
- `stage4_pof_representation_audit.pdf` and `.png`.
