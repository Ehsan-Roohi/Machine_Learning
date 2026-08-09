# Lekzian–Roohi matched-capacity bulk-to-wall Gate Test

## Stage 4: Physics of Fluids representation audit

Stage 3 is complete and its prospective verdict remains locked as
`NO_ACTIONABLE_SPATIAL_BULK_SIGNAL`. The next calculation is therefore not a
larger model and does not lower the original threshold. [`STAGE4_POF_AUDIT.md`](STAGE4_POF_AUDIT.md)
is an explicitly exploratory falsification audit for a possible **Physics of
Fluids** redesign.

It repeats the same folds, seeds, compact operator, preprocessing, and training
schedule while comparing the intact field with patch-mean, cell-permuted,
case-pooled, surface-permuted, and individual-channel-ablation controls. It also
localizes paired gains across Mach number, Knudsen number, and geometry.

It uses the completed Stage-3 dataset and performs **zero new DSMC/SPARTA
simulations**:

```bash
bash <(curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/Machine_Learning/agent/lekzian-gate-test/lekzian_bulk_wall_gate/run_unity_stage4_pof_audit.sh")
```

The only positive manuscript outcome is an accuracy-qualified, structured,
wall-aligned `Cp` signal. Even then, the allowed claim is quantity-dependent
predictive representation support—not a causal information horizon or physical
observability radius.

## Spatial Stage 3: the no-new-DSMC rescue experiment

The completed Conditional Stage 2 showed that six annular summary blocks were
too redundant to identify proximity or correct wall alignment. The next test
therefore preserves the existing field's two-dimensional structure instead of
adding moments or simulations. [`STAGE3_SPATIAL.md`](STAGE3_SPATIAL.md) samples
wall-aligned patches of the already archived `(u,v,T,logP)` fields and trains
one compact, capacity-matched three-target operator with near/far,
upstream/downstream, surface-shift, and radial-flip controls.

It performs **zero new SPARTA/DSMC runs** and needs **no higher-order moments**:

```bash
bash <(curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/Machine_Learning/agent/lekzian-gate-test/lekzian_bulk_wall_gate/run_unity_stage3_spatial.sh")
```

The design uses 108 resumable GPU tasks (`36 folds × 3 seeds`) and a hard
100,000-parameter ceiling. Its automatic decision explicitly separates a local
predictive-support result from a useful but nonlocal spatial-augmentation
result, preventing another overclaim.

## Conditional Stage-2 after the completed V2 diagnosis

The full Stage-1 V2 audit showed nearly uniform six-ring attention and no
significant near-versus-far or real-versus-permuted advantage. The follow-up is
therefore not a larger end-to-end network. [`STAGE2_CONDITIONAL.md`](STAGE2_CONDITIONAL.md)
freezes an independently trained wall baseline, removes base-predictable bulk
content, adds outer/adjacent annular contrasts, excludes nonphysical ring-count
channels, emphasizes the Stage-1-identified apex region, and evaluates cyclic
surface-alignment and radial-order controls.

It still uses the existing Phase-1 table and requires no new SPARTA output:

```bash
bash <(curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/Machine_Learning/agent/lekzian-gate-test/lekzian_bulk_wall_gate/run_unity_stage2_conditional.sh")
```

Stage-2 is the final existing-data test of a spatial-support interpretation. If
conditional bulk gain remains nonlocal, the manuscript must pivot to bulk-state
augmentation rather than weaken the prospective criteria.

## Stage-1 V2: existing-data route for a new Physics of Fluids paper

The completed first Gate Test is retained as an audit trail. Its five-seed
surface predictions showed that ensembling materially changes the conclusion,
while the direct 198-feature `M_full` input still confounds spatial support with
active statistical dimension.

The prospective existing-data test is now [`STAGE1_V2.md`](STAGE1_V2.md). It
uses independent quantity-specific models, the original strong structured
direct-wall path for `M0`, a shared fixed-dimensional annular encoder,
ensemble-first inner-fold radius selection with explicit censoring,
matched-count near/far/interleaved controls, and coherent case-profile
permutation. It does not require higher-order SPARTA moments or new DSMC
simulations.

Run it on Unity with:

```bash
bash <(curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/Machine_Learning/agent/lekzian-gate-test/lekzian_bulk_wall_gate/run_unity_stage1_v2.sh")
```

The target interpretation is an accuracy-qualified, model-conditional
**predictive spatial support** for a substantially new Physics of Fluids
manuscript—not a cosmetic revision of the rejected horizon claim. A future PoF
submission must still disclose the declined paper and include the journal's
required response letter.

This package answers the methodological objection raised for manuscript
`PF#POF26-AR-10014R-Appeal`: the old ExtraTrees parameter-only baseline and the
4–5% parameter-conditioned wall surrogate did not use matched capacity,
features, data blocks, or validation.

## Located legacy codes

The manuscript archive contains the following relevant sources. They are used
as provenance/reference, not as evidence that the old comparison was fair:

- `16_protrusion_nonlocal_bulk_to_wall_footprint.py`: original ExtraTrees
  descriptor extraction and relative `R95` definition.
- `22_protrusion_R95_robustness_controls.py`: second-regressor and threshold/
  reference sensitivity controls.
- `08_protrusion_train_surface_only_operator_v2.py`: the strong direct-wall
  neural surrogate that exposed the capacity mismatch.
- `18_protrusion_closed_loop_observability.py`: raw-versus-reconstructed-field
  closed-loop comparison.
- `10_protrusion_train_smooth_field_operator_v4_geomfix.py`: DSMC field reader
  used during descriptor reconstruction.

`prepare_gate_features.py` reuses the original `16`, `10`, and `06` surface
reader/geometry code found on Unity. It also embeds the exact case features,
surface features, and peak/apex weights used by the strong direct-wall operator.
It emits only the full six-annulus table; all radius variants are then produced
by masking that single fixed-width table.

## What is matched

All candidates use the same 27 Phase-1 physical cases, the strong surface-only
branch/trunk/gated-decoder architecture, its original engineered case/surface
features and weights, fixed full-annulus input width, number of parameters,
optimizer, epoch budget, and outer folds. Hidden annuli are median-filled (zero
after train-only standardisation) and represented by masks. The configurations are:

- `M0`: parameters, geometry, and surface location only.
- `M_shuffled`: full bulk descriptor marginals with their physical alignment
  destroyed; this is the capacity/representation negative control.
- `M_R0p1` … `M_R2`: real annular descriptors revealed cumulatively.
- `M_full`: every real annular descriptor.

Validation is both leave-one-physical-case-out (27 folds) and
leave-one-`(Ma,Kn)`-pair-out (9 folds, keeping all three orientations together),
with five random seeds and nested group early stopping. Confidence intervals
bootstrap physical cases—not trees and not surface points.

## One-line Unity submission

```bash
bash <(curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/Machine_Learning/agent/lekzian-gate-test/lekzian_bulk_wall_gate/run_unity_gate.sh")
```

Established defaults:

- working/output root: `/project/pi_roohie_umass_edu/Combustion`
- legacy DSMC/audit root: `/project/pi_roohie_umass_edu/Sabouri`
- Python: `/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu/bin/python`
- working directory: `/project/pi_roohie_umass_edu/Combustion/LEKZIAN_BULK_WALL_GATE`

The launcher first searches for an existing `surface_patch_dataset_full_gate.csv`.
If it is absent, it searches for the original audit and scripts and builds the
full annular table once. Every resolved path and the planned scientific test are
printed before Slurm submission. Missing inputs stop the run instead of silently
selecting a substitute. Explicit paths can be supplied, for example:

```bash
FEATURE_TABLE=/absolute/path/surface_patch_dataset_full_gate.csv bash <(curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/Machine_Learning/agent/lekzian-gate-test/lekzian_bulk_wall_gate/run_unity_gate.sh")
```

The full calculation is split into 180 resumable Slurm-array tasks
(`36 outer folds × 5 seeds`), each running all eight matched configurations.
Four GPUs are used concurrently by default; set `ARRAY_MAX_PARALLEL` to a
quota-appropriate value to change this.

## Predeclared gate

The bulk-footprint claim passes only if, in both validation schemes and for all
three targets:

1. The upper 95% physical-case bootstrap bound for `M_full` is at most 10% for
   `Cp` and `Cq`, and 20% for `|tau|`.
2. `M_full` improves on `M0` by at least two percentage points and the lower 95%
   bound of the paired gain is positive.
3. The lower 95% bound of `M_shuffled - M_full` is positive.

An accuracy-qualified radius is reported only when its mean error satisfies the
absolute target tolerance **and** lies within two percentage points of
`M_full`. Otherwise it is censored as `not_reached`.

The allowed interpretation after a pass is **predictive spatial footprint**.
This code never labels the result a physical or causal information horizon.

## Key outputs

- `gate_decision.txt` / `gate_decision.json`: automatic verdict and action.
- `summary_metrics.csv`: case-bootstrap errors for every model/configuration.
- `paired_gains.csv`: paired `M0` and shuffled-control gains over `M_full`.
- `adequacy_radius.csv`: absolute-accuracy-qualified spatial footprint.
- `case_metrics.csv`: audit-ready per-case errors.
- `surface_predictions.csv`: held-out surface predictions for profile checks.
- `gate_error_summary.pdf`: matched-capacity comparison figure.

The run is resumable. Re-running the one-line command continues unfinished
fold/configuration/seed tasks from `completed_tasks.txt`.
