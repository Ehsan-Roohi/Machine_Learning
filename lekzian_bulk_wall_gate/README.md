# Lekzian–Roohi matched-capacity bulk-to-wall Gate Test

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
reader/geometry code found on Unity. It emits only the full six-annulus table;
all radius variants are then produced by masking that single fixed-width table.

## What is matched

All candidates use the same 27 Phase-1 physical cases, fixed full-annulus input
width, neural architecture, number of parameters, optimizer, epoch budget, and
outer folds. Hidden annuli are median-filled (zero after train-only
standardisation) and represented by masks. The configurations are:

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

- project root: `/project/pi_roohie_umass_edu/Combustion`
- Python: `/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu/bin/python`
- working directory: `/project/pi_roohie_umass_edu/Combustion/LEKZIAN_BULK_WALL_GATE`

The launcher first searches for an existing `surface_patch_dataset_full.csv`.
If it is absent, it searches for the original audit and scripts and builds the
full annular table once. Every resolved path and the planned scientific test are
printed before Slurm submission. Ambiguous/missing inputs stop the run instead
of silently selecting a substitute. Explicit paths can be supplied, for example:

```bash
FEATURE_TABLE=/absolute/path/surface_patch_dataset_full.csv bash <(curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/Machine_Learning/agent/lekzian-gate-test/lekzian_bulk_wall_gate/run_unity_gate.sh")
```

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
