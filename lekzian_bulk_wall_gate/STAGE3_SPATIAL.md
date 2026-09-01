# Spatial Stage 3: existing-data rescue experiment

## The diagnosis

Stage 2 did not identify spatial support. Near, far, interleaved, shifted, and
radially permuted annular summaries produced essentially the same errors. This
does not prove that the resolved DSMC field has no useful spatial information;
it proves that six blocks of mean/min/max/gradient statistics do not preserve
enough location and direction to test that proposition.

There were also two avoidable statistical problems: approximately 705k
parameters were fitted per target from only 27 independent physical cases, and
the legacy peak/apex weighting was effectively applied twice. Stage 3 fixes
both without asking for new DSMC output or higher-order moments.

## What is new

For every existing surface point, `prepare_stage3_spatial.py` constructs a
fixed wall-aligned grid spanning `[-3,3] h_s` tangentially and
`[0.05,3] h_s` normally. It interpolates only the four already archived
macroscopic fields `(u,v,T,logP)` with inverse-distance weighting. The dataset
also stores interpolation distance and a validity flag.

The training code then uses:

1. one shared compact spatial CNN with three output heads (`Cp`, `Cq`, and
   `tau_abs`) and a hard 100,000-parameter ceiling;
2. physics-fixed Mach exponents `(2,3,2)` before train-fold-only target
   standardisation;
3. the original wall weights exactly once, normalised within each training
   fold;
4. nested grouped early stopping and three independent seeds;
5. one architecture for every input mask, so capacity is exactly matched.

No target profile, test-fold statistic, or Stage-2 result is used to fit a
spatial radius.

## Prospective controls

| Configuration | Information shown to the same trained model | Question |
|---|---|---|
| `P0` | no gas patch | context/surface baseline |
| `P_near` | cells within `0.75 h_s` | near-wall value |
| `P_far` | equally many farthest cells | proximity versus sample count |
| `P_upstream` | matched-count globally upstream cells | directional support |
| `P_downstream` | matched-count downstream cells | directional control |
| `P_full` | entire spatial patch | total resolved-field value |
| `P_shift` | full patch from another wall location in the same case | alignment control |
| `P_radial_flip` | normal coordinate reversed at inference | radial-order control |

`P_shift` preserves each held-out case's field distribution while destroying
wall/field registration. It is therefore a substantially stronger negative
control than comparing redundant annular moments.

## Validation and interpretation

The default run contains 27 leave-one-case-out folds and nine leave-one
`(Ma,Kn)`-pair-out folds. Three seeds give 108 resumable multi-target tasks,
one fifth of the Stage-2 task count.

The automatic verdict distinguishes three outcomes:

- `SPATIAL_BULK_SIGNAL_WITH_LOCAL_SUPPORT`: primary targets meet accuracy,
  spatial increment, alignment, and near-versus-far criteria;
- `SPATIAL_BULK_SIGNAL_WITHOUT_LOCAL_SUPPORT`: aligned fields help, but a local
  radius is not identified; the article must be about spatial bulk-state
  augmentation rather than an information horizon;
- `NO_ACTIONABLE_SPATIAL_BULK_SIGNAL`: the current observability claim should
  not be submitted.

Even the strongest passing interpretation is **predictive spatial support**,
not causality or a physical information horizon.

## One-line Unity run

```bash
bash <(curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/Machine_Learning/agent/lekzian-gate-test/lekzian_bulk_wall_gate/run_unity_stage3_spatial.sh")
```

The launcher finds the existing audit and legacy readers, builds the spatial
NPZ once, runs tests, and submits 108 resumable GPU tasks. It never invokes
SPARTA. A prebuilt dataset can be supplied explicitly:

```bash
DATASET=/absolute/path/stage3_spatial_dataset_phase1.npz bash <(curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/Machine_Learning/agent/lekzian-gate-test/lekzian_bulk_wall_gate/run_unity_stage3_spatial.sh")
```

The final decision is written to:

```text
/project/pi_roohie_umass_edu/Combustion/LEKZIAN_BULK_WALL_STAGE3_SPATIAL/results/final/stage3_decision.txt
```

The accompanying CSV files retain all individual-seed and ensemble surface
predictions, case-level errors, grouped-bootstrap intervals, and paired gains.
