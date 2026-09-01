# Lekzian–Roohi Stage-1 V2 for a new Physics of Fluids manuscript

## Scientific scope

Stage-1 V2 deliberately uses only the existing six-annulus macroscopic DSMC
table. It does **not** assume that SPARTA stress, heat-flux, higher-order moment,
particle, or velocity-distribution outputs are available.

The claim under test is:

> Under grouped physical-case validation, real near-wall macroscopic descriptors
> provide an accuracy-qualified and quantity-resolved predictive benefit over a
> matched parameter/surface baseline and matched-count nonlocal controls.

The allowed term is **predictive spatial support**. A physical or causal
information horizon is not claimed.

`Cp` and `Cq` are the primary bulk-support quantities. `|tau|` is retained as a
quantity-dependence contrast; a significant bulk gain is not assumed for shear.

## Why this is a new test

The first Gate Test correctly exposed that a full 198-descriptor MLP could be
statistically harder to fit than a finite-radius model even with the same number
of weights. Stage-1 V2 removes that active-dimension confound:

1. Each quantity has an independent model and early-stopping path.
2. The `M0` path retains the structured branch/trunk and windward/leeward gate
   of the manuscript's strong direct-wall surrogate.
3. Every annulus uses the same shared encoder and fixed latent width.
4. One mask-conditioned model generates all real radius variants.
5. All five inner-validation predictions are ensembled before radius selection.
6. Five held-out predictions are ensembled before profile errors are computed.
7. Pair-out confidence intervals use `(Ma,Kn)` groups as bootstrap units.

## Controls

The main locality comparison uses exactly three active annuli in every model:

- `M_R0p5`: the three nearest annuli;
- `M_far_K3`: the three farthest annuli;
- `M_interleaved_K3`: annuli 1, 3, and 5.

The parameter count and ring bottleneck are identical because these predictions
come from one model. A second model starts from identical weights but is trained
after coherent bulk profiles are reassigned across physical cases within each
geometry. It produces `M_permtrain_R0p5` and `M_permtrain_full` controls.

## Grouped validation and predeclared rules

- 27 leave-one-physical-case-out folds;
- 9 leave-one-`(Ma,Kn)`-pair-out folds, with all orientations held together;
- five seeds: `101,202,303,404,505`;
- 540 target-specific, resumable Slurm tasks;
- grouped bootstrap intervals use 27 cases for LOCO and 9 `(Ma,Kn)` clusters for
  pair-out.

Default absolute criteria:

- LOCO: upper 95% interval at most 10% for `Cp/Cq`, 20% for `|tau|`;
- pair-out stress test: mean error at most 15% for `Cp/Cq`, 25% for `|tau|`;
- primary `Cp/Cq` bulk gain: at least two percentage points with a positive
  lower 95% paired bound;
- the near-three descriptors must beat the far-three descriptors;
- real alignment must beat case-profile-permuted training.

The selected support is the smallest finite radius that is both below the
inner-validation absolute tolerance and within one percentage point of the
best inner-validation configuration. If no finite radius qualifies, the fold
is recorded as `M_full` and explicitly right-censored. A primary claim also
requires at most 20% censored outer groups in both validation schemes.

These Stage-1 V2 rules are prospective. The earlier V1 run is treated as the
diagnostic experiment that motivated them, not as confirmatory evidence.

## One-line Unity submission

```bash
bash <(curl -fsSL "https://raw.githubusercontent.com/Ehsan-Roohi/Machine_Learning/agent/lekzian-gate-test/lekzian_bulk_wall_gate/run_unity_stage1_v2.sh")
```

The launcher reuses the existing table at
`/project/pi_roohie_umass_edu/Combustion/LEKZIAN_BULK_WALL_GATE/features/surface_patch_dataset_full_gate.csv`.
No DSMC extraction or new SPARTA run is performed.

## Key outputs

- `stage1_v2_decision.txt` and `.json`: automatic publication gate;
- `ensemble_summary.csv`: ensemble-first grouped errors and confidence intervals;
- `ensemble_paired_gains.csv`: matched paired gains and controls;
- `ensemble_case_metrics.csv`: held-out profile metrics by physical case;
- `selected_radius_distribution.csv`: inner-fold radius selections;
- `selected_radius_censoring.csv`: explicit right-censoring rates;
- `individual_seed_attention_profiles.csv`: target-specific annular attention;
- `stage1_v2_error_summary.pdf`: publication-oriented comparison plot.

Scientifically, this must be a substantially new manuscript rather than a
cosmetic revision of the rejected horizon paper. Procedurally, Physics of
Fluids requires a response letter even for an initial submission that was
previously declined. The submission should therefore disclose the earlier
manuscript and map every referee objection to the new matched controls,
ensemble-first validation, absolute-accuracy gate, and censoring rule. A stronger
"quantity-dependent" claim should be used only if the Stage-1 results show a
clear between-quantity contrast; otherwise the paper remains quantity-resolved.
