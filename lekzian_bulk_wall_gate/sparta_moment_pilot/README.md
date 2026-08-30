# SPARTA moment-sufficiency pilot

This package generates the DSMC data needed to test whether full-range kinetic
moments improve bulk-to-wall reconstruction for the Lakzian protrusion cases.
It is a stock-SPARTA reproduction of the archived Phase-1 parameter slice, not
a particle-level continuation of the original private FPPC runs: the recovered
archives contain final Tecplot/XLSX fields but no SPARTA input or particle
restart.

## Fixed pilot matrix

The production pilot contains exactly six cases:

| Geometry | Mach | Knudsen numbers | `hp/hs` | `Tw/Tinf` |
|---|---:|---:|---:|---:|
| ISO, FWD, BWD | 6 | 0.1, 0.8 | 1.5 | 1 |

The geometry matches the archived surfaces: domain `[-0.1,1] x [0,0.5]` m,
flat plate `x=0:1`, protrusion base `x=0.22:0.24`, height `hp=0.03` m, and
apex `x={0.23,0.21,0.25}` m for `{ISO,FWD,BWD}`.  It also preserves the
original 1040 element IDs: flat elements `1:980`, protrusion `981:1040`.

The original calculation relied on a private fixed-particles-per-cell (FPPC)
source modification over a locally refined grid.  Standard SPARTA has no
equivalent planar arbitrary cell-weight option, so this package uses a uniform
`440 x 200` grid (`dx=dy=0.0025 m`) and targets 20 particles/cell.  That gives
about 1.76 million simulator particles and resolves the smallest pilot mean
free path (`lambda=0.003 m`).  The difference must be disclosed in any paper.

## Outputs

Each case equilibrates for 30,001 steps, saves a particle restart, and produces
four non-overlapping 10,000-step block averages:

- primitive grid fields: particle count, number/mass density, `u,v,w,T`;
- full momentum-flux tensor: `momxx,momyy,momzz,momxy,momyz,momxz`;
- translational energy flux: `heatx,heaty,heatz`;
- surface targets: `press,shx,shy,shz,ke,erot,evib,etot`;
- a short pre/post collision-velocity tally on the protrusion for both ISO
  cases only.

The exact column mapping and all derived physical constants are recorded in
each `metadata.json`; `id+split` uniquely identifies cut-cell subcells.  For
monatomic argon, `erot=evib=0` and `etot=ke` by
construction.  The four blocks are non-overlapping time windows, not four
independent DSMC realizations; their residual autocorrelation must be checked.
After every array case passes validation, a dependent Slurm job aligns the
blocks by `id+split`/surface ID and writes `output/moment_blocks.npz` for each
case plus a root `packed_manifest.json`.

## Unity workflow

The downloaded package is self-contained; extract it on Unity and run commands
from its directory.  It does not require a GitHub clone.

Run a one-case smoke test first.  If SPARTA is already on `PATH`:

```bash
PILOT_MODE=smoke bash submit_unity_moment_pilot.sh
```

If it is not on `PATH`, provide its absolute path (and a module if needed):

```bash
MPI_MODULE=openmpi/5.0.3 SPARTA_BIN=/absolute/path/to/spa_mpi \
PILOT_MODE=smoke bash submit_unity_moment_pilot.sh
```

After the smoke job ends with `PASS ISO_Ma6_Kn0p1`, submit all six cases:

```bash
MPI_MODULE=openmpi/5.0.3 SPARTA_BIN=/absolute/path/to/spa_mpi PILOT_MODE=production \
ARRAY_MAX_PARALLEL=2 bash submit_unity_moment_pilot.sh
```

The launcher defaults to
`/project/pi_roohie_umass_edu/Combustion/LEKZIAN_SPARTA_MOMENT_PILOT`, uses
26 MPI tasks and 120 GB per production case, and runs at most two cases at a
time.  It loads Unity `openmpi/5.0.3`, executes a one-rank smoke test directly,
and uses `mpirun` for production to match the known-good JFM SPARTA build.
Override `WORK_DIR`, `SLURM_PARTITION`, or other environment variables when
needed.  It refuses to overwrite an existing run.  `FORCE_REGENERATE=1` is
intentionally required to discard and regenerate it.

To generate locally without Slurm:

```bash
python generate_cases.py --output /path/to/pilot --mode smoke
cd /path/to/pilot/ISO_Ma6_Kn0p1
/path/to/spa_serial -in in.moment_pilot
python /path/to/validate_outputs.py /path/to/pilot
python /path/to/pack_outputs.py /path/to/pilot
```

## Scientific guardrails

- Do not launch the remaining 21 Phase-1 combinations before this pilot is
  analyzed.
- `press`, shear, and energy flux are targets only; do not feed them into the
  bulk-input model.
- Compare identical models using primitives, then `+Pij`, then `+Pij+qi`.
- Use the collision tally only as a half-range diagnostic upper bound.
- Continue only if `Pij` reduces signed-shear error by at least 20% relative,
  with the gain larger than the DSMC block uncertainty and consistent across
  geometries/Knudsen endpoints.

## Post-run moment-sufficiency gate

After `pack_outputs.py` creates all six `moment_blocks.npz` files, run the
leakage-safe pilot analysis:

```bash
python analyze_moment_gate.py /path/to/runs/production \
  --output /path/to/runs/production/gate_analysis
```

The script averages the four DSMC blocks before fitting, estimates target
uncertainty from their between-block variation, samples bulk fields at
`0.5, 1, 2 lambda` along the local gas-facing wall normal, and compares:

- `S0`: primitives;
- `S1`: primitives plus the locally projected momentum-flux tensor;
- `S2`: `S1` plus locally projected translational energy flux;
- shuffled `S1`: a locality/leakage control.

Every score uses leave-one-complete-geometry-out evaluation.  The primary gate
is the signed-shear NRMSE in the protrusion nearfield, with a case-level
bootstrap interval and the DSMC block standard error reported alongside it.

Audit the conclusion against a regularized linear model, two kinetic horizons,
and leave-one-case-out as well as LOCO:

```bash
python analyze_gate_sensitivity.py /path/to/runs/production \
  --output /path/to/runs/production/gate_analysis/sensitivity
```

For the S3 diagnostic, combine the short ISO collision tallies with the packed
wall targets:

```bash
python analyze_half_range_gate.py /path/to/collision_archive \
  /path/to/runs/production \
  --loco-predictions /path/to/gate_analysis/loco_predictions.npz \
  --output /path/to/gate_analysis/half_range
```

The parser supports both one-snapshot-per-file output and many tally snapshots
appended to a single file.  It reconstructs loads in two ways: exact pre/post
impulse as a control, and incident velocity plus the known diffuse wall kernel
as the leakage-safe half-range diagnostic.

The short pilot used a dump interval of 20, but `surf/collision/tally` is an
instantaneous compute.  If its ten snapshots are too sparse, continue only the
two steady ISO restarts with every-timestep tally output:

```bash
SPARTA_BIN=/absolute/path/to/spa_mpi MPI_MODULE=openmpi/5.0.3 \
bash submit_unity_half_range_continuation.sh
```

The continuation resets only the timestep counter, preserves the steady
particle state, samples 5000 evolved timesteps into one collision file per
case, records five co-temporal wall-target blocks, validates coverage, and
creates a compact ZIP automatically. Existing continuation output is never
overwritten.
