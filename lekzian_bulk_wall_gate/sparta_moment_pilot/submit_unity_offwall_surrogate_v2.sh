#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/project/pi_roohie_umass_edu/Combustion/LEKZIAN_SPARTA_MOMENT_PILOT}
REPO=${REPO:-$ROOT/Machine_Learning}
PYTHON_BIN=${PYTHON_BIN:-/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu/bin/python}
BRANCH=${BRANCH:-agent/lekzian-gate-test}
LABEL=${LABEL:-offwall_v1}

cd "$REPO"
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

SCRIPT="$REPO/lekzian_bulk_wall_gate/sparta_moment_pilot/train_offwall_bulk_to_wall_v2.py"
OUT="$ROOT/results/offwall_surrogate_v2"
LOG="$ROOT/logs/offwall_surrogate_v2_%j"
mkdir -p "$OUT" "$ROOT/logs"

JOB_ID=$(sbatch --parsable \
  --job-name=lekz_offwall_ml2 \
  --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --mem=32G --time=01:00:00 \
  --output="${LOG}.out" --error="${LOG}.err" \
  --mail-type=END,FAIL \
  --wrap="'$PYTHON_BIN' '$SCRIPT' \
    --train-root '$ROOT/runs/production' \
    --test-root '$ROOT/runs/interpolation' \
    --label '$LABEL' --out '$OUT'")

echo "Submitted target-balanced surrogate job: $JOB_ID"
echo "Monitor: squeue -j $JOB_ID"
echo "Output : $OUT"
