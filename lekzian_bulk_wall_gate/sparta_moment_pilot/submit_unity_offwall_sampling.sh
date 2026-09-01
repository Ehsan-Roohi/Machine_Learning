#!/usr/bin/env bash
set -euo pipefail

if [[ "${OFFWALL_WORKER:-0}" == 1 ]]; then
  module load "${MPI_MODULE:-openmpi/5.0.3}"
  C="$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$CASE_LIST")"; D="$RUN_ROOT/$C"; cd "$D"
  mpirun -np "$SLURM_NTASKS" "$SPARTA_BIN" -in "in.$OFFWALL_LABEL"
  "$PYTHON_BIN" "$REDUCER" "$D" --label "$OFFWALL_LABEL" --steps "$OFFWALL_STEPS" --block-steps "$OFFWALL_BLOCK_STEPS"
  [[ "${KEEP_RAW:-0}" == 1 ]] || rm -f "$D/output/$OFFWALL_LABEL/particles.gz"
  exit 0
fi

if [[ "${OFFWALL_PACK:-0}" == 1 ]]; then
  archive="$WORK_DIR/LEKZIAN_OFFWALL_${RUN_MODE}_JOB${ARRAY_JOB_ID}.zip"; files=("${OFFWALL_LABEL}_manifest.json" "${OFFWALL_LABEL}_case_list.txt")
  while read -r C; do files+=("$C/metadata.json" "$C/in.$OFFWALL_LABEL" "$C/output/$OFFWALL_LABEL/offwall_incoming_moments.npz" "$C/output/$OFFWALL_LABEL/wall_targets.00001000.dat" "$C/output/$OFFWALL_LABEL/wall_targets.00002000.dat" "$C/output/$OFFWALL_LABEL/wall_targets.00003000.dat" "$C/output/$OFFWALL_LABEL/wall_targets.00004000.dat" "$C/output/$OFFWALL_LABEL/wall_targets.00005000.dat"); done < "$CASE_LIST"
  (cd "$RUN_ROOT" && zip -q -1 "$archive" "${files[@]}"); sha256sum "$archive" | tee "$archive.sha256.txt"; echo "OFFWALL_PACKED=$archive"; exit 0
fi

UNITY_ROOT="${UNITY_ROOT:-/project/pi_roohie_umass_edu/Combustion}"; WORK_DIR="${WORK_DIR:-$UNITY_ROOT/LEKZIAN_SPARTA_MOMENT_PILOT}"
RUN_MODE="${RUN_MODE:-production}"; RUN_ROOT="$WORK_DIR/runs/$RUN_MODE"; OFFWALL_LABEL="${OFFWALL_LABEL:-offwall_half_range}"
PYTHON_BIN="${PYTHON_BIN:-/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu/bin/python}"
REPO_URL="${REPO_URL:-https://github.com/Ehsan-Roohi/Machine_Learning.git}"; GIT_REF="${GIT_REF:-agent/lekzian-gate-test}"; CODE_DIR="$WORK_DIR/Machine_Learning"
[[ -x "$SPARTA_BIN" ]] || { echo "ERROR: invalid SPARTA_BIN" >&2; exit 2; }; [[ -d "$RUN_ROOT" ]] || { echo "ERROR: missing $RUN_ROOT" >&2; exit 2; }
git -C "$CODE_DIR" fetch origin "$GIT_REF"; git -C "$CODE_DIR" checkout "$GIT_REF"; git -C "$CODE_DIR" pull --ff-only origin "$GIT_REF"
PKG="$CODE_DIR/lekzian_bulk_wall_gate/sparta_moment_pilot"; GENERATOR="$PKG/generate_offwall_sampling.py"; REDUCER="$PKG/reduce_offwall_samples.py"; SELF="$PKG/submit_unity_offwall_sampling.sh"
read -r -a CASES <<< "${OFFWALL_CASE_IDS:?set OFFWALL_CASE_IDS}"; OFFWALL_STEPS="${OFFWALL_STEPS:-5000}"; OFFWALL_BLOCK_STEPS="${OFFWALL_BLOCK_STEPS:-1000}"
"$PYTHON_BIN" "$GENERATOR" "$RUN_ROOT" --case-ids "${CASES[@]}" --steps "$OFFWALL_STEPS" --block-steps "$OFFWALL_BLOCK_STEPS" --dump-every "${OFFWALL_DUMP_EVERY:-50}" --label "$OFFWALL_LABEL"
CASE_LIST="$RUN_ROOT/${OFFWALL_LABEL}_case_list.txt"; mkdir -p "$WORK_DIR/logs"
export OFFWALL_WORKER=1 WORK_DIR RUN_MODE RUN_ROOT OFFWALL_LABEL OFFWALL_STEPS OFFWALL_BLOCK_STEPS CASE_LIST PYTHON_BIN SPARTA_BIN MPI_MODULE REDUCER KEEP_RAW
N="$(wc -l < "$CASE_LIST")"; submit="$(sbatch --parsable --job-name=lekz_offwall --partition=cpu --nodes=1 --ntasks=26 --cpus-per-task=1 --mem=120G --time="${OFFWALL_TIME:-04:00:00}" --array="0-$((N-1))%${ARRAY_MAX_PARALLEL:-2}" --output="$WORK_DIR/logs/offwall_%A_%a.out" --error="$WORK_DIR/logs/offwall_%A_%a.err" --export=ALL "$SELF")"; J="${submit%%;*}"
pack="$(sbatch --parsable --job-name=lekz_offwall_pack --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=01:00:00 --dependency="afterok:$J" --output="$WORK_DIR/logs/offwall_pack_%j.out" --error="$WORK_DIR/logs/offwall_pack_%j.err" --export=ALL,OFFWALL_WORKER=0,OFFWALL_PACK=1,ARRAY_JOB_ID="$J" "$SELF")"; P="${pack%%;*}"
echo "Off-wall array: $J"; echo "Pack job: $P"; echo "Monitor: squeue -j $J,$P"; echo "Expected: $WORK_DIR/LEKZIAN_OFFWALL_${RUN_MODE}_JOB${J}.zip"
