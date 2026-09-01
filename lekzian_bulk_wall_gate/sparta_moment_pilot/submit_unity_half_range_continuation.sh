#!/usr/bin/env bash
set -euo pipefail

if [[ "${LEKZIAN_HALF_RANGE_FINALIZE:-0}" == "1" ]]; then
  "${PYTHON_BIN}" "${VALIDATOR}" "${RUN_ROOT}" --label "${HALF_RANGE_LABEL}"
  suffix=""
  [[ "${HALF_RANGE_LABEL}" == "half_range_long" ]] || suffix="_${HALF_RANGE_LABEL}"
  archive="${WORK_DIR}/LEKZIAN_HALF_RANGE_${RUN_MODE}${suffix}_JOB${ARRAY_JOB_ID}.zip"
  relative=("${HALF_RANGE_LABEL}_manifest.json" "${HALF_RANGE_LABEL}_validation.json" "${HALF_RANGE_LABEL}_case_list.txt")
  while read -r case_id; do
    relative+=("${case_id}/metadata.json" "${case_id}/in.${HALF_RANGE_LABEL}" "${case_id}/output/${HALF_RANGE_LABEL}")
  done < "${CASE_LIST}"
  (cd "${RUN_ROOT}" && zip -q -1 -r "${archive}" "${relative[@]}")
  sha256sum "${archive}" | tee "${archive}.sha256.txt"
  echo "HALF_RANGE_LONG_VALIDATED_AND_PACKED=${archive}"
  exit 0
fi

if [[ "${LEKZIAN_HALF_RANGE_WORKER:-0}" == "1" ]]; then
  module load "${MPI_MODULE:-openmpi/5.0.3}"
  case_id="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${CASE_LIST}")"
  [[ -n "${case_id}" ]] || { echo "ERROR: missing array case" >&2; exit 2; }
  case_dir="${RUN_ROOT}/${case_id}"
  cd "${case_dir}"
  echo "Running half-range continuation ${case_id}"
  echo "SPARTA_BIN=${SPARTA_BIN}"
  mpirun -np "${SLURM_NTASKS}" "${SPARTA_BIN}" -in "in.${HALF_RANGE_LABEL}"
  "${PYTHON_BIN}" "${VALIDATOR}" "${RUN_ROOT}" --case "${case_id}" --label "${HALF_RANGE_LABEL}"
  exit 0
fi

UNITY_ROOT="${UNITY_ROOT:-/project/pi_roohie_umass_edu/Combustion}"
WORK_DIR="${WORK_DIR:-${UNITY_ROOT}/LEKZIAN_SPARTA_MOMENT_PILOT}"
RUN_MODE="${RUN_MODE:-production}"
HALF_RANGE_LABEL="${HALF_RANGE_LABEL:-half_range_long}"
RUN_ROOT="${WORK_DIR}/runs/${RUN_MODE}"
REPO_URL="${REPO_URL:-https://github.com/Ehsan-Roohi/Machine_Learning.git}"
GIT_REF="${GIT_REF:-agent/lekzian-gate-test}"
CODE_DIR="${WORK_DIR}/Machine_Learning"
PYTHON_BIN="${PYTHON_BIN:-/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu/bin/python}"
SPARTA_BIN="${SPARTA_BIN:-}"

[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: invalid PYTHON_BIN=${PYTHON_BIN}" >&2; exit 2; }
[[ -x "${SPARTA_BIN}" ]] || { echo "ERROR: set SPARTA_BIN to executable spa_mpi" >&2; exit 2; }
[[ -d "${RUN_ROOT}" ]] || { echo "ERROR: missing production root ${RUN_ROOT}" >&2; exit 2; }

mkdir -p "${WORK_DIR}/logs"
if [[ -d "${CODE_DIR}/.git" ]]; then
  git -C "${CODE_DIR}" fetch origin "${GIT_REF}"
  git -C "${CODE_DIR}" checkout "${GIT_REF}"
  git -C "${CODE_DIR}" pull --ff-only origin "${GIT_REF}"
else
  git clone --depth 1 --branch "${GIT_REF}" "${REPO_URL}" "${CODE_DIR}"
fi

PACKAGE_DIR="${CODE_DIR}/lekzian_bulk_wall_gate/sparta_moment_pilot"
GENERATOR="${PACKAGE_DIR}/generate_half_range_continuation.py"
VALIDATOR="${PACKAGE_DIR}/validate_half_range_continuation.py"
SUBMITTER="${PACKAGE_DIR}/submit_unity_half_range_continuation.sh"
CASE_LIST="${RUN_ROOT}/${HALF_RANGE_LABEL}_case_list.txt"

case_args=()
if [[ -n "${HALF_RANGE_CASE_IDS:-}" ]]; then
  read -r -a selected_cases <<< "${HALF_RANGE_CASE_IDS}"
  case_args=(--case-ids "${selected_cases[@]}")
fi
"${PYTHON_BIN}" "${GENERATOR}" "${RUN_ROOT}" --steps "${HALF_RANGE_STEPS:-5000}" --block-steps "${HALF_RANGE_BLOCK_STEPS:-1000}" --label "${HALF_RANGE_LABEL}" "${case_args[@]}"

export LEKZIAN_HALF_RANGE_WORKER=1 WORK_DIR RUN_MODE RUN_ROOT CASE_LIST VALIDATOR PYTHON_BIN SPARTA_BIN MPI_MODULE HALF_RANGE_LABEL
case_count="$(wc -l < "${CASE_LIST}")"
array_end="$((case_count - 1))"
submit="$(sbatch --parsable \
  --job-name=lekz_half_range \
  --partition="${SLURM_PARTITION:-cpu}" \
  --nodes=1 --ntasks=26 --cpus-per-task=1 --mem=120G --time="${HALF_RANGE_TIME:-02:00:00}" \
  --array="0-${array_end}%${ARRAY_MAX_PARALLEL:-2}" \
  --output="${WORK_DIR}/logs/half_range_%A_%a.out" \
  --error="${WORK_DIR}/logs/half_range_%A_%a.err" \
  --export=ALL \
  "${SUBMITTER}")"
job_id="${submit%%;*}"

final="$(sbatch --parsable \
  --job-name=lekz_half_pack \
  --partition="${SLURM_PARTITION:-cpu}" \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time=01:00:00 \
  --dependency="afterok:${job_id}" \
  --output="${WORK_DIR}/logs/half_range_pack_%j.out" \
  --error="${WORK_DIR}/logs/half_range_pack_%j.err" \
  --export=ALL,LEKZIAN_HALF_RANGE_WORKER=0,LEKZIAN_HALF_RANGE_FINALIZE=1,ARRAY_JOB_ID="${job_id}" \
  "${SUBMITTER}")"
pack_id="${final%%;*}"
archive_suffix=""
[[ "${HALF_RANGE_LABEL}" == "half_range_long" ]] || archive_suffix="_${HALF_RANGE_LABEL}"
expected_archive="${WORK_DIR}/LEKZIAN_HALF_RANGE_${RUN_MODE}${archive_suffix}_JOB${job_id}.zip"

cat <<EOF
Submitted half-range array: ${job_id}
Final validation/pack   : ${pack_id}
Cases                   : ${CASE_LIST}
Monitor                 : squeue -j ${job_id},${pack_id}
Logs                    : ${WORK_DIR}/logs/half_range_${job_id}_*.out
Expected archive        : ${expected_archive}
EOF
