#!/usr/bin/env bash
set -euo pipefail

if [[ "${LEKZIAN_HALF_RANGE_FINALIZE:-0}" == "1" ]]; then
  "${PYTHON_BIN}" "${VALIDATOR}" "${RUN_ROOT}"
  archive="${WORK_DIR}/LEKZIAN_ISO_HALF_RANGE_LONG_JOB${ARRAY_JOB_ID}.zip"
  relative=(
    half_range_long_manifest.json
    half_range_long_validation.json
    ISO_Ma6_Kn0p1/metadata.json
    ISO_Ma6_Kn0p1/in.half_range_long
    ISO_Ma6_Kn0p1/output/half_range_long
    ISO_Ma6_Kn0p8/metadata.json
    ISO_Ma6_Kn0p8/in.half_range_long
    ISO_Ma6_Kn0p8/output/half_range_long
  )
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
  mpirun -np "${SLURM_NTASKS}" "${SPARTA_BIN}" -in in.half_range_long
  "${PYTHON_BIN}" "${VALIDATOR}" "${RUN_ROOT}" --case "${case_id}"
  exit 0
fi

UNITY_ROOT="${UNITY_ROOT:-/project/pi_roohie_umass_edu/Combustion}"
WORK_DIR="${WORK_DIR:-${UNITY_ROOT}/LEKZIAN_SPARTA_MOMENT_PILOT}"
RUN_ROOT="${WORK_DIR}/runs/production"
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
CASE_LIST="${RUN_ROOT}/half_range_long_case_list.txt"

"${PYTHON_BIN}" "${GENERATOR}" "${RUN_ROOT}" --steps "${HALF_RANGE_STEPS:-5000}" --block-steps "${HALF_RANGE_BLOCK_STEPS:-1000}"

export LEKZIAN_HALF_RANGE_WORKER=1 WORK_DIR RUN_ROOT CASE_LIST VALIDATOR PYTHON_BIN SPARTA_BIN MPI_MODULE
submit="$(sbatch --parsable \
  --job-name=lekz_half_range \
  --partition="${SLURM_PARTITION:-cpu}" \
  --nodes=1 --ntasks=26 --cpus-per-task=1 --mem=120G --time=02:00:00 \
  --array=0-1%2 \
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

cat <<EOF
Submitted half-range array: ${job_id}
Final validation/pack   : ${pack_id}
Cases                   : ${CASE_LIST}
Monitor                 : squeue -j ${job_id},${pack_id}
Logs                    : ${WORK_DIR}/logs/half_range_${job_id}_*.out
Expected archive        : ${WORK_DIR}/LEKZIAN_ISO_HALF_RANGE_LONG_JOB${job_id}.zip
EOF
