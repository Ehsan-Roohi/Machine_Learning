#!/usr/bin/env bash
set -euo pipefail

# This file is the login-node submitter, Slurm array worker, and final packer.
# It deliberately submits a one-case smoke test by default.

if [[ "${LEKZIAN_SPARTA_FINALIZE:-0}" == "1" ]]; then
  "${PYTHON_BIN}" "${VALIDATOR}" "${RUN_ROOT}"
  "${PYTHON_BIN}" "${PACKER}" "${RUN_ROOT}"
  echo "MOMENT PILOT VALIDATED AND PACKED: ${RUN_ROOT}/packed_manifest.json"
  exit 0
fi

if [[ "${LEKZIAN_SPARTA_WORKER:-0}" == "1" ]]; then
  if [[ -n "${SPARTA_MODULE:-}" ]]; then
    module load "${SPARTA_MODULE}"
  fi

  if [[ -z "${SPARTA_BIN:-}" ]]; then
    for candidate in spa_mpi spa_kk sparta spa_serial; do
      if command -v "${candidate}" >/dev/null 2>&1; then
        SPARTA_BIN="$(command -v "${candidate}")"
        break
      fi
    done
  fi
  if [[ -z "${SPARTA_BIN:-}" || ! -x "${SPARTA_BIN}" ]]; then
    echo "ERROR: SPARTA executable was not found." >&2
    echo "Set SPARTA_BIN=/absolute/path/to/spa_mpi (and optionally SPARTA_MODULE=name)." >&2
    exit 2
  fi

  case_id="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${CASE_LIST}")"
  if [[ -z "${case_id}" ]]; then
    echo "ERROR: no case for array index ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
  fi
  case_dir="${RUN_ROOT}/${case_id}"
  cd "${case_dir}"

  echo "Running ${case_id}"
  echo "SPARTA_BIN=${SPARTA_BIN}"
  echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}"
  bin_name="$(basename "${SPARTA_BIN}")"
  if [[ "${SPARTA_MPI:-auto}" == "1" || ( "${SPARTA_MPI:-auto}" == "auto" && "${bin_name}" =~ (mpi|_kk) ) ]]; then
    srun --ntasks="${SLURM_CPUS_PER_TASK:-1}" "${SPARTA_BIN}" -in in.moment_pilot
  else
    "${SPARTA_BIN}" -in in.moment_pilot
  fi

  "${PYTHON_BIN}" "${VALIDATOR}" "${RUN_ROOT}" --case "${case_id}"
  exit 0
fi

UNITY_ROOT="${UNITY_ROOT:-/project/pi_roohie_umass_edu/Combustion}"
WORK_DIR="${WORK_DIR:-${UNITY_ROOT}/LEKZIAN_SPARTA_MOMENT_PILOT}"
REPO_URL="${REPO_URL:-https://github.com/Ehsan-Roohi/Machine_Learning.git}"
GIT_REF="${GIT_REF:-agent/lekzian-gate-test}"
CODE_DIR="${WORK_DIR}/Machine_Learning"
PILOT_MODE="${PILOT_MODE:-smoke}"
PYTHON_BIN="${PYTHON_BIN:-/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu/bin/python}"
ARRAY_MAX_PARALLEL="${ARRAY_MAX_PARALLEL:-2}"

if [[ "${PILOT_MODE}" != "smoke" && "${PILOT_MODE}" != "production" ]]; then
  echo "ERROR: PILOT_MODE must be smoke or production" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python 3 was not found" >&2
  exit 2
fi

mkdir -p "${WORK_DIR}" "${WORK_DIR}/logs"
LOCAL_PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${LOCAL_PACKAGE_DIR}/generate_cases.py" && "${FETCH_REPO:-0}" != "1" ]]; then
  PACKAGE_DIR="${LOCAL_PACKAGE_DIR}"
else
  if [[ -d "${CODE_DIR}/.git" ]]; then
    git -C "${CODE_DIR}" fetch origin "${GIT_REF}"
    git -C "${CODE_DIR}" checkout "${GIT_REF}"
    git -C "${CODE_DIR}" pull --ff-only origin "${GIT_REF}"
  else
    git clone --depth 1 --branch "${GIT_REF}" "${REPO_URL}" "${CODE_DIR}"
  fi
  PACKAGE_DIR="${CODE_DIR}/lekzian_bulk_wall_gate/sparta_moment_pilot"
fi

GENERATOR="${PACKAGE_DIR}/generate_cases.py"
VALIDATOR="${PACKAGE_DIR}/validate_outputs.py"
PACKER="${PACKAGE_DIR}/pack_outputs.py"
SUBMITTER="${PACKAGE_DIR}/submit_unity_moment_pilot.sh"
RUN_ROOT="${WORK_DIR}/runs/${PILOT_MODE}"
CASE_LIST="${RUN_ROOT}/case_list.txt"

if [[ -e "${RUN_ROOT}/manifest.json" && "${FORCE_REGENERATE:-0}" != "1" ]]; then
  echo "ERROR: ${RUN_ROOT} already exists; existing DSMC outputs were not touched." >&2
  echo "Use a new WORK_DIR, or set FORCE_REGENERATE=1 only if deletion is intentional." >&2
  exit 2
fi

generate_args=(--output "${RUN_ROOT}" --mode "${PILOT_MODE}")
if [[ "${FORCE_REGENERATE:-0}" == "1" ]]; then
  generate_args+=(--force)
fi
"${PYTHON_BIN}" "${GENERATOR}" "${generate_args[@]}"
find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | LC_ALL=C sort > "${CASE_LIST}"
case_count="$(wc -l < "${CASE_LIST}")"
if [[ "${PILOT_MODE}" == "smoke" && "${case_count}" -ne 1 ]]; then
  echo "ERROR: smoke manifest must contain one case" >&2
  exit 3
fi
if [[ "${PILOT_MODE}" == "production" && "${case_count}" -ne 6 ]]; then
  echo "ERROR: production manifest must contain six cases" >&2
  exit 3
fi

export LEKZIAN_SPARTA_WORKER=1 RUN_ROOT CASE_LIST VALIDATOR PACKER PYTHON_BIN
array_end="$((case_count - 1))"
if [[ "${PILOT_MODE}" == "smoke" ]]; then
  resources=(--cpus-per-task=1 --mem=4G --time=00:20:00 --array=0)
else
  resources=(--cpus-per-task=26 --mem=120G --time=24:00:00 --array="0-${array_end}%${ARRAY_MAX_PARALLEL}")
fi

submit_output="$(sbatch --parsable \
  --job-name="lekz_${PILOT_MODE}" \
  --partition="${SLURM_PARTITION:-cpu}" \
  --output="${WORK_DIR}/logs/${PILOT_MODE}_%A_%a.out" \
  --error="${WORK_DIR}/logs/${PILOT_MODE}_%A_%a.err" \
  --export=ALL \
  "${resources[@]}" \
  "${SUBMITTER}")"
job_id="${submit_output%%;*}"

finalize_output="$(sbatch --parsable \
  --job-name="lekz_${PILOT_MODE}_pack" \
  --partition="${SLURM_PARTITION:-cpu}" \
  --cpus-per-task=2 --mem=32G --time=01:00:00 \
  --dependency="afterok:${job_id}" \
  --output="${WORK_DIR}/logs/${PILOT_MODE}_pack_%j.out" \
  --error="${WORK_DIR}/logs/${PILOT_MODE}_pack_%j.err" \
  --export=ALL,LEKZIAN_SPARTA_WORKER=0,LEKZIAN_SPARTA_FINALIZE=1 \
  "${SUBMITTER}")"
finalize_id="${finalize_output%%;*}"

cat <<EOF
Submitted ${PILOT_MODE} job ${job_id}
Final pack: ${finalize_id} (runs only after all cases pass)
Cases     : ${CASE_LIST}
Run root  : ${RUN_ROOT}
Monitor   : squeue -j ${job_id}
Log       : tail -f ${WORK_DIR}/logs/${PILOT_MODE}_${job_id}_0.out
Validate  : ${PYTHON_BIN} ${VALIDATOR} ${RUN_ROOT}
Packed    : ${RUN_ROOT}/packed_manifest.json
EOF
