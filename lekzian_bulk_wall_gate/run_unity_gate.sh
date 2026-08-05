#!/usr/bin/env bash
set -euo pipefail

# Unity defaults already established for the Roohi account.  Every path can be
# overridden as an environment variable before invoking this launcher.
UNITY_ROOT="${UNITY_ROOT:-/project/pi_roohie_umass_edu/Combustion}"
PYTHON_BIN="${PYTHON_BIN:-/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu/bin/python}"
WORK_DIR="${WORK_DIR:-${UNITY_ROOT}/LEKZIAN_BULK_WALL_GATE}"
REPO_URL="${REPO_URL:-https://github.com/Ehsan-Roohi/Machine_Learning.git}"
GIT_REF="${GIT_REF:-agent/lekzian-gate-test}"
CODE_DIR="${WORK_DIR}/Machine_Learning"
RESULT_DIR="${RESULT_DIR:-${WORK_DIR}/results}"
FEATURE_DIR="${FEATURE_DIR:-${WORK_DIR}/features}"
LOG_DIR="${WORK_DIR}/logs"

mkdir -p "${WORK_DIR}" "${RESULT_DIR}" "${FEATURE_DIR}" "${LOG_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python environment not found: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN=/absolute/path/to/python and rerun." >&2
  exit 2
fi

if [[ -d "${CODE_DIR}/.git" ]]; then
  git -C "${CODE_DIR}" fetch origin "${GIT_REF}"
  git -C "${CODE_DIR}" checkout "${GIT_REF}"
  git -C "${CODE_DIR}" pull --ff-only origin "${GIT_REF}"
else
  git clone --depth 1 --branch "${GIT_REF}" "${REPO_URL}" "${CODE_DIR}"
fi

GATE_DIR="${CODE_DIR}/lekzian_bulk_wall_gate"
if [[ ! -f "${GATE_DIR}/gate_test.py" ]]; then
  echo "ERROR: gate code not found under ${GATE_DIR}" >&2
  exit 2
fi

find_first() {
  local root="$1"
  shift
  find "${root}" -type f \( "$@" \) -print -quit 2>/dev/null || true
}

# Prefer an explicitly supplied/precomputed full-ring table.  This avoids
# repeating the expensive DSMC descriptor extraction.
FEATURE_TABLE="${FEATURE_TABLE:-}"
if [[ -z "${FEATURE_TABLE}" ]]; then
  FEATURE_TABLE="$(find_first "${UNITY_ROOT}" -name surface_patch_dataset_full.csv)"
fi

AUDIT_DIR="${AUDIT_DIR:-}"
NONLOCAL_SCRIPT="${NONLOCAL_SCRIPT:-}"
FIELD_BASE_SCRIPT="${FIELD_BASE_SCRIPT:-}"
SURFACE_BASE_SCRIPT="${SURFACE_BASE_SCRIPT:-}"

if [[ -z "${FEATURE_TABLE}" ]]; then
  if [[ -z "${AUDIT_DIR}" ]]; then
    manifest="$(find_first "${UNITY_ROOT}" -name manifest_raw.csv)"
    [[ -n "${manifest}" ]] && AUDIT_DIR="$(dirname "${manifest}")"
  fi
  if [[ -z "${NONLOCAL_SCRIPT}" ]]; then
    NONLOCAL_SCRIPT="$(find_first "${UNITY_ROOT}" -name 16_protrusion_nonlocal_bulk_to_wall_footprint.py)"
  fi
  if [[ -z "${FIELD_BASE_SCRIPT}" ]]; then
    FIELD_BASE_SCRIPT="$(find_first "${UNITY_ROOT}" -name '10_protrusion_train_smooth_field_operator_v4_geomfix.py' -o -name '10_protrusion_train_smooth_field_operator*.py')"
  fi
  if [[ -z "${SURFACE_BASE_SCRIPT}" ]]; then
    SURFACE_BASE_SCRIPT="$(find_first "${UNITY_ROOT}" -name '06_protrusion_train_unified_operator_v8_geomfix.py' -o -name '06_protrusion_train_unified_operator_v7*.py' -o -name '06_protrusion_train_unified_operator*.py')"
  fi
  FEATURE_TABLE="${FEATURE_DIR}/surface_patch_dataset_full.csv"
fi

echo "Resolved Lekzian Gate Test paths"
echo "  Unity root       : ${UNITY_ROOT}"
echo "  Python           : ${PYTHON_BIN}"
echo "  Code             : ${GATE_DIR}"
echo "  Feature table    : ${FEATURE_TABLE}"
echo "  Audit directory  : ${AUDIT_DIR:-not needed (precomputed table found)}"
echo "  Nonlocal script  : ${NONLOCAL_SCRIPT:-not needed}"
echo "  Field base       : ${FIELD_BASE_SCRIPT:-not needed}"
echo "  Surface base     : ${SURFACE_BASE_SCRIPT:-not needed}"
echo "  Results          : ${RESULT_DIR}"
echo
echo "The job will run M0, M_shuffled, five finite radii, and M_full with identical"
echo "neural capacity; 27-fold case-out and 9-fold (Ma,Kn)-pair-out CV; five seeds;"
echo "and physical-case bootstrap uncertainty."

if [[ ! -f "${FEATURE_TABLE}" ]]; then
  for required in "${AUDIT_DIR}" "${NONLOCAL_SCRIPT}" "${FIELD_BASE_SCRIPT}" "${SURFACE_BASE_SCRIPT}"; do
    if [[ -z "${required}" || ! -e "${required}" ]]; then
      echo "ERROR: feature table is absent and a legacy extraction path could not be resolved." >&2
      echo "Set FEATURE_TABLE, or set AUDIT_DIR/NONLOCAL_SCRIPT/FIELD_BASE_SCRIPT/SURFACE_BASE_SCRIPT." >&2
      exit 2
    fi
  done
fi

SBATCH_FILE="${WORK_DIR}/run_lekzian_gate.sbatch"
cat > "${SBATCH_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=lekzian_gate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=${LOG_DIR}/lekzian_gate_%j.out
#SBATCH --error=${LOG_DIR}/lekzian_gate_%j.err
set -euo pipefail
module load cuda/12.6 >/dev/null 2>&1 || true

if [[ ! -f "${FEATURE_TABLE}" ]]; then
  "${PYTHON_BIN}" "${GATE_DIR}/prepare_gate_features.py" \
    --audit-dir "${AUDIT_DIR}" \
    --nonlocal-script "${NONLOCAL_SCRIPT}" \
    --field-base-script "${FIELD_BASE_SCRIPT}" \
    --surface-base-script "${SURFACE_BASE_SCRIPT}" \
    --out "${FEATURE_DIR}" \
    --max-gas-points 60000
fi

"${PYTHON_BIN}" "${GATE_DIR}/gate_test.py" \
  --feature-table "${FEATURE_TABLE}" \
  --out "${RESULT_DIR}" \
  --radii 0.1,0.25,0.5,1.0,2.0 \
  --cv loco,pairout \
  --seeds 101,202,303,404,505 \
  --absolute-tolerances Cp=0.10,Cq=0.10,tau_abs=0.20 \
  --min-gain-pp 2.0 \
  --delta-full-pp 2.0 \
  --resume

echo "FINAL DECISION"
cat "${RESULT_DIR}/gate_decision.txt"
EOF

if command -v sbatch >/dev/null 2>&1; then
  submit_output="$(sbatch "${SBATCH_FILE}")"
  job_id="$(awk '{print $NF}' <<< "${submit_output}")"
  env_file="${WORK_DIR}/LAST_LEKZIAN_GATE_JOB.env"
  cat > "${env_file}" <<EOF
JOB_ID=${job_id}
OUT=${LOG_DIR}/lekzian_gate_${job_id}.out
ERR=${LOG_DIR}/lekzian_gate_${job_id}.err
RESULT_DIR=${RESULT_DIR}
EOF
  echo "${submit_output}"
  echo "Monitor: squeue -j ${job_id}"
  echo "Log    : tail -f ${LOG_DIR}/lekzian_gate_${job_id}.out"
  echo "Result : cat ${RESULT_DIR}/gate_decision.txt"
else
  echo "sbatch not found; running the generated job script interactively."
  bash "${SBATCH_FILE}"
fi
