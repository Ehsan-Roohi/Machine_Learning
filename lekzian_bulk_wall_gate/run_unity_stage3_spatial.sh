#!/usr/bin/env bash
set -euo pipefail

# Spatial Stage 3 reads existing DSMC files only.  It never submits SPARTA.
UNITY_ROOT="${UNITY_ROOT:-/project/pi_roohie_umass_edu/Combustion}"
LEGACY_ROOT="${LEGACY_ROOT:-/project/pi_roohie_umass_edu/Sabouri}"
PYTHON_BIN="${PYTHON_BIN:-/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu/bin/python}"
WORK_DIR="${WORK_DIR:-${UNITY_ROOT}/LEKZIAN_BULK_WALL_STAGE3_SPATIAL}"
REPO_URL="${REPO_URL:-https://github.com/Ehsan-Roohi/Machine_Learning.git}"
GIT_REF="${GIT_REF:-agent/lekzian-gate-test}"
CODE_DIR="${WORK_DIR}/Machine_Learning"
GATE_DIR="${CODE_DIR}/lekzian_bulk_wall_gate"
DATA_DIR="${DATA_DIR:-${WORK_DIR}/data}"
DATASET="${DATASET:-${DATA_DIR}/stage3_spatial_dataset_phase1.npz}"
RESULT_DIR="${RESULT_DIR:-${WORK_DIR}/results}"
TASK_ROOT="${RESULT_DIR}/tasks"
FINAL_DIR="${RESULT_DIR}/final"
LOG_DIR="${WORK_DIR}/logs"
TASK_FILE="${WORK_DIR}/stage3_array_tasks.txt"
ARRAY_MAX_PARALLEL="${ARRAY_MAX_PARALLEL:-4}"

mkdir -p "${WORK_DIR}" "${DATA_DIR}" "${TASK_ROOT}" "${FINAL_DIR}" "${LOG_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python environment not found: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ -d "${CODE_DIR}/.git" ]]; then
  git -C "${CODE_DIR}" fetch origin "${GIT_REF}"
  git -C "${CODE_DIR}" checkout "${GIT_REF}"
  git -C "${CODE_DIR}" pull --ff-only origin "${GIT_REF}"
else
  git clone --depth 1 --branch "${GIT_REF}" "${REPO_URL}" "${CODE_DIR}"
fi
if [[ ! -f "${GATE_DIR}/stage3_spatial.py" ]]; then
  echo "ERROR: Stage-3 code is absent under ${GATE_DIR}" >&2
  exit 2
fi

find_first() {
  local root="$1"
  shift
  find "${root}" -type f \( "$@" \) -print -quit 2>/dev/null || true
}

if [[ ! -f "${DATASET}" ]]; then
  existing_dataset="$(find_first "${UNITY_ROOT}" -name stage3_spatial_dataset_phase1.npz)"
  [[ -n "${existing_dataset}" ]] && DATASET="${existing_dataset}"
fi
AUDIT_DIR="${AUDIT_DIR:-}"
NONLOCAL_SCRIPT="${NONLOCAL_SCRIPT:-}"
FIELD_BASE_SCRIPT="${FIELD_BASE_SCRIPT:-}"
SURFACE_BASE_SCRIPT="${SURFACE_BASE_SCRIPT:-}"
if [[ ! -f "${DATASET}" ]]; then
  [[ -z "${AUDIT_DIR}" && -d "${LEGACY_ROOT}/protrusion_01_audit_v2" ]] &&
    AUDIT_DIR="${LEGACY_ROOT}/protrusion_01_audit_v2"
  [[ -z "${NONLOCAL_SCRIPT}" && -f "${LEGACY_ROOT}/16_protrusion_nonlocal_bulk_to_wall_footprint.py" ]] &&
    NONLOCAL_SCRIPT="${LEGACY_ROOT}/16_protrusion_nonlocal_bulk_to_wall_footprint.py"
  [[ -z "${FIELD_BASE_SCRIPT}" && -f "${LEGACY_ROOT}/10_protrusion_train_smooth_field_operator_v4_geomfix.py" ]] &&
    FIELD_BASE_SCRIPT="${LEGACY_ROOT}/10_protrusion_train_smooth_field_operator_v4_geomfix.py"
  [[ -z "${SURFACE_BASE_SCRIPT}" && -f "${LEGACY_ROOT}/06_protrusion_train_unified_operator_v8_geomfix.py" ]] &&
    SURFACE_BASE_SCRIPT="${LEGACY_ROOT}/06_protrusion_train_unified_operator_v8_geomfix.py"
  if [[ -z "${AUDIT_DIR}" ]]; then
    manifest="$(find_first /project/pi_roohie_umass_edu -path '*/protrusion_01_audit_v2/manifest_raw.csv')"
    [[ -n "${manifest}" ]] && AUDIT_DIR="$(dirname "${manifest}")"
  fi
  [[ -z "${NONLOCAL_SCRIPT}" ]] &&
    NONLOCAL_SCRIPT="$(find_first /project/pi_roohie_umass_edu -name 16_protrusion_nonlocal_bulk_to_wall_footprint.py)"
  [[ -z "${FIELD_BASE_SCRIPT}" ]] &&
    FIELD_BASE_SCRIPT="$(find_first /project/pi_roohie_umass_edu -name '10_protrusion_train_smooth_field_operator_v4_geomfix.py' -o -name '10_protrusion_train_smooth_field_operator*.py')"
  [[ -z "${SURFACE_BASE_SCRIPT}" ]] &&
    SURFACE_BASE_SCRIPT="$(find_first /project/pi_roohie_umass_edu -name '06_protrusion_train_unified_operator_v8_geomfix.py' -o -name '06_protrusion_train_unified_operator*.py')"
fi

echo "Resolved Lekzian--Roohi Spatial Stage 3"
echo "  Python           : ${PYTHON_BIN}"
echo "  Code             : ${GATE_DIR}"
echo "  Existing dataset : ${DATASET}"
echo "  Audit directory  : ${AUDIT_DIR:-not needed when dataset exists}"
echo "  Existing reader  : ${FIELD_BASE_SCRIPT:-not needed when dataset exists}"
echo "  Results          : ${FINAL_DIR}"
echo "  Concurrent GPUs  : ${ARRAY_MAX_PARALLEL}"
echo
echo "Scientific contract"
echo "  - zero new SPARTA/DSMC simulations and zero higher-order moments"
echo "  - wall-aligned patches of existing u,v,T,logP fields"
echo "  - one compact multi-target operator; hard 100,000-parameter cap"
echo "  - P0/near/far/upstream/downstream/full with shifted and radial-flip controls"
echo "  - Mach flux scaling, train-fold-only normalization, no double apex weighting"
echo "  - 27 LOCO + 9 pair-out folds; three seeds; 108 resumable GPU tasks"

if [[ ! -f "${DATASET}" ]]; then
  missing=()
  [[ -n "${AUDIT_DIR}" && -f "${AUDIT_DIR}/manifest_raw.csv" ]] || missing+=("audit manifest")
  [[ -n "${NONLOCAL_SCRIPT}" && -f "${NONLOCAL_SCRIPT}" ]] || missing+=("legacy script 16")
  [[ -n "${FIELD_BASE_SCRIPT}" && -f "${FIELD_BASE_SCRIPT}" ]] || missing+=("legacy field reader 10")
  [[ -n "${SURFACE_BASE_SCRIPT}" && -f "${SURFACE_BASE_SCRIPT}" ]] || missing+=("legacy surface reader 06")
  if (( ${#missing[@]} > 0 )); then
    echo "ERROR: cannot read the existing DSMC archive. Missing:" >&2
    printf '  - %s\n' "${missing[@]}" >&2
    exit 2
  fi
fi

WORKER_FILE="${WORK_DIR}/run_stage3_worker.sbatch"
AGGREGATE_FILE="${WORK_DIR}/run_stage3_aggregate.sbatch"
CONTROLLER_FILE="${WORK_DIR}/run_stage3_controller.sbatch"

cat > "${WORKER_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=lekz_s3_w
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --output=${LOG_DIR}/stage3_worker_%A_%a.out
#SBATCH --error=${LOG_DIR}/stage3_worker_%A_%a.err
set -euo pipefail
module load cuda/12.6 >/dev/null 2>&1 || true
export MPLCONFIGDIR="${WORK_DIR}/matplotlib"
mkdir -p "\${MPLCONFIGDIR}"
task_line="\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${TASK_FILE}")"
IFS='|' read -r scheme outer_group seed <<< "\${task_line}"
task_name="\$(printf 'task_%03d' "\${SLURM_ARRAY_TASK_ID}")"
"${PYTHON_BIN}" "${GATE_DIR}/stage3_spatial.py" \
  --dataset "${DATASET}" \
  --out "${TASK_ROOT}/\${task_name}" \
  --cv loco,pairout \
  --seeds "\${seed}" \
  --only-scheme "\${scheme}" \
  --only-outer-group "\${outer_group}" \
  --epochs 180 \
  --patience 25 \
  --batch-size 192 \
  --near-radius-hs 0.75 \
  --surface-shift-fraction 0.25 \
  --max-parameters 100000 \
  --loco-tolerances Cp=0.10,Cq=0.10,tau_abs=0.20 \
  --pairout-tolerances Cp=0.15,Cq=0.15,tau_abs=0.25 \
  --min-gain-pp 1.0 \
  --resume
EOF

cat > "${AGGREGATE_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=lekz_s3_final
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=${LOG_DIR}/stage3_final_%j.out
#SBATCH --error=${LOG_DIR}/stage3_final_%j.err
set -euo pipefail
export MPLCONFIGDIR="${WORK_DIR}/matplotlib"
mkdir -p "\${MPLCONFIGDIR}"
"${PYTHON_BIN}" "${GATE_DIR}/stage3_spatial.py" \
  --dataset "${DATASET}" \
  --out "${FINAL_DIR}" \
  --aggregate-task-root "${TASK_ROOT}" \
  --cv loco,pairout \
  --seeds 101,202,303 \
  --near-radius-hs 0.75 \
  --surface-shift-fraction 0.25 \
  --max-parameters 100000 \
  --loco-tolerances Cp=0.10,Cq=0.10,tau_abs=0.20 \
  --pairout-tolerances Cp=0.15,Cq=0.15,tau_abs=0.25 \
  --min-gain-pp 1.0
echo "FINAL SPATIAL STAGE-3 DECISION"
cat "${FINAL_DIR}/stage3_decision.txt"
EOF

cat > "${CONTROLLER_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=lekz_s3_ctl
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=${LOG_DIR}/stage3_controller_%j.out
#SBATCH --error=${LOG_DIR}/stage3_controller_%j.err
set -euo pipefail
export MPLCONFIGDIR="${WORK_DIR}/matplotlib"
mkdir -p "\${MPLCONFIGDIR}"
"${PYTHON_BIN}" -m unittest discover -s "${GATE_DIR}/tests" -p 'test_stage3_spatial.py' -v
if [[ ! -f "${DATASET}" ]]; then
  "${PYTHON_BIN}" "${GATE_DIR}/prepare_stage3_spatial.py" \
    --audit-dir "${AUDIT_DIR}" \
    --nonlocal-script "${NONLOCAL_SCRIPT}" \
    --field-base-script "${FIELD_BASE_SCRIPT}" \
    --surface-base-script "${SURFACE_BASE_SCRIPT}" \
    --out "${DATA_DIR}" \
    --max-gas-points 60000 \
    --tangential-points 21 \
    --normal-points 11 \
    --resume
fi
"${PYTHON_BIN}" "${GATE_DIR}/make_stage3_manifest.py" \
  --dataset "${DATASET}" --out "${TASK_FILE}" --seeds 101,202,303
task_count="\$(wc -l < "${TASK_FILE}")"
if [[ "\${task_count}" -ne 108 ]]; then
  echo "ERROR: expected 108 Stage-3 tasks, found \${task_count}" >&2
  exit 3
fi
array_submit="\$(sbatch --array=0-107%${ARRAY_MAX_PARALLEL} "${WORKER_FILE}")"
array_job_id="\$(awk '{print \$NF}' <<< "\${array_submit}")"
final_submit="\$(sbatch --dependency=afterok:\${array_job_id} "${AGGREGATE_FILE}")"
final_job_id="\$(awk '{print \$NF}' <<< "\${final_submit}")"
cat > "${WORK_DIR}/LAST_STAGE3_JOB.env" <<ENV
CONTROLLER_JOB_ID=\${SLURM_JOB_ID}
ARRAY_JOB_ID=\${array_job_id}
FINAL_JOB_ID=\${final_job_id}
TASK_FILE=${TASK_FILE}
RESULT_DIR=${FINAL_DIR}
ENV
echo "\${array_submit}"
echo "\${final_submit}"
echo "Monitor array: squeue -j \${array_job_id}"
echo "Final result : cat ${FINAL_DIR}/stage3_decision.txt"
EOF

submit_output="$(sbatch "${CONTROLLER_FILE}")"
controller_job_id="$(awk '{print $NF}' <<< "${submit_output}")"
cat > "${WORK_DIR}/LAST_STAGE3_JOB.env" <<EOF
CONTROLLER_JOB_ID=${controller_job_id}
OUT=${LOG_DIR}/stage3_controller_${controller_job_id}.out
ERR=${LOG_DIR}/stage3_controller_${controller_job_id}.err
RESULT_DIR=${FINAL_DIR}
EOF
echo "${submit_output}"
echo "Monitor controller: squeue -j ${controller_job_id}"
echo "Controller log    : tail -f ${LOG_DIR}/stage3_controller_${controller_job_id}.out"
echo "Final decision    : cat ${FINAL_DIR}/stage3_decision.txt"
