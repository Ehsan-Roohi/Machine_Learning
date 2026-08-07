#!/usr/bin/env bash
set -euo pipefail

UNITY_ROOT="${UNITY_ROOT:-/project/pi_roohie_umass_edu/Combustion}"
PYTHON_BIN="${PYTHON_BIN:-/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu/bin/python}"
WORK_DIR="${WORK_DIR:-${UNITY_ROOT}/LEKZIAN_BULK_WALL_STAGE2_CONDITIONAL}"
REPO_URL="${REPO_URL:-https://github.com/Ehsan-Roohi/Machine_Learning.git}"
GIT_REF="${GIT_REF:-agent/lekzian-gate-test}"
CODE_DIR="${WORK_DIR}/Machine_Learning"
GATE_DIR="${CODE_DIR}/lekzian_bulk_wall_gate"
RESULT_DIR="${RESULT_DIR:-${WORK_DIR}/results}"
TASK_ROOT="${RESULT_DIR}/tasks"
FINAL_DIR="${RESULT_DIR}/final"
LOG_DIR="${WORK_DIR}/logs"
TASK_FILE="${WORK_DIR}/stage2_array_tasks.txt"
ARRAY_MAX_PARALLEL="${ARRAY_MAX_PARALLEL:-4}"

mkdir -p "${WORK_DIR}" "${RESULT_DIR}" "${TASK_ROOT}" "${FINAL_DIR}" "${LOG_DIR}"

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

if [[ ! -f "${GATE_DIR}/stage2_conditional.py" ]]; then
  echo "ERROR: Stage-2 conditional code is absent under ${GATE_DIR}" >&2
  exit 2
fi

FEATURE_TABLE="${FEATURE_TABLE:-${UNITY_ROOT}/LEKZIAN_BULK_WALL_GATE/features/surface_patch_dataset_full_gate.csv}"
if [[ ! -f "${FEATURE_TABLE}" ]]; then
  FEATURE_TABLE="$(find "${UNITY_ROOT}" -type f -name surface_patch_dataset_full_gate.csv -print -quit 2>/dev/null || true)"
fi
if [[ -z "${FEATURE_TABLE}" || ! -f "${FEATURE_TABLE}" ]]; then
  echo "ERROR: existing six-annulus Phase-1 feature table was not found." >&2
  echo "Set FEATURE_TABLE=/absolute/path/surface_patch_dataset_full_gate.csv and rerun." >&2
  exit 2
fi

echo "Resolved Lekzian--Roohi conditional Stage-2 paths"
echo "  Unity root       : ${UNITY_ROOT}"
echo "  Python           : ${PYTHON_BIN}"
echo "  Code             : ${GATE_DIR}"
echo "  Existing data    : ${FEATURE_TABLE}"
echo "  Work directory   : ${WORK_DIR}"
echo "  Results          : ${FINAL_DIR}"
echo "  Concurrent GPUs  : ${ARRAY_MAX_PARALLEL}"
echo
echo "Prospective scientific design"
echo "  - existing macroscopic six-annulus table; no new SPARTA/DSMC output"
echo "  - independently trained and frozen target-specific wall baseline"
echo "  - train-fold ridge residualization of physical ring descriptors"
echo "  - conditional raw, outer-excess, and adjacent-ring contrast channels"
echo "  - capacity-matched C0, finite-radius, near/far/interleaved, and full models"
echo "  - cyclic surface-alignment and radial-order inference controls"
echo "  - apex-focused diagnostic weighting justified by completed Stage-1"
echo "  - 27 LOCO + 9 pair-out groups; five seeds; 540 resumable GPU tasks"

WORKER_FILE="${WORK_DIR}/run_stage2_worker.sbatch"
AGGREGATE_FILE="${WORK_DIR}/run_stage2_aggregate.sbatch"
CONTROLLER_FILE="${WORK_DIR}/run_stage2_controller.sbatch"

cat > "${WORKER_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=lekz_s2_w
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=08:00:00
#SBATCH --output=${LOG_DIR}/stage2_worker_%A_%a.out
#SBATCH --error=${LOG_DIR}/stage2_worker_%A_%a.err
set -euo pipefail
module load cuda/12.6 >/dev/null 2>&1 || true
export MPLCONFIGDIR="${WORK_DIR}/matplotlib"
mkdir -p "\${MPLCONFIGDIR}"
task_line="\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${TASK_FILE}")"
IFS='|' read -r scheme outer_group seed target <<< "\${task_line}"
task_name="\$(printf 'task_%03d' "\${SLURM_ARRAY_TASK_ID}")"
task_out="${TASK_ROOT}/\${task_name}"
"${PYTHON_BIN}" "${GATE_DIR}/stage2_conditional.py" \
  --feature-table "${FEATURE_TABLE}" \
  --out "\${task_out}" \
  --cv loco,pairout \
  --seeds "\${seed}" \
  --only-scheme "\${scheme}" \
  --only-outer-group "\${outer_group}" \
  --only-target "\${target}" \
  --full-residual-scale 0.05 \
  --ridge 0.1 \
  --apex-boost 3.0 \
  --apex-sigma 0.07 \
  --surface-shifts 12 \
  --loco-tolerances Cp=0.10,Cq=0.10,tau_abs=0.20 \
  --pairout-tolerances Cp=0.15,Cq=0.15,tau_abs=0.25 \
  --selection-tolerances Cp=0.15,Cq=0.15,tau_abs=0.25 \
  --selection-margin-pp 1.0 \
  --max-censor-fraction 0.20 \
  --min-gain-pp 1.0 \
  --resume
EOF

cat > "${AGGREGATE_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=lekz_s2_final
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=${LOG_DIR}/stage2_final_%j.out
#SBATCH --error=${LOG_DIR}/stage2_final_%j.err
set -euo pipefail
export MPLCONFIGDIR="${WORK_DIR}/matplotlib"
mkdir -p "\${MPLCONFIGDIR}"
"${PYTHON_BIN}" "${GATE_DIR}/stage2_conditional.py" \
  --feature-table "${FEATURE_TABLE}" \
  --out "${FINAL_DIR}" \
  --aggregate-task-root "${TASK_ROOT}" \
  --cv loco,pairout \
  --seeds 101,202,303,404,505 \
  --full-residual-scale 0.05 \
  --ridge 0.1 \
  --apex-boost 3.0 \
  --apex-sigma 0.07 \
  --surface-shifts 12 \
  --loco-tolerances Cp=0.10,Cq=0.10,tau_abs=0.20 \
  --pairout-tolerances Cp=0.15,Cq=0.15,tau_abs=0.25 \
  --selection-tolerances Cp=0.15,Cq=0.15,tau_abs=0.25 \
  --selection-margin-pp 1.0 \
  --max-censor-fraction 0.20 \
  --min-gain-pp 1.0
echo "FINAL CONDITIONAL STAGE-2 DECISION"
cat "${FINAL_DIR}/stage2_decision.txt"
EOF

cat > "${CONTROLLER_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=lekz_s2_ctl
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=${LOG_DIR}/stage2_controller_%j.out
#SBATCH --error=${LOG_DIR}/stage2_controller_%j.err
set -euo pipefail
export MPLCONFIGDIR="${WORK_DIR}/matplotlib"
mkdir -p "\${MPLCONFIGDIR}"
"${PYTHON_BIN}" -m unittest discover \
  -s "${GATE_DIR}/tests" \
  -p 'test_stage2_conditional.py' \
  -v
"${PYTHON_BIN}" "${GATE_DIR}/make_stage2_manifest.py" \
  --feature-table "${FEATURE_TABLE}" \
  --out "${TASK_FILE}" \
  --seeds 101,202,303,404,505
task_count="\$(wc -l < "${TASK_FILE}")"
if [[ "\${task_count}" -ne 540 ]]; then
  echo "ERROR: expected 540 Stage-2 tasks, found \${task_count}" >&2
  exit 3
fi
array_submit="\$(sbatch --array=0-539%${ARRAY_MAX_PARALLEL} "${WORKER_FILE}")"
array_job_id="\$(awk '{print \$NF}' <<< "\${array_submit}")"
final_submit="\$(sbatch --dependency=afterok:\${array_job_id} "${AGGREGATE_FILE}")"
final_job_id="\$(awk '{print \$NF}' <<< "\${final_submit}")"
cat > "${WORK_DIR}/LAST_STAGE2_JOB.env" <<ENV
CONTROLLER_JOB_ID=\${SLURM_JOB_ID}
ARRAY_JOB_ID=\${array_job_id}
FINAL_JOB_ID=\${final_job_id}
TASK_FILE=${TASK_FILE}
RESULT_DIR=${FINAL_DIR}
ENV
echo "\${array_submit}"
echo "\${final_submit}"
echo "Monitor array: squeue -j \${array_job_id}"
echo "Final result : cat ${FINAL_DIR}/stage2_decision.txt"
EOF

submit_output="$(sbatch "${CONTROLLER_FILE}")"
controller_job_id="$(awk '{print $NF}' <<< "${submit_output}")"
cat > "${WORK_DIR}/LAST_STAGE2_JOB.env" <<EOF
CONTROLLER_JOB_ID=${controller_job_id}
OUT=${LOG_DIR}/stage2_controller_${controller_job_id}.out
ERR=${LOG_DIR}/stage2_controller_${controller_job_id}.err
RESULT_DIR=${FINAL_DIR}
EOF
echo "${submit_output}"
echo "Monitor controller: squeue -j ${controller_job_id}"
echo "Controller log    : tail -f ${LOG_DIR}/stage2_controller_${controller_job_id}.out"
echo "Final decision    : cat ${FINAL_DIR}/stage2_decision.txt"
