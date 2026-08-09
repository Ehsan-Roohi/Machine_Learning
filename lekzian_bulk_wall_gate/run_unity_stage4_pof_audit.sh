#!/usr/bin/env bash
set -euo pipefail

# Stage 4 is a representation audit over the completed Stage-3 archive.
# It never launches SPARTA/DSMC and never requests higher-order moments.
UNITY_ROOT="${UNITY_ROOT:-/project/pi_roohie_umass_edu/Combustion}"
PYTHON_BIN="${PYTHON_BIN:-/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu/bin/python}"
WORK_DIR="${WORK_DIR:-${UNITY_ROOT}/LEKZIAN_BULK_WALL_STAGE4_POF_AUDIT}"
STAGE3_DIR="${STAGE3_DIR:-${UNITY_ROOT}/LEKZIAN_BULK_WALL_STAGE3_SPATIAL}"
REPO_URL="${REPO_URL:-https://github.com/Ehsan-Roohi/Machine_Learning.git}"
GIT_REF="${GIT_REF:-agent/lekzian-gate-test}"
CODE_DIR="${WORK_DIR}/Machine_Learning"
GATE_DIR="${CODE_DIR}/lekzian_bulk_wall_gate"
DATASET="${DATASET:-${STAGE3_DIR}/data/stage3_spatial_dataset_phase1.npz}"
STAGE3_DECISION="${STAGE3_DECISION:-${STAGE3_DIR}/results/final/stage3_decision.json}"
STAGE3_SUMMARY="${STAGE3_SUMMARY:-${STAGE3_DIR}/results/final/stage3_ensemble_summary.csv}"
RESULT_DIR="${RESULT_DIR:-${WORK_DIR}/results}"
TASK_ROOT="${RESULT_DIR}/tasks"
FINAL_DIR="${RESULT_DIR}/final"
LOG_DIR="${WORK_DIR}/logs"
TASK_FILE="${WORK_DIR}/stage4_array_tasks.txt"
ARRAY_MAX_PARALLEL="${ARRAY_MAX_PARALLEL:-4}"

mkdir -p "${WORK_DIR}" "${TASK_ROOT}" "${FINAL_DIR}" "${LOG_DIR}"

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

for required in "${GATE_DIR}/stage3_spatial.py" "${GATE_DIR}/stage4_pof_audit.py" \
                "${GATE_DIR}/make_stage3_manifest.py"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: required code is absent: ${required}" >&2
    exit 2
  fi
done
if [[ ! -f "${DATASET}" ]]; then
  echo "ERROR: completed Stage-3 dataset not found: ${DATASET}" >&2
  exit 2
fi
if [[ ! -f "${STAGE3_DECISION}" ]]; then
  echo "ERROR: completed Stage-3 decision not found: ${STAGE3_DECISION}" >&2
  exit 2
fi
if [[ ! -f "${STAGE3_SUMMARY}" ]]; then
  echo "ERROR: completed Stage-3 summary not found: ${STAGE3_SUMMARY}" >&2
  exit 2
fi

"${PYTHON_BIN}" - "${STAGE3_DECISION}" <<'PY'
import json
import sys
decision = json.load(open(sys.argv[1], encoding="utf-8"))
expected = "NO_ACTIONABLE_SPATIAL_BULK_SIGNAL"
if decision.get("verdict") != expected:
    raise SystemExit(f"Stage-3 verdict must be locked as {expected}; got {decision.get('verdict')}")
print(f"Locked Stage-3 verdict: {expected}")
PY

echo "Resolved Lekzian--Roohi Stage 4 for Physics of Fluids"
echo "  Python          : ${PYTHON_BIN}"
echo "  Code            : ${GATE_DIR}"
echo "  Existing dataset: ${DATASET}"
echo "  Stage-3 decision: ${STAGE3_DECISION}"
echo "  Stage-3 summary : ${STAGE3_SUMMARY}"
echo "  Results         : ${FINAL_DIR}"
echo "  Concurrent GPUs : ${ARRAY_MAX_PARALLEL}"
echo
echo "Scientific contract"
echo "  - target venue is Physics of Fluids"
echo "  - Stage-3 prospective verdict remains locked and negative"
echo "  - zero new SPARTA/DSMC simulations and zero higher-order moments"
echo "  - same compact model, 36 folds, three seeds, and training schedule"
echo "  - patch-mean, cell-permutation, case-pool, surface-permutation controls"
echo "  - u, v, T, and logP ablations plus Ma/Kn/geometry subgroup audit"

WORKER_FILE="${WORK_DIR}/run_stage4_worker.sbatch"
AGGREGATE_FILE="${WORK_DIR}/run_stage4_aggregate.sbatch"
CONTROLLER_FILE="${WORK_DIR}/run_stage4_controller.sbatch"

cat > "${WORKER_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=lekz_s4_w
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --output=${LOG_DIR}/stage4_worker_%A_%a.out
#SBATCH --error=${LOG_DIR}/stage4_worker_%A_%a.err
set -euo pipefail
module load cuda/12.6 >/dev/null 2>&1 || true
export MPLCONFIGDIR="${WORK_DIR}/matplotlib"
mkdir -p "\${MPLCONFIGDIR}"
task_line="\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${TASK_FILE}")"
IFS='|' read -r scheme outer_group seed <<< "\${task_line}"
task_name="\$(printf 'task_%03d' "\${SLURM_ARRAY_TASK_ID}")"
"${PYTHON_BIN}" "${GATE_DIR}/stage4_pof_audit.py" \
  --dataset "${DATASET}" \
  --stage3-decision "${STAGE3_DECISION}" \
  --stage3-summary "${STAGE3_SUMMARY}" \
  --out "${TASK_ROOT}/\${task_name}" \
  --cv loco,pairout \
  --seeds "\${seed}" \
  --only-scheme "\${scheme}" \
  --only-outer-group "\${outer_group}" \
  --epochs 180 \
  --patience 25 \
  --batch-size 192 \
  --near-radius-hs 0.75 \
  --max-parameters 100000 \
  --loco-tolerances Cp=0.10,Cq=0.10,tau_abs=0.20 \
  --pairout-tolerances Cp=0.15,Cq=0.15,tau_abs=0.25 \
  --replication-tolerance-pp 0.50 \
  --resume
EOF

cat > "${AGGREGATE_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=lekz_s4_final
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=${LOG_DIR}/stage4_final_%j.out
#SBATCH --error=${LOG_DIR}/stage4_final_%j.err
set -euo pipefail
export MPLCONFIGDIR="${WORK_DIR}/matplotlib"
mkdir -p "\${MPLCONFIGDIR}"
"${PYTHON_BIN}" "${GATE_DIR}/stage4_pof_audit.py" \
  --dataset "${DATASET}" \
  --stage3-decision "${STAGE3_DECISION}" \
  --stage3-summary "${STAGE3_SUMMARY}" \
  --out "${FINAL_DIR}" \
  --aggregate-task-root "${TASK_ROOT}" \
  --cv loco,pairout \
  --seeds 101,202,303 \
  --near-radius-hs 0.75 \
  --max-parameters 100000 \
  --loco-tolerances Cp=0.10,Cq=0.10,tau_abs=0.20 \
  --pairout-tolerances Cp=0.15,Cq=0.15,tau_abs=0.25 \
  --replication-tolerance-pp 0.50
echo "FINAL STAGE-4 POF DECISION"
cat "${FINAL_DIR}/stage4_pof_decision.txt"
EOF

cat > "${CONTROLLER_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=lekz_s4_ctl
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=${LOG_DIR}/stage4_controller_%j.out
#SBATCH --error=${LOG_DIR}/stage4_controller_%j.err
set -euo pipefail
"${PYTHON_BIN}" -m unittest discover -s "${GATE_DIR}/tests" -p 'test_stage4_pof_audit.py' -v
"${PYTHON_BIN}" "${GATE_DIR}/make_stage3_manifest.py" \
  --dataset "${DATASET}" --out "${TASK_FILE}" --seeds 101,202,303
task_count="\$(wc -l < "${TASK_FILE}")"
if [[ "\${task_count}" -ne 108 ]]; then
  echo "ERROR: expected 108 Stage-4 tasks, found \${task_count}" >&2
  exit 3
fi
array_submit="\$(sbatch --array=0-107%${ARRAY_MAX_PARALLEL} "${WORKER_FILE}")"
array_job_id="\$(awk '{print \$NF}' <<< "\${array_submit}")"
final_submit="\$(sbatch --dependency=afterok:\${array_job_id} "${AGGREGATE_FILE}")"
final_job_id="\$(awk '{print \$NF}' <<< "\${final_submit}")"
cat > "${WORK_DIR}/LAST_STAGE4_JOB.env" <<ENV
CONTROLLER_JOB_ID=\${SLURM_JOB_ID}
ARRAY_JOB_ID=\${array_job_id}
FINAL_JOB_ID=\${final_job_id}
TASK_FILE=${TASK_FILE}
RESULT_DIR=${FINAL_DIR}
ENV
echo "\${array_submit}"
echo "\${final_submit}"
echo "Monitor array: squeue -j \${array_job_id}"
echo "Final result : cat ${FINAL_DIR}/stage4_pof_decision.txt"
EOF

submit_output="$(sbatch "${CONTROLLER_FILE}")"
controller_job_id="$(awk '{print $NF}' <<< "${submit_output}")"
cat > "${WORK_DIR}/LAST_STAGE4_JOB.env" <<EOF
CONTROLLER_JOB_ID=${controller_job_id}
OUT=${LOG_DIR}/stage4_controller_${controller_job_id}.out
ERR=${LOG_DIR}/stage4_controller_${controller_job_id}.err
RESULT_DIR=${FINAL_DIR}
EOF
echo "${submit_output}"
echo "Monitor controller: squeue -j ${controller_job_id}"
echo "Controller log    : tail -f ${LOG_DIR}/stage4_controller_${controller_job_id}.out"
echo "Final decision    : cat ${FINAL_DIR}/stage4_pof_decision.txt"
