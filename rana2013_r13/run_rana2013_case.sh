#!/usr/bin/env bash
set -euo pipefail

CASE=${1:?case directory}
EXE=${2:?ASTR executable}
REFERENCE=${3:?reference JSON}
WALLCLOCK_MINUTES=${WALLCLOCK_MINUTES:-300}
OUT=${OUT:-"${CASE}/analysis"}
mkdir -p "$OUT" "${CASE}/logs"
INPUT=$(basename "$(find "$CASE/datin" -maxdepth 1 -name 'input.astr*' -print -quit)")
RANKS=$(python3 - "$CASE/rana2013_case_metadata.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['mpi_ranks'])
PY
)
TARGET_STEP=$(python3 - "$CASE/rana2013_case_metadata.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['maxstep'])
PY
)
START=$(date -u +%FT%TZ)
set +e
(
  cd "$CASE"
  timeout --signal=TERM --kill-after=30s "${WALLCLOCK_MINUTES}m" \
    /usr/bin/time -v mpirun --oversubscribe -np "$RANKS" "$EXE" run "datin/$INPUT"
) > "$CASE/logs/astr.log" 2> "$CASE/logs/astr_time.log"
RC=$?
set -e

case "$RC" in
  0) STATUS=completed ;;
  124|137|143) STATUS=wallclock_partial ;;
  *) STATUS=solver_failed ;;
esac
if grep -q 'COMPUTATION CRASHED' "$CASE/logs/astr.log" "$CASE/logs/astr_time.log" 2>/dev/null; then
  STATUS=crashed
fi

STEP=-1
SIM_TIME=0.0
if [[ ! -s "$CASE/outdat/flowfield.h5" ]]; then
  echo "No flowfield checkpoint produced" >&2
  STATUS=no_checkpoint
else
  read -r STEP SIM_TIME < <(python3 - "$CASE/outdat/flowfield.h5" <<'PY'
import h5py,sys
with h5py.File(sys.argv[1],'r') as h:
    print(int(h['nstep'][0]),float(h['time'][0]))
PY
  )
  if [[ "$STEP" -le 0 && "$RC" -ne 0 ]]; then
    STATUS=no_progress
  elif [[ "$STATUS" == completed && "$STEP" -lt "$TARGET_STEP" ]]; then
    STATUS=incomplete
    RC=4
  fi

  set +e
  python3 "$(dirname "$0")/analyze_rana2013_case.py" "$CASE" --reference "$REFERENCE" --output-dir "$OUT"
  ANALYZE_RC=$?
  set -e
  if [[ "$ANALYZE_RC" -ne 0 && "$STATUS" != crashed && "$STATUS" != no_checkpoint && "$STATUS" != no_progress ]]; then
    STATUS=analysis_failed
    RC=5
  fi
fi

python3 - "$CASE" "$START" "$STATUS" "$RC" "$STEP" "$SIM_TIME" "$TARGET_STEP" <<'PY'
from pathlib import Path
import json,sys,datetime
case=Path(sys.argv[1]); start,status=sys.argv[2],sys.argv[3]
rc,step=int(sys.argv[4]),int(sys.argv[5]); sim_time=float(sys.argv[6]); target=int(sys.argv[7])
d={
  'started_at':start,
  'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'status':status,
  'return_code':rc,
  'last_checkpoint_step':step if step >= 0 else None,
  'target_step':target,
  'simulation_time':sim_time,
  'completed_fraction':(step/target if step >= 0 and target > 0 else None),
}
(case/'analysis/run_status.json').write_text(json.dumps(d,indent=2)); print(json.dumps(d,indent=2))
PY

case "$STATUS" in
  completed|wallclock_partial) exit 0 ;;
  *) exit 3 ;;
esac
