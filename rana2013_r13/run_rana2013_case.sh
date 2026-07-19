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
START=$(date -u +%FT%TZ)
set +e
(
  cd "$CASE"
  timeout --signal=TERM --kill-after=30s "${WALLCLOCK_MINUTES}m" \
    /usr/bin/time -v mpirun --oversubscribe -np "$RANKS" "$EXE" run "datin/$INPUT"
) > "$CASE/logs/astr.log" 2> "$CASE/logs/astr_time.log"
RC=$?
set -e
STATUS=completed
if [[ $RC -eq 124 || $RC -eq 137 || $RC -eq 143 ]]; then STATUS=wallclock_partial; fi
if grep -q 'COMPUTATION CRASHED' "$CASE/logs/astr.log" 2>/dev/null; then STATUS=crashed; fi
if [[ ! -s "$CASE/outdat/flowfield.h5" ]]; then
  echo "No flowfield checkpoint produced" >&2
  STATUS=no_checkpoint
  RC=3
else
  python3 "$(dirname "$0")/analyze_rana2013_case.py" "$CASE" --reference "$REFERENCE" --output-dir "$OUT"
fi
python3 - "$CASE" "$START" "$STATUS" "$RC" <<'PY'
from pathlib import Path
import h5py,json,sys,datetime
case=Path(sys.argv[1]); start,status,rc=sys.argv[2],sys.argv[3],int(sys.argv[4])
step=None;time=None; p=case/'outdat/flowfield.h5'
if p.exists():
  with h5py.File(p,'r') as h: step=int(h['nstep'][0]); time=float(h['time'][0])
d={'started_at':start,'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':status,'return_code':rc,'last_checkpoint_step':step,'simulation_time':time}
(case/'analysis/run_status.json').write_text(json.dumps(d,indent=2)); print(json.dumps(d,indent=2))
PY
if [[ "$STATUS" == crashed || "$STATUS" == no_checkpoint ]]; then exit 3; fi
exit 0
