#!/usr/bin/env bash
set -euo pipefail
STATE=${1:?state dir}; TARGET=${2:?target}; EXE="$STATE/astr/build/bin/astr"; CASE="$STATE/case"; chmod +x "$EXE"
python3 - "$CASE" "$TARGET" <<'PY'
from pathlib import Path
import json,sys
case=Path(sys.argv[1]); target=int(sys.argv[2]);ctrl=case/'datin/controller';ls=ctrl.read_text().splitlines()
for i,l in enumerate(ls):
 if l.strip().startswith('# maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg'):ls[i+1]=f'{target:9d},    1000,    1000,     500,     100,   100';break
else:raise RuntimeError('controller target line')
ctrl.write_text('\n'.join(ls)+'\n');meta=json.loads((case/'case_metadata.json').read_text());meta['maxstep']=target;meta['target_time']=meta['dt']*target;(case/'case_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
PY
SEG="$STATE/segments/step_$(printf '%06d' "$TARGET")"; HIST="$SEG/checkpoints"; mkdir -p "$HIST" "$SEG/logs"
cat > "$SEG/watch.py" <<'PY'
from pathlib import Path
import h5py,numpy as np,os,shutil,sys,time
case=Path(sys.argv[1]);out=Path(sys.argv[2]);target=int(sys.argv[3]);stop=Path(sys.argv[4]);wanted=set(range(target-3000,target+1,1000))
def snap(p):
 try:
  with h5py.File(p,'r') as h:step=int(np.asarray(h['nstep']).reshape(-1)[0])
  if step not in wanted:return
  dst=out/f'step_{step:06d}.h5'
  if dst.exists():return
  tmp=out/f'.step_{step:06d}.tmp';shutil.copyfile(p,tmp)
  with h5py.File(tmp,'r') as h:
   assert int(np.asarray(h['nstep']).reshape(-1)[0])==step
   bad=[];h.visititems(lambda n,o:bad.append(n) if isinstance(o,h5py.Dataset) and o.dtype.kind in 'fc' and not np.isfinite(o[...]).all() else None);assert not bad
  os.replace(tmp,dst);print('captured',step,flush=True)
 except Exception as e:print('retry',p,repr(e),flush=True)
while True:
 snap(case/'outdat/flowfield.h5');snap(case/'bakup/flowfield.h5')
 if stop.exists():break
 time.sleep(2)
PY
STOP="$SEG/stop";python3 -u "$SEG/watch.py" "$CASE" "$HIST" "$TARGET" "$STOP" > "$SEG/logs/watcher.log" 2>&1 & WPID=$!
INPUT=$(basename "$(find "$CASE/datin" -maxdepth 1 -name 'input.astr*' -print -quit)")
set +e
(cd "$CASE" && timeout --preserve-status 330m mpirun --oversubscribe -np 4 "$OLDPWD/$EXE" run "datin/$INPUT") > "$SEG/logs/runtime.log" 2>&1
RC=$?
set -e;echo "$RC" > "$SEG/runtime_rc.txt";touch "$STOP";wait "$WPID" || true
if [[ "$RC" -ne 0 ]]; then tail -n 160 "$SEG/logs/runtime.log"; exit "$RC"; fi
if grep -Eq 'IEEE_INVALID_FLAG|SIGFPE|Floating-point exception|RANA_P_FAILURE|UPDATEFVAR_DENSITY_GATE|COMPUTATION CRASHED|(^|[^A-Za-z])(NaN|Inf)([^A-Za-z]|$)' "$SEG/logs/runtime.log"; then tail -n 160 "$SEG/logs/runtime.log"; exit 7; fi
python3 - "$CASE/outdat/flowfield.h5" "$TARGET" <<'PY'
import h5py,numpy as np,sys
p,t=sys.argv[1],int(sys.argv[2])
with h5py.File(p,'r') as h:
 assert int(np.asarray(h['nstep']).reshape(-1)[0])==t
 for k in ('ro','p','t'):
  a=np.asarray(h[k],float);assert np.isfinite(a).all() and a.min()>0
 bad=[];h.visititems(lambda n,o:bad.append(n) if isinstance(o,h5py.Dataset) and o.dtype.kind in 'fc' and not np.isfinite(o[...]).all() else None);assert not bad,bad
PY
python3 diagnostics/analyze_r26_milestone.py --snapshots "$HIST" --grid "$CASE/datin/grid.h5" --target "$TARGET" --output "$SEG/milestone_report.json"
