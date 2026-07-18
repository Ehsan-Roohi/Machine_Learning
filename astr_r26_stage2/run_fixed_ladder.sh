#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 CASE_DIR ASTR_BINARY START_STEP END_STEP INTERVAL" >&2
  exit 2
fi

CASE_DIR="$1"
ASTR_BINARY="$2"
START_STEP="$3"
END_STEP="$4"
INTERVAL="$5"

if (( END_STEP <= START_STEP )); then
  echo "END_STEP must be larger than START_STEP" >&2
  exit 2
fi
if (( (END_STEP - START_STEP) % INTERVAL != 0 )); then
  echo "Requested range is not divisible by INTERVAL" >&2
  exit 2
fi

mkdir -p logs summaries convergence checkpoints
INPUT=$(basename "$(find "$CASE_DIR/datin" -maxdepth 1 -name 'input.astr*' -print -quit)")
test -n "$INPUT"

python3 - "$CASE_DIR" "$START_STEP" <<'PY'
import h5py
import shutil
import sys
from pathlib import Path

case = Path(sys.argv[1])
start = int(sys.argv[2])
flow = case / "outdat" / "flowfield.h5"
aux = case / "outdat" / "auxiliary.h5"
with h5py.File(flow, "r") as h:
    if int(h["nstep"][0]) != start:
        raise RuntimeError(f"Seed checkpoint mismatch: {int(h['nstep'][0])} != {start}")

input_path = next((case / "datin").glob("input.astr*"))
lines = input_path.read_text(encoding="utf-8").splitlines()
for index, line in enumerate(lines):
    if line.strip().startswith("# lrestar"):
        lines[index + 1] = "t"
        break
else:
    raise RuntimeError("lrestar entry not found")
input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

label = f"step_{start:05d}"
destination = Path("checkpoints") / label
destination.mkdir(parents=True, exist_ok=True)
shutil.copy2(flow, destination / "flowfield.h5")
shutil.copy2(aux, destination / "auxiliary.h5")
PY

PREVIOUS="$START_STEP"
for TARGET in $(seq $((START_STEP + INTERVAL)) "$INTERVAL" "$END_STEP"); do
  python3 - "$CASE_DIR" "$TARGET" "$INTERVAL" <<'PY'
import sys
from pathlib import Path

case = Path(sys.argv[1])
target = int(sys.argv[2])
interval = int(sys.argv[3])
controller = case / "datin" / "controller"
lines = controller.read_text(encoding="utf-8").splitlines()
for index, line in enumerate(lines):
    if line.strip().startswith("# maxstep,feqchkpt"):
        lines[index + 1] = (
            f"{target:9d}, {interval:7d}, {interval:7d},     500,     200,   200"
        )
        break
else:
    raise RuntimeError("Controller maxstep entry not found")
controller.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

  LABEL=$(printf 'step_%05d' "$TARGET")
  PREVIOUS_LABEL=$(printf 'step_%05d' "$PREVIOUS")
  (
    cd "$CASE_DIR"
    /usr/bin/time -v mpirun --oversubscribe -np 4 "$ASTR_BINARY" run "datin/${INPUT}" \
      > "../logs/${LABEL}.log" 2> "../logs/${LABEL}_time.log"
  )
  ! grep -q 'COMPUTATION CRASHED' "logs/${LABEL}.log"
  python3 astr_r26_stage1/validate_r26_case.py "$CASE_DIR" \
    --target-step "$TARGET" --output "summaries/${LABEL}.json"
  python3 astr_r26_stage2/checkpoint_convergence.py \
    --previous "checkpoints/${PREVIOUS_LABEL}/flowfield.h5" \
    --current "$CASE_DIR/outdat/flowfield.h5" \
    --output "convergence/${PREVIOUS_LABEL}_to_${LABEL}.json"
  mkdir "checkpoints/${LABEL}"
  cp "$CASE_DIR/outdat/flowfield.h5" "checkpoints/${LABEL}/"
  cp "$CASE_DIR/outdat/auxiliary.h5" "checkpoints/${LABEL}/"
  PREVIOUS="$TARGET"
done
