#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 TARGET_STEP PREVIOUS_STEP CASE_DIR ASTR_BINARY" >&2
  exit 2
fi

TARGET="$1"
PREVIOUS="$2"
CASE_DIR="$3"
ASTR_BINARY="$4"
LABEL=$(printf 'step_%05d' "$TARGET")
PREVIOUS_LABEL=$(printf 'step_%05d' "$PREVIOUS")

mkdir -p logs summaries convergence checkpoints

python3 - "$CASE_DIR" "$TARGET" <<'PY'
import sys
from pathlib import Path

case = Path(sys.argv[1])
target = int(sys.argv[2])
controller = case / "datin" / "controller"
lines = controller.read_text(encoding="utf-8").splitlines()
for index, line in enumerate(lines):
    if line.strip().startswith("# maxstep,feqchkpt"):
        lines[index + 1] = (
            f"{target:9d},    2000,    2000,     500,     200,   200"
        )
        break
else:
    raise RuntimeError("Controller maxstep entry not found")
controller.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

INPUT=$(basename "$(find "$CASE_DIR/datin" -maxdepth 1 -name 'input.astr*' -print -quit)")
test -n "$INPUT"

(
  cd "$CASE_DIR"
  /usr/bin/time -v mpirun --oversubscribe -np 4 "$ASTR_BINARY" run "datin/${INPUT}" \
    > "../logs/${LABEL}.log" 2> "../logs/${LABEL}_time.log"
)

! grep -q 'COMPUTATION CRASHED' "logs/${LABEL}.log"
python3 astr_r26_stage1/validate_r26_case.py "$CASE_DIR" \
  --target-step "$TARGET" --output "summaries/${LABEL}.json"

python3 - "$PREVIOUS" "$TARGET" "$CASE_DIR" <<'PY'
import h5py
import json
import numpy as np
import shutil
import sys
from pathlib import Path

previous = int(sys.argv[1])
target = int(sys.argv[2])
case = Path(sys.argv[3])
previous_label = f"step_{previous:05d}"
label = f"step_{target:05d}"
fields = ["ro", "p", "t", "u1", "u2", "qx", "qy", "Rxx", "Rxy", "Ryy", "Delta"]
prev_flow = Path("checkpoints") / previous_label / "flowfield.h5"
cur_flow = case / "outdat" / "flowfield.h5"
report = {
    "previous_step": previous,
    "current_step": target,
    "relative_RMS_change_percent": {},
}
with h5py.File(prev_flow, "r") as old, h5py.File(cur_flow, "r") as new:
    if int(old["nstep"][0]) != previous:
        raise RuntimeError("Previous checkpoint step mismatch")
    if int(new["nstep"][0]) != target:
        raise RuntimeError("Current checkpoint step mismatch")
    for key in fields:
        a = np.asarray(old[key], dtype=float)
        b = np.asarray(new[key], dtype=float)
        report["relative_RMS_change_percent"][key] = float(
            100.0 * np.sqrt(np.mean((b - a) ** 2))
            / (np.sqrt(np.mean(b ** 2)) + 1.0e-30)
        )

out = Path("convergence") / f"{previous_label}_to_{label}.json"
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
checkpoint = Path("checkpoints") / label
checkpoint.mkdir(parents=True, exist_ok=False)
shutil.copy2(cur_flow, checkpoint / "flowfield.h5")
shutil.copy2(case / "outdat" / "auxiliary.h5", checkpoint / "auxiliary.h5")
print(json.dumps(report, indent=2))
PY
