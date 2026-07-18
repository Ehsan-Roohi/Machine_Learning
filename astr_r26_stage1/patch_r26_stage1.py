#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the conservative Stage-1 audit patch to ASTR mom_merge R26: "
            "correct Apsi1 and use full lid speed from the first step. Disabled "
            "nonlinear R26 blocks remain disabled and are reported."
        )
    )
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--report", type=Path, default=Path("r26_stage1_source_audit.json"))
    parser.add_argument("--patch-output", type=Path, default=Path("ASTR_R26_STAGE1.patch"))
    args = parser.parse_args()

    method_file = args.source_root / "src" / "methodmoment.F90"
    bc_file = args.source_root / "src" / "bc.F90"
    for path in [method_file, bc_file]:
        if not path.is_file():
            raise FileNotFoundError(path)

    originals = {
        method_file: method_file.read_text(encoding="utf-8"),
        bc_file: bc_file.read_text(encoding="utf-8"),
    }

    method_text = originals[method_file]
    apsi_pattern = re.compile(r"Apsi1\s*=\s*1\.698d9")
    if len(list(apsi_pattern.finditer(method_text))) != 1:
        raise RuntimeError("Expected exactly one Apsi1=1.698d9 occurrence")
    method_text = apsi_pattern.sub("Apsi1=1.698d0", method_text, count=1)

    ramp_pattern = re.compile(
        r"uwall\s*=\s*dble\(nstep\)\s*\*\s*deltat(?:\s*\*\s*10\.0d0)?",
        flags=re.I,
    )
    method_ramps = len(list(ramp_pattern.finditer(method_text)))
    bc_text = originals[bc_file]
    bc_ramps = len(list(ramp_pattern.finditer(bc_text)))
    if method_ramps != 2 or bc_ramps != 2:
        raise RuntimeError(
            f"Unexpected lid-ramp occurrence counts: methodmoment={method_ramps}, bc={bc_ramps}"
        )
    method_text = ramp_pattern.sub("uwall = 1.0d0", method_text)
    bc_text = ramp_pattern.sub("uwall = 1.0d0", bc_text)

    method_file.write_text(method_text, encoding="utf-8")
    bc_file.write_text(bc_text, encoding="utf-8")

    disabled_lines = [
        index
        for index, line in enumerate(method_text.splitlines(), start=1)
        if re.search(r"if\s*\(\s*\.false\.\s*\)\s*then", line, flags=re.I)
    ]

    patch_parts: list[str] = []
    for path, patched in [(method_file, method_text), (bc_file, bc_text)]:
        patch_parts.extend(
            difflib.unified_diff(
                originals[path].splitlines(keepends=True),
                patched.splitlines(keepends=True),
                fromfile=str(path) + ".upstream",
                tofile=str(path) + ".stage1",
            )
        )
    args.patch_output.parent.mkdir(parents=True, exist_ok=True)
    args.patch_output.write_text("".join(patch_parts), encoding="utf-8")

    audit = {
        "patch_scope": "Stage-1 short-gate R26 pilot",
        "Apsi1_before": "1.698d9",
        "Apsi1_after": "1.698d0",
        "Apsi1_replacements": 1,
        "methodmoment_lid_ramps_replaced": method_ramps,
        "bc_lid_ramps_replaced": bc_ramps,
        "lid_condition": "uwall=1.0d0 from first step",
        "disabled_nonlinear_blocks_retained": len(disabled_lines),
        "disabled_block_start_lines": disabled_lines,
        "important_limitation": (
            "The existing if(.false.) nonlinear R26 source blocks are intentionally "
            "not enabled. Results must be labelled audited Maxwell/semi-linear R26 Stage 1."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
