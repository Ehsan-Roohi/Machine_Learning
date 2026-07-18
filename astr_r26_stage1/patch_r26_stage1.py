#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the conservative Stage-1 audit patch to ASTR mom_merge R26. "
            "Only the Apsi1 exponent typo is corrected; disabled nonlinear blocks "
            "remain disabled and are reported."
        )
    )
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--report", type=Path, default=Path("r26_stage1_source_audit.json"))
    args = parser.parse_args()

    method_file = args.source_root / "src" / "methodmoment.F90"
    if not method_file.is_file():
        raise FileNotFoundError(method_file)

    original = method_file.read_text(encoding="utf-8")
    pattern = re.compile(r"Apsi1\s*=\s*1\.698d9")
    matches = list(pattern.finditer(original))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one Apsi1=1.698d9 occurrence, found {len(matches)}"
        )

    patched = pattern.sub("Apsi1=1.698d0", original, count=1)
    method_file.write_text(patched, encoding="utf-8")

    disabled_lines = [
        index
        for index, line in enumerate(patched.splitlines(), start=1)
        if re.search(r"if\s*\(\s*\.false\.\s*\)\s*then", line, flags=re.I)
    ]

    audit = {
        "patch_scope": "Stage-1 conservative R26 pilot",
        "source_file": str(method_file),
        "Apsi1_before": "1.698d9",
        "Apsi1_after": "1.698d0",
        "Apsi1_replacements": 1,
        "disabled_nonlinear_blocks_retained": len(disabled_lines),
        "disabled_block_start_lines": disabled_lines,
        "important_limitation": (
            "The existing if(.false.) nonlinear R26 source blocks are intentionally "
            "not enabled in Stage 1. Results must be labelled audited Maxwell/semi-linear R26."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
