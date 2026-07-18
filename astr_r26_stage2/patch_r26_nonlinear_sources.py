#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path

TARGET_SUBROUTINES = {
    "src_delta": "nonlinear N term in the Delta equation",
    "src_rij_B": "nonlinear S_ij terms in the R_ij equations",
    "src_mijk_B": "nonlinear M_ijk terms in the m_ijk equations",
}


def patch_one_subroutine(text: str, name: str) -> tuple[str, dict[str, object]]:
    pattern = re.compile(
        rf"(?P<body>\bsubroutine\s+{re.escape(name)}\b.*?\bend\s+subroutine\s+{re.escape(name)}\b)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        raise RuntimeError(f"Subroutine not found: {name}")

    body = match.group("body")
    disabled = list(re.finditer(r"if\s*\(\s*\.false\.\s*\)\s*then", body, flags=re.IGNORECASE))
    if len(disabled) != 1:
        raise RuntimeError(
            f"Expected exactly one disabled nonlinear block in {name}; found {len(disabled)}"
        )

    patched_body = re.sub(
        r"if\s*\(\s*\.false\.\s*\)\s*then",
        "if (.true.) then  ! R26 nonlinear source enabled by audited Stage-2 patch",
        body,
        count=1,
        flags=re.IGNORECASE,
    )
    updated = text[: match.start("body")] + patched_body + text[match.end("body") :]
    return updated, {
        "subroutine": name,
        "enabled_block_count": 1,
        "purpose": TARGET_SUBROUTINES[name],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Enable the existing Gu-Emerson nonlinear evolution-source blocks for "
            "m_ijk, R_ij and Delta. This does not enable experimental wall extrapolation "
            "blocks and does not claim complete nonlinear regularization closures."
        )
    )
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--patch-output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_root / "src" / "methodmoment.F90"
    if not source.is_file():
        raise FileNotFoundError(source)

    original = source.read_text(encoding="utf-8")
    patched = original
    records: list[dict[str, object]] = []
    for name in TARGET_SUBROUTINES:
        patched, record = patch_one_subroutine(patched, name)
        records.append(record)

    # The two remaining disabled blocks are deliberately wall/extrapolation experiments.
    remaining_false = [
        i
        for i, line in enumerate(patched.splitlines(), start=1)
        if re.search(r"if\s*\(\s*\.false\.\s*\)\s*then", line, flags=re.IGNORECASE)
    ]
    if len(remaining_false) != 2:
        raise RuntimeError(
            f"Expected two deliberately disabled extrapolation blocks to remain; found {len(remaining_false)}"
        )

    source.write_text(patched, encoding="utf-8")
    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile="methodmoment.F90.stage1",
            tofile="methodmoment.F90.nonlinear-source-v1",
        )
    )
    args.patch_output.parent.mkdir(parents=True, exist_ok=True)
    args.patch_output.write_text(diff, encoding="utf-8")

    report = {
        "model_label": "Audited Maxwell R26 nonlinear-source v1",
        "enabled_evolution_terms": records,
        "paper_mapping": {
            "src_mijk_B": "M_ijk in Gu-Emerson equation (19)",
            "src_rij_B": "S_ij in Gu-Emerson equation (20)",
            "src_delta": "N in Gu-Emerson equation (21)",
        },
        "deliberately_disabled_blocks_remaining": remaining_false,
        "limitations": [
            "The non-gradient remainder terms phi_R, psi_R and Omega_R are not newly completed by this patch.",
            "Existing high-order wall treatment is retained unchanged.",
            "Collision coefficients remain Maxwell-molecule coefficients, not VHS argon omega=0.81.",
            "This is an experimental nonlinear-source path and must be validated before publication use.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
