#!/usr/bin/env python3
"""Conservative source audit for a distinct, formula-complete LR13 model.

This script never runs ASTR.  It records evidence that the currently audited
solver is nonlinear R13 and prevents that code from being relabelled LR13.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


FILES = ("methodmoment.F90", "mainloop.F90", "solver.F90", "readwrite.F90")


def occurrences(text: str, pattern: str) -> list[dict[str, object]]:
    rx = re.compile(pattern, re.IGNORECASE)
    return [
        {"line": number, "text": line.strip()[:240]}
        for number, line in enumerate(text.splitlines(), 1)
        if rx.search(line)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for name in FILES:
        path = args.source_root / name
        if not path.is_file():
            raise SystemExit(f"missing required source: {path}")
        raw = path.read_bytes()
        sources[name] = raw.decode("utf-8", errors="replace")
        hashes[name] = hashlib.sha256(raw).hexdigest()

    method = sources["methodmoment.F90"]
    mainloop = sources["mainloop.F90"]
    solver = sources["solver.F90"]
    readwrite = sources["readwrite.F90"]
    combined = "\n".join(sources.values())

    evidence = {
        "explicit_lr13_token": occurrences(combined, r"['\"]lr13['\"]"),
        "active_unconditional_nonlinear_blocks": occurrences(
            method, r"if\s*\(\s*\.true\.\s*\)"
        ),
        "nonlinear_moment_advection": occurrences(
            method, r"q_mom\s*\([^\n]*\)\s*\*\s*uu"
        ),
        "state_dependent_mu_T_over_p": occurrences(
            method, r"miu\s*\([^\n]*tmp\s*\([^\n]*/\s*prs\s*\("
        ),
        "nonlinear_wall_products": occurrences(
            method, r"slip_[ts]\s*\*\*\s*[234]|slip_t\s*\*\s*slip_s"
        ),
        "nonlinear_base_flux_path": occurrences(
            solver, r"call\s+convrsd(?:cal6|uwd|cmp)"
        ),
        "r13_only_routing": (
            occurrences(mainloop, r"moment\s*==\s*['\"]r13['\"]")
            + occurrences(solver, r"moment\s*==\s*['\"]r13['\"]")
            + occurrences(readwrite, r"moment\s*==\s*['\"]r13['\"]")
        ),
        "derived_corner_policy": occurrences(method, r"lr13[^\n]*corner|corner[^\n]*lr13"),
        "linear_formula_oracle": occurrences(combined, r"lr13[^\n]*(oracle|epsilon|linearization)"),
    }

    gates = {
        "distinct_lr13_model_label_and_routing": bool(evidence["explicit_lr13_token"]),
        "all_five_primary_balances_are_O_epsilon_linear": False,
        "stress_and_heat_flux_balances_are_O_epsilon_linear": False,
        "regularizing_moments_use_equilibrium_frozen_coefficients": False,
        "complete_linear_wall_boundary_system_is_implemented": False,
        "wall_corner_system_is_derived_and_documented": bool(evidence["derived_corner_policy"]),
        "symbolic_O_epsilon_equation_oracle_passes": bool(evidence["linear_formula_oracle"]),
        "strict_FPE_2k_smoke_passes": False,
        "strict_FPE_20k_diagnostic_passes": False,
    }

    blockers = [
        "No independent `lr13` model token or dispatch path exists.",
        "The primary mass, momentum, and energy solver still uses the nonlinear compressible flux path.",
        "The moment equations retain q_mom*u advection and active unconditional nonlinear source terms.",
        "Regularizer coefficients retain local mu*T/p dependence instead of an explicitly frozen equilibrium coefficient.",
        "The wall map contains quadratic/higher slip products and is the nonlinear Rana map, not a complete LR13 boundary system.",
        "No derived LR13 corner system or symbolic first-order equation oracle is present.",
    ]

    report = {
        "schema_version": 1,
        "audited_model": "ASTR nonlinear R13 (not LR13)",
        "ready_for_2k_smoke": False,
        "ready_for_20k_diagnostic": False,
        "solver_execution_authorized": False,
        "action": "BLOCK_BEFORE_SOLVER_EXECUTION",
        "source_sha256": hashes,
        "evidence": evidence,
        "required_gates": gates,
        "blockers": blockers,
        "authoritative_references": [
            {
                "title": "Rana, Torrilhon & Struchtrup (2013), A robust numerical method for the R13 equations",
                "url": "https://www.engr.uvic.ca/~struchtr/2013_JCP_Lidcavity.pdf",
                "scope": "nonlinear balances, regularized closures, and nonlinear wall relations",
            },
            {
                "title": "Lin et al. (2025), Time-dependent R13 equations with Onsager boundary conditions in the linear regime",
                "url": "https://doi.org/10.1017/jfm.2025.215",
                "scope": "explicit linear perturbation system and complete Onsager boundary formulation",
            },
            {
                "title": "Cai, Torrilhon & Yang (2024), Linear regularized 13-moment equations",
                "url": "https://doi.org/10.1137/23M1556472",
                "scope": "well-posed steady LR13 formulation",
            },
        ],
        "note": (
            "Keeping ASTR's CFD discretization is compatible with the project goal, "
            "but numerical similarity does not replace a complete LR13 equation and boundary implementation."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "action": report["action"],
        "ready_for_2k_smoke": report["ready_for_2k_smoke"],
        "ready_for_20k_diagnostic": report["ready_for_20k_diagnostic"],
        "blocker_count": len(blockers),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
