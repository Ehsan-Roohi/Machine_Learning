#!/usr/bin/env python3
"""Audit whether an ASTR source tree implements Rana et al.'s *discrete* method.

The existing formula audit checks individual R13 closure and wall-equation
coefficients.  That is necessary, but it does not establish equivalence to the
steady 17-state discretization in Rana, Torrilhon & Struchtrup (2013),
Eqs. (11), (14)--(25).  This audit deliberately treats those as separate gates.

It is a source-structure audit, not a proof of numerical correctness.  A pass
means only that the required architectural pieces are explicit enough to be
reviewed and tested; a fail prevents a run from being labelled paper-exact.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REFERENCE = (
    "A. Rana, M. Torrilhon and H. Struchtrup, Journal of Computational "
    "Physics 236 (2013) 169--186, doi:10.1016/j.jcp.2012.11.023"
)


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL) is not None


def _line_numbers(text: str, pattern: str) -> list[int]:
    rx = re.compile(pattern, re.IGNORECASE)
    return [i for i, line in enumerate(text.splitlines(), 1) if rx.search(line)]


def audit_source(root: Path) -> dict[str, Any]:
    src = root / "src"
    required = {
        "methodmoment": src / "methodmoment.F90",
        "bc": src / "bc.F90",
        "mainloop": src / "mainloop.F90",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required source files: " + ", ".join(missing))

    mm = required["methodmoment"].read_text(errors="replace")
    bc = required["bc"].read_text(errors="replace")
    main = required["mainloop"].read_text(errors="replace")
    all_source = "\n".join(path.read_text(errors="replace") for path in src.glob("*.F90"))

    # These checks intentionally preserve the distinction between equation-level
    # coefficient fidelity and the paper's discrete nonlinear solver.
    equation_level = {
        "eq13_transformed_closure_fingerprints": all(
            token in mm
            for token in (
                "4.0d0/(3.0d0*pv)",
                "64.0d0/(25.0d0*pv)",
                "56.0d0/(5.0d0*pv)",
            )
        ),
        "eq7_effective_pressure_fingerprint": (
            "0.5d0*stt-Deltav/(120.0d0*theta)-Rtt/(28.0d0*theta)" in mm
        ),
        "eq7_wall_maps_present": "subroutine R13wbc" in mm
        and "subroutine R13wbc_slip" in mm,
    }

    observed_astr_path = {
        "explicit_pseudotime_rk": _has(main, r"call\s+rk3\b")
        and _has(mm, r"subroutine\s+rk3mom\b"),
        "primary_faces_applied_sequentially": _has(
            bc,
            r"do\s+n\s*=\s*1\s*,\s*6.*?call\s+NSslip_wall_boundary\s*\(\s*n\s*\)",
        ),
        "moment_faces_applied_sequentially": _has(
            mm,
            r"do\s+n\s*=\s*1\s*,\s*6.*?call\s+MOM_wall_boundary\s*\(\s*n\s*\)",
        ),
        "corner_nodes_included_in_each_face_loop": all(
            _has(text, pattern)
            for text, pattern in (
                (mm, r"do\s+j\s*=\s*0\s*,\s*jm"),
                (mm, r"do\s+i\s*=\s*0\s*,\s*im"),
                (bc, r"do\s+j\s*=\s*0\s*,\s*jm"),
                (bc, r"do\s+i\s*=\s*0\s*,\s*im"),
            )
        ),
        "damped_local_wall_projection": "bc_relax_primary" in mm
        and "bc_relax_moment" in mm,
    }

    # A paper-exact implementation must expose these pieces explicitly.  The
    # accepted names are intentionally descriptive and narrow: matching generic
    # words such as "matrix" or "mass" would create dangerous false positives.
    paper_discrete = {
        "eq11_17_state_ordering": _has(
            all_source,
            r"rana(?:2013)?[_ ](?:state|unknown)[_ ](?:order|ordering).*?17",
        ),
        "eq15_eq16_boundary_matrices": _has(
            all_source,
            r"(?:rana(?:2013)?[_ ])?(?:xplus|xminus|yplus|yminus|x_plus|x_minus|y_plus|y_minus)",
        ),
        "eq20_coupled_corner_row": _has(
            all_source,
            r"(?:rana(?:2013)?[_ ])?(?:eq20|corner[_ ](?:row|block|operator)).*?(?:xplus|x_plus).*?(?:yplus|y_plus)",
        ),
        "eq21_global_steady_assembly": _has(
            all_source,
            r"rana(?:2013)?[_ ](?:global[_ ]steady|steady|global)[_ ](?:matrix|operator|assembly)",
        ),
        "eq24_eq25_bordered_mass_system": _has(
            all_source,
            r"rana(?:2013)?[_ ](?:bordered|left[_ ]null|right[_ ]null).*?(?:mass|constraint)",
        ),
        "qmr_linear_solve": _has(all_source, r"\bqmr\b"),
        "paper_fixed_point_norm_1e_minus_6": _has(
            all_source,
            r"rana(?:2013)?[_ ](?:fixed[_ ]point|nonlinear).*?(?:1\.0?d-6|1e-6)",
        ),
    }

    critical = (
        "eq15_eq16_boundary_matrices",
        "eq20_coupled_corner_row",
        "eq21_global_steady_assembly",
        "eq24_eq25_bordered_mass_system",
        "qmr_linear_solve",
    )
    paper_exact_ready = all(paper_discrete[name] for name in critical)

    evidence = {
        "methodmoment_face_loop_lines": _line_numbers(
            mm, r"call\s+MOM_wall_boundary\s*\(\s*n\s*\)"
        ),
        "primary_face_loop_lines": _line_numbers(
            bc, r"call\s+NSslip_wall_boundary\s*\(\s*n\s*\)"
        ),
        "moment_corner_inclusive_i_loop_lines": _line_numbers(mm, r"do\s+i\s*=\s*0\s*,\s*im"),
        "moment_corner_inclusive_j_loop_lines": _line_numbers(mm, r"do\s+j\s*=\s*0\s*,\s*jm"),
        "primary_corner_inclusive_i_loop_lines": _line_numbers(bc, r"do\s+i\s*=\s*0\s*,\s*im"),
        "primary_corner_inclusive_j_loop_lines": _line_numbers(bc, r"do\s+j\s*=\s*0\s*,\s*jm"),
    }

    return {
        "reference": REFERENCE,
        "scope": "discrete-operator architecture; not a convergence or solution-validation test",
        "equation_level_checks": equation_level,
        "observed_astr_path": observed_astr_path,
        "paper_discrete_checks": paper_discrete,
        "evidence_lines": evidence,
        "paper_exact_ready": paper_exact_ready,
        "production_run_gate": "GO" if paper_exact_ready else "STOP",
        "ranked_root_causes": [
            {
                "rank": 1,
                "finding": "Corner boundary equations are not coupled.",
                "impact": (
                    "The same corner node is visited by two separate face loops; the later face "
                    "can overwrite the earlier one. Rana Eq. (20) contains both A*X and B*Y "
                    "contributions in one row. This directly contaminates moving-lid shear D."
                ),
            },
            {
                "rank": 2,
                "finding": "Smooth-wall equations are applied as damped local projections.",
                "impact": (
                    "A local Eq. (7) fixed point is not the same discrete equation as inserting "
                    "the X/Y boundary matrices into the central-difference PDE rows, Eqs. (15)--(19)."
                ),
            },
            {
                "rank": 3,
                "finding": "The paper's bordered null-space mass constraint is absent.",
                "impact": (
                    "Post-step density/pressure rescaling, when enabled, is not equivalent to "
                    "Eqs. (23)--(25) and can alter the nonlinear iteration."
                ),
            },
            {
                "rank": 4,
                "finding": "The solve path is explicit pseudo-time RK, not the steady QMR iteration.",
                "impact": (
                    "This can still converge to a valid steady solution only after an independent "
                    "discrete-residual equivalence proof; that proof and a 1e-6 fixed-point gate are absent."
                ),
            },
        ],
        "next_action": (
            "Implement and unit-test the 17-state A/B/P and X/Y matrix oracle, including an "
            "Eq. (20) two-wall corner row and the bordered mass row, before another production run."
        ),
        "scientific_label": (
            "Equation-level diagnostic only; current ASTR path must not be labelled paper-exact, "
            "converged, validated, or publication-grade."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("astr", type=Path, help="ASTR root containing src/*.F90")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fail-on-stop",
        action="store_true",
        help="return status 2 when the paper-exact production gate is STOP",
    )
    args = parser.parse_args()
    report = audit_source(args.astr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.fail_on_stop and not report["paper_exact_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
