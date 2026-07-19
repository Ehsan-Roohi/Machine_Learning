#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def subroutine_match(text: str, name: str) -> re.Match[str]:
    match = re.search(
        rf"(?P<body>\bsubroutine\s+{re.escape(name)}\b.*?\bend\s+subroutine\s+{re.escape(name)}\b)",
        text,
        re.I | re.S,
    )
    if not match:
        raise RuntimeError(f"Missing subroutine: {name}")
    return match


def replace_in_subroutine(
    text: str,
    name: str,
    old: str,
    new: str,
    expected: int,
) -> str:
    match = subroutine_match(text, name)
    body = match.group("body")
    count = body.count(old)
    if count != expected:
        raise RuntimeError(
            f"{name}: expected {expected} occurrences of {old!r}; found {count}"
        )
    body = body.replace(old, new)
    return text[: match.start("body")] + body + text[match.end("body") :]


def replace_all_lid_assignments(
    text: str,
    name: str,
    replacement: str,
    expected: int,
) -> str:
    match = subroutine_match(text, name)
    body = match.group("body")
    pattern = re.compile(r"uwall\s*=\s*1\.0d0")
    hits = list(pattern.finditer(body))
    if len(hits) != expected:
        raise RuntimeError(
            f"{name}: expected {expected} full-lid assignments; found {len(hits)}"
        )
    body = pattern.sub(replacement, body)
    return text[: match.start("body")] + body + text[match.end("body") :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    fludyna_path = args.source_root / "src" / "fludyna.F90"
    moment_path = args.source_root / "src" / "methodmoment.F90"
    bc_path = args.source_root / "src" / "bc.F90"
    for path in (fludyna_path, moment_path, bc_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    fludyna = fludyna_path.read_text()
    moment = moment_path.read_text()
    bc = bc_path.read_text()

    # Gu--Emerson R26 coefficients in this branch are Maxwell-molecule
    # coefficients.  Replace the inherited nondimensional Sutherland law by
    # the Maxwell scaling mu/mu_ref = T/T_ref and trap invalid temperatures at
    # their first use.
    import_line = "    use commvar, only :  tempconst,tempconst1\n"
    if fludyna.count(import_line) != 1:
        raise RuntimeError("Unexpected miucal import block")
    fludyna = fludyna.replace(
        import_line,
        import_line
        + "    use, intrinsic :: ieee_arithmetic, only : ieee_is_finite\n"
        + "    use, intrinsic :: iso_fortran_env, only : error_unit\n",
    )

    old_viscosity = "      miucal=temper*sqrt(temper)*tempconst1/(temper+tempconst)"
    if fludyna.count(old_viscosity) != 1:
        raise RuntimeError("Unexpected Sutherland miucal occurrence count")
    fludyna = fludyna.replace(
        old_viscosity,
        "      if (.not. ieee_is_finite(temper) .or. temper <= 0.d0) then\n"
        "        write(error_unit,*) 'MAXWELL_VISCOSITY_T_FAILURE',temper\n"
        "        flush(error_unit)\n"
        "        stop 'non-positive temperature in Maxwell viscosity'\n"
        "      endif\n"
        "      miucal=temper",
    )

    # Stage 1 set the full lid from the first iteration.  Use a physical
    # pseudo-time continuation instead.  The final wall speed and all R26
    # bulk/WBC formulas are unchanged.
    ramp = "min(1.0d0,dble(nstep)*deltat/0.1d0)"
    moment = replace_all_lid_assignments(
        moment, "MOM_wall_boundary", f"uwall = {ramp}", 3
    )
    moment = replace_all_lid_assignments(
        moment, "MOM_wall_boun_init", f"uwall = {ramp}", 3
    )
    bc = replace_all_lid_assignments(
        bc, "NSslip_wall_boundary", f"uwall = {ramp}", 5
    )

    # Deterministic corner ownership: horizontal walls own the four corners;
    # vertical wall loops exclude endpoints.  This removes the inherited
    # double overwrite while retaining the smooth-face R26 wall map.
    moment = replace_in_subroutine(
        moment, "MOM_wall_boundary", "do j=0,jm", "do j=1,jm-1", 2
    )
    moment = replace_in_subroutine(
        moment, "MOM_wall_boun_init", "do j=0,jm", "do j=1,jm-1", 2
    )
    bc = replace_in_subroutine(
        bc, "NSslip_wall_boundary", "do j=0,jm", "do j=1,jm-1", 2
    )

    fludyna_path.write_text(fludyna)
    moment_path.write_text(moment)
    bc_path.write_text(bc)

    checks = {
        "maxwell_viscosity_linear_T": (
            "miucal=temper" in fludyna
            and "miucal=temper*sqrt" not in fludyna
        ),
        "temperature_guard_present": "MAXWELL_VISCOSITY_T_FAILURE" in fludyna,
        "methodmoment_physical_time_lid_assignments": moment.count(ramp),
        "bc_physical_time_lid_assignments": bc.count(ramp),
        "methodmoment_vertical_corner_exclusions": moment.count("do j=1,jm-1"),
        "bc_vertical_corner_exclusions": bc.count("do j=1,jm-1"),
        "r26_v3_bulk_calls_preserved": all(
            item in moment
            for item in [
                "call r26_full_phi_closure()",
                "call r26_full_psi_closure()",
                "call r26_full_omega_closure()",
            ]
        ),
        "nonlinear_evolution_sources_preserved": all(
            re.search(
                rf"subroutine\s+{name}.*?if \(\.true\.\) then\s*! R26 nonlinear source",
                moment,
                re.I | re.S,
            )
            for name in ["src_mijk_B", "src_rij_B", "src_delta"]
        ),
        "relaxed_wall_memory_update_retained": (
            "deltatp = subdeltat" in moment
            and "deltatt = subdeltat" in moment
        ),
    }
    passed = (
        checks["maxwell_viscosity_linear_T"]
        and checks["temperature_guard_present"]
        and checks["methodmoment_physical_time_lid_assignments"] >= 6
        and checks["bc_physical_time_lid_assignments"] >= 5
        and checks["methodmoment_vertical_corner_exclusions"] >= 4
        and checks["bc_vertical_corner_exclusions"] >= 2
        and checks["r26_v3_bulk_calls_preserved"]
        and checks["nonlinear_evolution_sources_preserved"]
        and checks["relaxed_wall_memory_update_retained"]
    )

    report = {
        "scope": (
            "Maxwell transport law, physical-time lid continuation, and "
            "deterministic corner ownership; R26 v3 bulk equations and smooth-face "
            "wall formulas are unchanged"
        ),
        "checks": checks,
        "all_passed": bool(passed),
        "remaining_scientific_holds": [
            "R26 wall-memory arrays are still not serialized in restart files",
            "The complete Gu--Emerson equations (19)--(26) still need an independent component-wise tensor oracle",
            "This ASTR path remains an explicit pseudo-time diagnostic, not a paper-equivalent steady nonlinear solver",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
