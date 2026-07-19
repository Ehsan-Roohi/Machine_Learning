#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path


def replace_exact(text: str, old: str, new: str, expected: int, label: str) -> str:
    count = text.count(old)
    if count != expected:
        raise RuntimeError(f"{label}: expected {expected} occurrences, found {count}")
    return text.replace(old, new)


def replace_in_subroutine(text: str, name: str, old: str, new: str, expected: int) -> str:
    pattern = re.compile(
        rf"(?P<body>\bsubroutine\s+{re.escape(name)}\b.*?\bend\s+subroutine\s+{re.escape(name)}\b)",
        re.I | re.S,
    )
    match = pattern.search(text)
    if not match:
        raise RuntimeError(f"subroutine not found: {name}")
    body = match.group("body")
    count = body.count(old)
    if count != expected:
        raise RuntimeError(f"{name}: expected {expected} occurrences of {old!r}, found {count}")
    body2 = body.replace(old, new)
    return text[:match.start("body")] + body2 + text[match.end("body"):]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source_root", type=Path)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    fludyna_path = args.source_root / "src" / "fludyna.F90"
    moment_path = args.source_root / "src" / "methodmoment.F90"
    bc_path = args.source_root / "src" / "bc.F90"
    for path in (fludyna_path, moment_path, bc_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    fludyna = fludyna_path.read_text()
    moment = moment_path.read_text()
    bc = bc_path.read_text()

    # The paper system is for Maxwell molecules.  The inherited ASTR
    # nondimensional Sutherland law is inconsistent with that model.  For
    # Maxwell molecules, mu/mu_ref = T/T_ref.
    fludyna = replace_exact(
        fludyna,
        "    use commvar, only :  tempconst,tempconst1\n",
        "    use commvar, only :  tempconst,tempconst1\n"
        "    use, intrinsic :: ieee_arithmetic, only : ieee_is_finite\n"
        "    use, intrinsic :: iso_fortran_env, only : error_unit\n",
        1,
        "miucal imports",
    )
    fludyna = replace_exact(
        fludyna,
        "      miucal=temper*sqrt(temper)*tempconst1/(temper+tempconst)",
        "      if (.not. ieee_is_finite(temper) .or. temper <= 0.d0) then\n"
        "        write(error_unit,*) 'MAXWELL_VISCOSITY_T_FAILURE',temper\n"
        "        flush(error_unit)\n"
        "        stop 'non-positive temperature in Maxwell viscosity'\n"
        "      endif\n"
        "      miucal=temper",
        1,
        "Maxwell viscosity",
    )

    # Numerical continuation in physical pseudo-time.  This avoids applying
    # the full lid much faster in physical time when the Kn/grid stability cap
    # reduces dt.  The final steady wall condition is unchanged.
    ramp_old = "min(1.0d0,dble(nstep)/1000.0d0)"
    ramp_new = "min(1.0d0,dble(nstep)*deltat/0.1d0)"
    moment = replace_exact(moment, ramp_old, ramp_new, 2, "moment lid homotopy")
    bc = replace_exact(bc, ramp_old, ramp_new, 2, "primary lid homotopy")

    # Scale wall Picard relaxation with dt.  The previous fixed 0.05 per call
    # over-forced the wall map per unit pseudo-time when dt was reduced.
    moment = replace_exact(
        moment,
        "    real(8), parameter :: bc_relax_primary=0.05d0",
        "    real(8) :: bc_relax_primary",
        2,
        "wall relaxation declaration",
    )
    moment = replace_exact(
        moment,
        "    use commvar,   only : const2,nstep,rkstep\n",
        "    use commvar,   only : const2,nstep,rkstep,deltat\n",
        1,
        "slip-map deltat import",
    )
    moment = replace_exact(
        moment,
        "    epsp=1.0d-14\n",
        "    epsp=1.0d-14\n"
        "    bc_relax_primary=0.05d0*min(1.0d0,deltat/1.0d-5)\n",
        2,
        "dt-scaled wall relaxation assignment",
    )

    # Deterministic corner ownership: horizontal walls own the four corners;
    # vertical wall maps exclude endpoints.  This removes double overwrite.
    for subroutine in ("MOM_wall_boundary", "MOM_wall_boun_init"):
        moment = replace_in_subroutine(
            moment, subroutine, "do j=0,jm", "do j=1,jm-1", 2
        )
    bc = replace_in_subroutine(
        bc, "NSslip_wall_boundary", "do j=0,jm", "do j=1,jm-1", 2
    )

    fludyna_path.write_text(fludyna)
    moment_path.write_text(moment)
    bc_path.write_text(bc)

    checks = {
        "maxwell_viscosity_linear_T": "miucal=temper" in fludyna,
        "sutherland_nondimensional_removed": "miucal=temper*sqrt(temper)" not in fludyna,
        "temperature_guard_present": "MAXWELL_VISCOSITY_T_FAILURE" in fludyna,
        "physical_time_lid_homotopy_methodmoment_count": moment.count(ramp_new),
        "physical_time_lid_homotopy_bc_count": bc.count(ramp_new),
        "dt_scaled_primary_wall_relaxation_count": moment.count(
            "bc_relax_primary=0.05d0*min(1.0d0,deltat/1.0d-5)"
        ),
        "vertical_corner_exclusion_methodmoment_count": moment.count("do j=1,jm-1"),
        "vertical_corner_exclusion_bc_count": bc.count("do j=1,jm-1"),
        "paper_equation_strings_preserved": all(x in moment for x in [
            "4.0d0/(3.0d0*pv)",
            "20.0d0/7.0d0",
            "64.0d0/(25.0d0*pv)",
            "5.0d0*rrho*sig2",
            "0.5d0*stt-Deltav/(120.0d0*theta)-Rtt/(28.0d0*theta)",
        ]),
    }
    passed = (
        checks["maxwell_viscosity_linear_T"]
        and checks["sutherland_nondimensional_removed"]
        and checks["temperature_guard_present"]
        and checks["physical_time_lid_homotopy_methodmoment_count"] == 2
        and checks["physical_time_lid_homotopy_bc_count"] == 2
        and checks["dt_scaled_primary_wall_relaxation_count"] == 2
        and checks["vertical_corner_exclusion_methodmoment_count"] >= 4
        and checks["vertical_corner_exclusion_bc_count"] >= 2
        and checks["paper_equation_strings_preserved"]
    )
    report = {
        "scope": "numerical-continuation pilot; no change to Rana bulk or smooth-face fixed-point equations",
        "changes": [
            "Maxwell-molecule viscosity mu/mu_ref=T/T_ref replaces inherited nondimensional Sutherland law",
            "lid homotopy uses physical pseudo-time tau_ramp=0.1",
            "wall Picard relaxation scales with dt relative to 1e-5",
            "horizontal walls own corners and vertical endpoint updates are excluded",
        ],
        "checks": checks,
        "all_passed": passed,
        "scientific_boundary": (
            "These changes test stability of ASTR explicit pseudo-time integration. "
            "They do not reproduce Rana et al.'s global steady fixed-point/QMR algorithm."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
