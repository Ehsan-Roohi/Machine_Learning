#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_rana2013_discrete_operator import audit_source


def write_tree(root: Path, *, paper_operator: bool) -> None:
    src = root / "src"
    src.mkdir(parents=True)
    mm = """
subroutine rk3mom
do n=1,6
  call MOM_wall_boundary(n)
enddo
do j=0,jm
enddo
do i=0,im
enddo
subroutine R13wbc
bc_relax_primary=0.05d0
bc_relax_moment=0.01d0
x=4.0d0/(3.0d0*pv)+64.0d0/(25.0d0*pv)+56.0d0/(5.0d0*pv)
x=0.5d0*stt-Deltav/(120.0d0*theta)-Rtt/(28.0d0*theta)
end subroutine R13wbc
subroutine R13wbc_slip
end subroutine R13wbc_slip
"""
    bc = """
do n=1,6
  call NSslip_wall_boundary(n)
enddo
do j=0,jm
enddo
do i=0,im
enddo
"""
    main = "call rk3\n"
    if paper_operator:
        mm += """
! rana2013_state_order 17
! rana2013 x_plus x_minus y_plus y_minus
! rana2013 eq20 corner_operator x_plus y_plus
! rana2013 global steady matrix assembly
! rana2013 bordered mass constraint
! qmr
! rana2013 nonlinear fixed_point 1.0d-6
"""
    (src / "methodmoment.F90").write_text(mm)
    (src / "bc.F90").write_text(bc)
    (src / "mainloop.F90").write_text(main)


def test_current_style_is_blocked() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_tree(root, paper_operator=False)
        report = audit_source(root)
        assert report["production_run_gate"] == "STOP"
        assert report["equation_level_checks"]["eq13_transformed_closure_fingerprints"]
        assert report["observed_astr_path"]["corner_nodes_included_in_each_face_loop"]
        assert not report["paper_discrete_checks"]["eq20_coupled_corner_row"]


def test_explicit_paper_operator_can_open_gate() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_tree(root, paper_operator=True)
        report = audit_source(root)
        assert report["production_run_gate"] == "GO"
        assert report["paper_discrete_checks"]["eq20_coupled_corner_row"]
        assert report["paper_discrete_checks"]["eq24_eq25_bordered_mass_system"]


if __name__ == "__main__":
    test_current_style_is_blocked()
    test_explicit_paper_operator_can_open_gate()
    print("rana2013 discrete-operator audit tests: PASS")
