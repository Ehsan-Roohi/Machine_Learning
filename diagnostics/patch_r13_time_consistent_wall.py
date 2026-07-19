#!/usr/bin/env python3
from pathlib import Path
import argparse, json, re


def replace_exact(text: str, old: str, new: str, count: int, label: str) -> str:
    n = text.count(old)
    if n != count:
        raise RuntimeError(f'{label}: expected {count} occurrences, found {n}')
    return text.replace(old, new)


def wrap_r13_calls(text: str, call_name: str) -> tuple[str, int]:
    pattern = re.compile(
        rf"(?P<lead>^(?P<indent>[ \t]*)if\s*\(\s*moment\s*==\s*'r13'\s*\)\s*then\s*\n)"
        rf"(?P<call>[ \t]*call\s+{re.escape(call_name)}\([^\n]*&\s*\n[ \t]*uwall[^\n]*\)\s*\n)"
        rf"(?P<end>[ \t]*end\s*if\s*)$",
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if len(matches) != 4:
        raise RuntimeError(f'{call_name}: expected four wall call blocks, found {len(matches)}')

    def repl(m: re.Match[str]) -> str:
        indent = m.group('indent')
        call = m.group('call')
        nested_call = ''.join(
            indent + '  ' + line[len(indent):] if line.startswith(indent) else indent + '  ' + line
            for line in call.splitlines(keepends=True)
        )
        return (
            m.group('lead')
            + indent + "  if (.not. ((ndir==1 .or. ndir==2) .and. &\n"
            + indent + "       ((jrk==0 .and. j==0) .or. (jrk==jrkm .and. j==jm)))) then\n"
            + nested_call
            + indent + "  end if\n"
            + m.group('end')
        )

    return pattern.sub(repl, text), len(matches)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('source_root', type=Path)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()

    mm = a.source_root / 'src/methodmoment.F90'
    bc = a.source_root / 'src/bc.F90'
    mt = mm.read_text()
    bt = bc.read_text()

    mt = replace_exact(
        mt,
        'real(8), parameter :: bc_relax_primary=0.05d0',
        'real(8) :: bc_relax_primary\n    real(8), parameter :: tau_bc=1.0d-3',
        2,
        'wall relaxation declaration',
    )
    mt = replace_exact(
        mt,
        '    epsp=1.0d-14\n    if (subdeltat <= 0.0d0 .or. deltat <= 0.0d0 .or. &',
        '    epsp=1.0d-14\n    bc_relax_primary=1.0d0-exp(-deltat/tau_bc)\n    if (subdeltat <= 0.0d0 .or. deltat <= 0.0d0 .or. &',
        1,
        'moment wall relaxation assignment',
    )
    mt = replace_exact(
        mt,
        '    epsp=1.0d-14\n\n    call rana2013_wall_tensors',
        '    epsp=1.0d-14\n    bc_relax_primary=1.0d0-exp(-deltat/tau_bc)\n\n    call rana2013_wall_tensors',
        1,
        'primary wall relaxation assignment',
    )
    mt = replace_exact(
        mt,
        'use commvar,   only : const2,nstep,rkstep\n',
        'use commvar,   only : const2,nstep,rkstep,deltat\n',
        1,
        'primary wall deltat import',
    )

    ramp_old = 'min(1.0d0,dble(nstep)/1000.0d0)'
    ramp_new = 'min(1.0d0,dble(nstep)*deltat/5.0d-2)'
    mt_ramps = mt.count(ramp_old)
    bt_ramps = bt.count(ramp_old)
    if mt_ramps < 1 or bt_ramps < 1:
        raise RuntimeError(f'lid ramp not found: method={mt_ramps}, bc={bt_ramps}')
    mt = mt.replace(ramp_old, ramp_new)
    bt = bt.replace(ramp_old, ramp_new)

    mt, moment_blocks = wrap_r13_calls(mt, 'R13wbc')
    bt, slip_blocks = wrap_r13_calls(bt, 'R13wbc_slip')

    mm.write_text(mt)
    bc.write_text(bt)

    checks = {
        'time_consistent_relaxation_assignments': mt.count('bc_relax_primary=1.0d0-exp(-deltat/tau_bc)') == 2,
        'tau_bc_declarations': mt.count('real(8), parameter :: tau_bc=1.0d-3') == 2,
        'physical_time_ramp_method': ramp_new in mt and ramp_old not in mt,
        'physical_time_ramp_bc': ramp_new in bt and ramp_old not in bt,
        'four_moment_wall_blocks_guarded': moment_blocks == 4 and mt.count('(ndir==1 .or. ndir==2)') >= 4,
        'four_slip_wall_blocks_guarded': slip_blocks == 4 and bt.count('(ndir==1 .or. ndir==2)') >= 4,
        'rana_effective_pressure_unchanged': '0.5d0*stt-Deltav/(120.0d0*theta)-Rtt/(28.0d0*theta)' in mt,
        'eq13_fingerprints_unchanged': all(x in mt for x in [
            '4.0d0/(3.0d0*pv)', '64.0d0/(25.0d0*pv)', '56.0d0/(5.0d0*pv)'
        ]),
    }
    report = {
        'purpose': 'physics-preserving numerical-path stabilization of exact nonlinear Rana R13',
        'tau_bc': 1.0e-3,
        'lid_ramp_pseudotime': 5.0e-2,
        'corner_owner': 'horizontal walls; vertical-wall R13 calls skip only global physical corners',
        'method_ramp_replacements': mt_ramps,
        'bc_ramp_replacements': bt_ramps,
        'checks': checks,
        'all_passed': all(checks.values()),
        'scientific_boundary': 'bulk equations and smooth-face fixed point unchanged; corner ownership and tau_bc require sensitivity testing',
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['all_passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
