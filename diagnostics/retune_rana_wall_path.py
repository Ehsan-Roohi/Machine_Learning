#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fortran_real(value: float) -> str:
    return f'{value:.12e}'.replace('e', 'd')


def replace_exact(text: str, old: str, new: str, count: int, label: str) -> str:
    found = text.count(old)
    if found != count:
        raise RuntimeError(f'{label}: expected {count} occurrences, found {found}')
    return text.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('source_root', type=Path)
    parser.add_argument('--tau-bc', type=float, required=True)
    parser.add_argument('--ramp-time', type=float, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    method = source_root / 'src/methodmoment.F90'
    bc = source_root / 'src/bc.F90'
    mt = method.read_text()
    bt = bc.read_text()

    tau_old = 'real(8), parameter :: tau_bc=1.0d-3'
    tau_new = f'real(8), parameter :: tau_bc={fortran_real(args.tau_bc)}'
    mt = replace_exact(mt, tau_old, tau_new, 2, 'tau_bc declarations')

    ramp_old = 'dble(nstep)*deltat/5.0d-2'
    ramp_new = f'dble(nstep)*deltat/{fortran_real(args.ramp_time)}'
    method_count = mt.count(ramp_old)
    bc_count = bt.count(ramp_old)
    if method_count < 1 or bc_count < 1:
        raise RuntimeError(f'physical-time ramp fingerprint missing: method={method_count}, bc={bc_count}')
    mt = mt.replace(ramp_old, ramp_new)
    bt = bt.replace(ramp_old, ramp_new)

    method.write_text(mt)
    bc.write_text(bt)

    checks = {
        'tau_declarations': mt.count(tau_new) == 2,
        'method_ramp': ramp_new in mt and ramp_old not in mt,
        'bc_ramp': ramp_new in bt and ramp_old not in bt,
        'eq13_unchanged': all(x in mt for x in [
            '4.0d0/(3.0d0*pv)',
            '64.0d0/(25.0d0*pv)',
            '56.0d0/(5.0d0*pv)',
        ]),
        'effective_pressure_unchanged': '0.5d0*stt-Deltav/(120.0d0*theta)-Rtt/(28.0d0*theta)' in mt,
        'no_pressure_floor': 'max(Pv' not in mt and 'max(pv' not in mt,
    }
    report = {
        'tau_bc': args.tau_bc,
        'ramp_time': args.ramp_time,
        'checks': checks,
        'passed': all(checks.values()),
        'scientific_scope': 'Numerical sensitivity only. Eq. (13), smooth-face Eq. (7), effective pressure, accommodation, grid, and lid endpoint are unchanged. No clipping/floor is introduced.',
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
