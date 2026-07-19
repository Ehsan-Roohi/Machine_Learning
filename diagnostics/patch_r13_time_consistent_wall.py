#!/usr/bin/env python3
from pathlib import Path
import argparse, json, re


def replace_exact(text, old, new, count):
    n=text.count(old)
    if n != count:
        raise RuntimeError(f'expected {count} occurrences of {old!r}, found {n}')
    return text.replace(old,new)

p=argparse.ArgumentParser()
p.add_argument('source_root',type=Path)
p.add_argument('--report',type=Path,required=True)
a=p.parse_args()
mm=a.source_root/'src/methodmoment.F90'
bc=a.source_root/'src/bc.F90'
mt=mm.read_text(); bt=bc.read_text()

# Preserve the exact Rana fixed point, but make the iteration rate depend on
# physical pseudo-time rather than raw step count.  tau_bc=1e-3 gives about
# 0.00995 per outer step at dt=1e-5 and proportionally smaller updates for
# the high-Kn cases whose stable dt is much smaller.
mt=replace_exact(mt,'real(8), parameter :: bc_relax_primary=0.05d0',
'''real(8) :: bc_relax_primary
    real(8), parameter :: tau_bc=1.0d-3''',2)
mt=replace_exact(mt,
'''    epsp=1.0d-14
    if (subdeltat <= 0.0d0 .or. deltat <= 0.0d0 .or. &''',
'''    epsp=1.0d-14
    bc_relax_primary=1.0d0-exp(-deltat/tau_bc)
    if (subdeltat <= 0.0d0 .or. deltat <= 0.0d0 .or. &''',1)
mt=replace_exact(mt,
'''    epsp=1.0d-14

    call rana2013_wall_tensors''',
'''    epsp=1.0d-14
    bc_relax_primary=1.0d0-exp(-deltat/tau_bc)

    call rana2013_wall_tensors''',1)
mt=replace_exact(mt,'use commvar,   only : const2,nstep,rkstep\n',
                    'use commvar,   only : const2,nstep,rkstep,deltat\n',1)

# Use the same physical-time lid ramp in both the primary and moment wall
# maps.  The final boundary value is unchanged; only the startup path changes.
ramp_old='min(1.0d0,dble(nstep)/1000.0d0)'
ramp_new='min(1.0d0,dble(nstep)*deltat/5.0d-2)'
if mt.count(ramp_old) < 1 or bt.count(ramp_old) < 1:
    raise RuntimeError('expected lid-ramp fingerprints were not found')
mt=mt.replace(ramp_old,ramp_new)
bt=bt.replace(ramp_old,ramp_new)

# Give horizontal walls ownership of physical corners.  This removes the
# previous double update with two incompatible normals while leaving every
# smooth-face equation unchanged.  Apply only to the dedicated R13 path.
old_call='''             if (moment=='r13') then
                call R13wbc(ndir, n1, n2, n3, i, j, k, ip, jp, kp, &
                        uwall, vwall, wwall, twall, alpha)
             end if'''
new_call='''             if (moment=='r13') then
                if (.not. ((jrk==0 .and. j==0) .or. (jrk==jrkm .and. j==jm))) then
                  call R13wbc(ndir, n1, n2, n3, i, j, k, ip, jp, kp, &
                          uwall, vwall, wwall, twall, alpha)
                end if
             end if'''
if mt.count(old_call) != 2:
    raise RuntimeError(f'expected two vertical R13 moment-wall calls, found {mt.count(old_call)}')
mt=mt.replace(old_call,new_call)

old_slip='''                  if (moment == 'r13') then
                   call R13wbc_slip(ndir, n1, n2, n3, i, j, k, ip, jp, kp,   &
                        uwall, vwall, wwall, twall, alpha)
                  end if'''
new_slip='''                  if (moment == 'r13') then
                   if (.not. ((jrk==0 .and. j==0) .or. (jrk==jrkm .and. j==jm))) then
                     call R13wbc_slip(ndir, n1, n2, n3, i, j, k, ip, jp, kp,   &
                          uwall, vwall, wwall, twall, alpha)
                   end if
                  end if'''
if bt.count(old_slip) != 2:
    raise RuntimeError(f'expected two vertical R13 slip-wall calls, found {bt.count(old_slip)}')
bt=bt.replace(old_slip,new_slip)

mm.write_text(mt); bc.write_text(bt)
checks={
 'time_consistent_relaxation':mt.count('bc_relax_primary=1.0d0-exp(-deltat/tau_bc)')==2,
 'physical_time_ramp_method':ramp_new in mt,
 'physical_time_ramp_bc':ramp_new in bt,
 'unique_corner_owner_moment':mt.count('jrk==0 .and. j==0')==2,
 'unique_corner_owner_slip':bt.count('jrk==0 .and. j==0')==2,
 'rana_effective_pressure_unchanged':'0.5d0*stt-Deltav/(120.0d0*theta)-Rtt/(28.0d0*theta)' in mt,
 'eq13_unchanged':'4.0d0/(3.0d0*pv)' in mt and '64.0d0/(25.0d0*pv)' in mt and '56.0d0/(5.0d0*pv)' in mt,
}
report={'purpose':'physics-preserving numerical-path stabilization of exact nonlinear Rana R13','checks':checks,'all_passed':all(checks.values()),'scientific_boundary':'bulk equations and smooth-face fixed point unchanged; horizontal-wall corner ownership is an explicit numerical convention requiring sensitivity testing'}
a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
if not report['all_passed']: raise SystemExit(2)
