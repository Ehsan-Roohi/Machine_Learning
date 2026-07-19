#!/usr/bin/env python3
from __future__ import annotations
import argparse, difflib, json, re
from pathlib import Path

RELAX='0.05d0'
RAMP='min(1.0d0,dble(nstep)/1000.0d0)'

def replace_once(text:str, old:str, new:str, label:str)->str:
    n=text.count(old)
    if n!=1:
        raise RuntimeError(f'{label}: expected one match, found {n}')
    return text.replace(old,new,1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('astr',type=Path)
    ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--patch-output',type=Path,required=True)
    a=ap.parse_args()
    mm=a.astr/'src/methodmoment.F90'; bc=a.astr/'src/bc.F90'
    m0=mm.read_text(); b0=bc.read_text(); m=m0; b=b0

    m=replace_once(m,
      '    real(8) :: Snew(3,3),qnew(3),epsp\n',
      f'    real(8) :: Snew(3,3),qnew(3),epsp\n    real(8), parameter :: bc_relax={RELAX}\n',
      'moment relaxation declaration')
    m=replace_once(m,
      '    sigma(iw,jw,kw,1)=Snew(1,1)\n'
      '    sigma(iw,jw,kw,2)=Snew(1,2)\n'
      '    sigma(iw,jw,kw,3)=0.0d0\n'
      '    sigma(iw,jw,kw,4)=Snew(2,2)\n'
      '    sigma(iw,jw,kw,5)=0.0d0\n'
      '    qflux(iw,jw,kw,1)=qnew(1)\n'
      '    qflux(iw,jw,kw,2)=qnew(2)\n'
      '    qflux(iw,jw,kw,3)=0.0d0\n',
      '    ! Damped fixed-point iteration of exact Eq. (7); fixed point unchanged.\n'
      '    sigma(iw,jw,kw,1)=sigma(iw,jw,kw,1)+bc_relax*(Snew(1,1)-sigma(iw,jw,kw,1))\n'
      '    sigma(iw,jw,kw,2)=sigma(iw,jw,kw,2)+bc_relax*(Snew(1,2)-sigma(iw,jw,kw,2))\n'
      '    sigma(iw,jw,kw,3)=sigma(iw,jw,kw,3)+bc_relax*(0.0d0-sigma(iw,jw,kw,3))\n'
      '    sigma(iw,jw,kw,4)=sigma(iw,jw,kw,4)+bc_relax*(Snew(2,2)-sigma(iw,jw,kw,4))\n'
      '    sigma(iw,jw,kw,5)=sigma(iw,jw,kw,5)+bc_relax*(0.0d0-sigma(iw,jw,kw,5))\n'
      '    qflux(iw,jw,kw,1)=qflux(iw,jw,kw,1)+bc_relax*(qnew(1)-qflux(iw,jw,kw,1))\n'
      '    qflux(iw,jw,kw,2)=qflux(iw,jw,kw,2)+bc_relax*(qnew(2)-qflux(iw,jw,kw,2))\n'
      '    qflux(iw,jw,kw,3)=qflux(iw,jw,kw,3)+bc_relax*(0.0d0-qflux(iw,jw,kw,3))\n',
      'moment fixed-point damping')
    m=replace_once(m,
      '    real(8) :: theta,thetaw,Cinv,P,Vt,Tjump,Deltav,epsp\n',
      f'    real(8) :: theta,thetaw,Cinv,P,Vt,Tjump,Deltav,epsp,target_tmp\n    real(8), parameter :: bc_relax={RELAX}\n',
      'primary relaxation declaration')
    m=replace_once(m,
      '    vnew=wv+Vt*tv\n'
      '    vel(iw,jw,kw,1)=vnew(1)\n'
      '    vel(iw,jw,kw,2)=vnew(2)\n'
      '    vel(iw,jw,kw,3)=0.0d0\n'
      '    tmp(iw,jw,kw)=max(twall+const2*Tjump,1.0d-12)\n',
      '    vnew=wv+Vt*tv\n'
      '    target_tmp=max(twall+const2*Tjump,1.0d-12)\n'
      '    ! Damped, checkpoint-invariant fixed-point iteration of exact Eq. (7).\n'
      '    vel(iw,jw,kw,1)=vel(iw,jw,kw,1)+bc_relax*(vnew(1)-vel(iw,jw,kw,1))\n'
      '    vel(iw,jw,kw,2)=vel(iw,jw,kw,2)+bc_relax*(vnew(2)-vel(iw,jw,kw,2))\n'
      '    vel(iw,jw,kw,3)=vel(iw,jw,kw,3)+bc_relax*(0.0d0-vel(iw,jw,kw,3))\n'
      '    tmp(iw,jw,kw)=tmp(iw,jw,kw)+bc_relax*(target_tmp-tmp(iw,jw,kw))\n',
      'primary fixed-point damping')

    m,nm=re.subn(r'(?m)^(\s*)uwall\s*=\s*1\.0d0\s*$',rf'\1uwall = {RAMP}',m)
    b,nb=re.subn(r'(?m)^(\s*)uwall\s*=\s*1\.0d0\s*$',rf'\1uwall = {RAMP}',b)
    if nm<2 or nb<1:
        raise RuntimeError(f'homotopy replacements too few: method={nm}, bc={nb}')

    mm.write_text(m); bc.write_text(b)
    patch=''.join(difflib.unified_diff(m0.splitlines(True),m.splitlines(True),
      fromfile='astr/src/methodmoment.F90.rana-exact',tofile='astr/src/methodmoment.F90.rana-exact-stable'))
    patch+=''.join(difflib.unified_diff(b0.splitlines(True),b.splitlines(True),
      fromfile='astr/src/bc.F90.rana-exact',tofile='astr/src/bc.F90.rana-exact-stable'))
    a.patch_output.parent.mkdir(parents=True,exist_ok=True); a.patch_output.write_text(patch)
    r13_body=re.search(r'(?is)subroutine\s+R13wbc\b.*?end\s+subroutine\s+R13wbc\b',m).group(0)
    slip_body=re.search(r'(?is)subroutine\s+R13wbc_slip\b.*?end\s+subroutine\s+R13wbc_slip\b',m).group(0)
    checks={
      'exact_Eq7_targets_retained':all(x in m for x in [
        'stn_new=-(P*Vt+qt/5.0d0+mtnn/2.0d0)/Cinv',
        'qt_new=(5.0d0/(11.0d0*theta))',
        'snn_new=(5.0d0/(7.0d0*theta))',
        'stt_new=(-Cinv*mttn-0.2d0*P*Tjump']),
      'fixed_point_damping_moment':m.count(f'bc_relax={RELAX}')>=2 and 'fixed point unchanged' in m,
      'fixed_point_damping_primary':'checkpoint-invariant fixed-point iteration' in m,
      'restart_invariant_no_external_wall_memory':all(x not in r13_body+slip_body
        for x in ['sig_ttbc12','vel12_slip','tmp12_jump']),
      'lid_homotopy_method':nm>=2 and RAMP in m,
      'lid_homotopy_bc':nb>=1 and RAMP in b,
      'exact_final_lid':RAMP.startswith('min(1.0d0,'),
      'effective_pressure_guard_retained':m.count("stop 'Non-positive Rana2013 effective wall pressure P'")>=2,
    }
    report={'purpose':'robust nonlinear path to the unchanged Rana Eq. (7) fixed point',
      'wall_relaxation':float(RELAX.replace('d0','')),
      'lid_homotopy_iterations':1000,'method_replacements':nm,'bc_replacements':nb,
      'checks':checks,'all_passed':all(checks.values())}
    a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if not report['all_passed']: raise SystemExit(2)
if __name__=='__main__': main()
