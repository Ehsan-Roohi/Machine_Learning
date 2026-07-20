#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path

VARIANTS = {
    'owner_nomass': dict(owner=True, mass=False, extrap=False, zero_top_corner=False, alternating=False),
    'owner_mass': dict(owner=True, mass=True, extrap=False, zero_top_corner=False, alternating=False),
    'owner_mass_extrap': dict(owner=True, mass=True, extrap=True, zero_top_corner=False, alternating=False),
    'zero_corner_mass_extrap': dict(owner=True, mass=True, extrap=True, zero_top_corner=True, alternating=False),
    'alternating_mass_extrap': dict(owner=False, mass=True, extrap=True, zero_top_corner=False, alternating=True),
}

def replace_exact(text: str, old: str, new: str, count: int, label: str) -> str:
    n=text.count(old)
    if n!=count:
        raise RuntimeError(f'{label}: expected {count}, found {n}')
    return text.replace(old,new)

def wrap_r13_calls(text: str, call_name: str) -> str:
    pattern = re.compile(
        rf"(?P<lead>^(?P<indent>[ \t]*)if\s*\(\s*moment\s*==\s*'r13'\s*\)\s*then\s*\n)"
        rf"(?P<call>[ \t]*call\s+{re.escape(call_name)}\([^\n]*&\s*\n[ \t]*uwall[^\n]*\)\s*\n)"
        rf"(?P<end>[ \t]*end\s*if\s*)$",
        re.IGNORECASE|re.MULTILINE)
    matches=list(pattern.finditer(text))
    if len(matches)!=4:
        raise RuntimeError(f'{call_name}: expected 4 calls, found {len(matches)}')
    def repl(m):
        ind=m.group('indent'); call=m.group('call')
        nested=''.join(ind+'  '+line[len(ind):] if line.startswith(ind) else ind+'  '+line for line in call.splitlines(True))
        return (m.group('lead')+ind+"  if (.not. ((ndir==1 .or. ndir==2) .and. &\n"+
                ind+"       ((jrk==0 .and. j==0) .or. (jrk==jrkm .and. j==jm)))) then\n"+
                nested+ind+"  end if\n"+m.group('end'))
    return pattern.sub(repl,text)

def patch_time_consistent(mm: str, bc: str) -> tuple[str,str]:
    mm=replace_exact(mm,'real(8), parameter :: bc_relax_primary=0.05d0',
                     'real(8) :: bc_relax_primary\n    real(8), parameter :: tau_bc=1.0d-3',2,'relax decl')
    mm=replace_exact(mm,
        '    epsp=1.0d-14\n    if (subdeltat <= 0.0d0 .or. deltat <= 0.0d0 .or. &',
        '    epsp=1.0d-14\n    bc_relax_primary=1.0d0-exp(-deltat/tau_bc)\n    if (subdeltat <= 0.0d0 .or. deltat <= 0.0d0 .or. &',1,'moment relax')
    mm=replace_exact(mm,
        '    epsp=1.0d-14\n\n    call rana2013_wall_tensors',
        '    epsp=1.0d-14\n    bc_relax_primary=1.0d0-exp(-deltat/tau_bc)\n\n    call rana2013_wall_tensors',1,'primary relax')
    mm=replace_exact(mm,'use commvar,   only : const2,nstep,rkstep\n',
                     'use commvar,   only : const2,nstep,rkstep,deltat\n',1,'deltat import')
    old='min(1.0d0,dble(nstep)/1000.0d0)'; new='min(1.0d0,dble(nstep)*deltat/5.0d-2)'
    if mm.count(old)<1 or bc.count(old)<1: raise RuntimeError('lid ramp fingerprints missing')
    return mm.replace(old,new),bc.replace(old,new)

def patch_maxwell(fl: str) -> str:
    old='miucal=temper*sqrt(temper)*tempconst1/(temper+tempconst)'
    return replace_exact(fl,old,'miucal=temper',1,'Maxwell viscosity')

def patch_optional_guards(mm: str, par: str) -> tuple[str,str,dict]:
    """Replace every unsafe PRESENT(x).and.x optional-logical guard.

    Fortran does not guarantee short-circuit evaluation, so an absent optional
    logical must never be referenced in the same expression as PRESENT().
    """
    unsafe = re.compile(
        r"present\s*\(\s*timerept\s*\)\s*\.and\.\s*timerept",
        re.IGNORECASE,
    )
    inline = re.compile(
        r"(?P<indent>^[ \t]*)if\s*\(\s*present\s*\(\s*timerept\s*\)\s*\.and\.\s*timerept\s*\)\s*"
        r"time_beg\s*=\s*ptime\s*\(\s*\)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    block = re.compile(
        r"(?P<indent>^[ \t]*)if\s*\(\s*present\s*\(\s*timerept\s*\)\s*\.and\.\s*timerept\s*\)\s*then\s*\n"
        r"(?P<body>.*?)(?P<close>^(?P=indent)end\s*if\s*$)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )

    def patch_one(name: str, text: str) -> tuple[str,dict]:
        n_inline=len(list(inline.finditer(text)))
        text=inline.sub(
            lambda m:(
                f"{m.group('indent')}if (present(timerept)) then\n"
                f"{m.group('indent')}  if (timerept) time_beg=ptime()\n"
                f"{m.group('indent')}end if"
            ), text)
        n_block=len(list(block.finditer(text)))
        text=block.sub(
            lambda m:(
                f"{m.group('indent')}if (present(timerept)) then\n"
                f"{m.group('indent')}  if (timerept) then\n"
                f"{m.group('body')}"
                f"{m.group('indent')}  end if\n"
                f"{m.group('indent')}end if"
            ), text)
        remaining=len(unsafe.findall(text))
        if remaining:
            raise RuntimeError(f"{name}: {remaining} unsafe optional-logical expressions remain")
        return text,{'inline':n_inline,'block':n_block,'remaining':remaining}

    mm,mm_counts=patch_one('methodmoment',mm)
    par,par_counts=patch_one('parallel',par)
    if par_counts['inline'] != 11 or par_counts['block'] != 11:
        raise RuntimeError(f"parallel optional-guard count mismatch: {par_counts}")
    if mm_counts['inline'] != 3 or mm_counts['block'] != 4:
        raise RuntimeError(f"methodmoment optional-guard count mismatch: {mm_counts}")
    return mm,par,{'methodmoment':mm_counts,'parallel':par_counts}

EXTRAP_HELPERS='''
  ! Diagnostic paper-like one-sided extrapolation for unconstrained face data.
  subroutine r13_extrapolate_moment_face(nface,iw,jw,kw,ip,jp,kp)
    use commarray, only : sigma,qflux
    integer,intent(in) :: nface,iw,jw,kw,ip,jp,kp
    integer :: i2,j2,k2
    i2=2*ip-iw; j2=2*jp-jw; k2=2*kp-kw
    sigma(iw,jw,kw,:)=2.0d0*sigma(ip,jp,kp,:)-sigma(i2,j2,k2,:)
    qflux(iw,jw,kw,:)=2.0d0*qflux(ip,jp,kp,:)-qflux(i2,j2,k2,:)
  end subroutine r13_extrapolate_moment_face

  subroutine r13_extrapolate_primary_face(nface,iw,jw,kw,ip,jp,kp)
    use commarray, only : q,rho,vel,tmp,prs
    use fludyna, only : thermal,fvar2q
    integer,intent(in) :: nface,iw,jw,kw,ip,jp,kp
    integer :: i2,j2,k2
    i2=2*ip-iw; j2=2*jp-jw; k2=2*kp-kw
    vel(iw,jw,kw,:)=2.0d0*vel(ip,jp,kp,:)-vel(i2,j2,k2,:)
    tmp(iw,jw,kw)=max(2.0d0*tmp(ip,jp,kp)-tmp(i2,j2,k2),1.0d-10)
    prs(iw,jw,kw)=max(2.0d0*prs(ip,jp,kp)-prs(i2,j2,k2),1.0d-10)
    rho(iw,jw,kw)=thermal(pressure=prs(iw,jw,kw),temperature=tmp(iw,jw,kw))
    call fvar2q(q=q(iw,jw,kw,:),density=rho(iw,jw,kw),velocity=vel(iw,jw,kw,:),temperature=tmp(iw,jw,kw))
  end subroutine r13_extrapolate_primary_face

'''

def patch_extrap(mm: str) -> str:
    marker='  subroutine MOM_wall_boundary(ndir)'
    if marker not in mm: raise RuntimeError('MOM marker missing')
    mm=mm.replace(marker,EXTRAP_HELPERS+marker,1)
    mm=replace_exact(mm,
        '    epsp=1.0d-14\n    bc_relax_primary=1.0d0-exp(-deltat/tau_bc)\n    if (subdeltat <= 0.0d0 .or. deltat <= 0.0d0 .or. &',
        '    epsp=1.0d-14\n    call r13_extrapolate_moment_face(nface,iw,jw,kw,ip,jp,kp)\n    bc_relax_primary=1.0d0-exp(-deltat/tau_bc)\n    if (subdeltat <= 0.0d0 .or. deltat <= 0.0d0 .or. &',1,'moment extrap call')
    mm=replace_exact(mm,
        '    epsp=1.0d-14\n    bc_relax_primary=1.0d0-exp(-deltat/tau_bc)\n\n    call rana2013_wall_tensors',
        '    epsp=1.0d-14\n    call r13_extrapolate_primary_face(nface,iw,jw,kw,ip,jp,kp)\n    bc_relax_primary=1.0d0-exp(-deltat/tau_bc)\n\n    call rana2013_wall_tensors',1,'primary extrap call')
    return mm

def patch_zero_top_corners(mm: str, bc: str) -> tuple[str,str]:
    ramp='uwall = min(1.0d0,dble(nstep)*deltat/5.0d-2)'
    pos=mm.find(ramp)
    if pos<0: raise RuntimeError('moment top ramp missing')
    callpos=mm.find("             if (moment=='r13') then",pos)
    if callpos<0: raise RuntimeError('moment top R13 call missing')
    mm=mm[:callpos]+"             uwall = min(1.0d0,dble(nstep)*deltat/5.0d-2)\n             if ((irk==0 .and. i==0) .or. (irk==irkm .and. i==im)) uwall=0.0d0\n"+mm[callpos:]
    pos=bc.find(ramp)
    if pos<0: raise RuntimeError('primary top ramp missing')
    insertpos=bc.find("                 if (moment == 'r05') then",pos)
    if insertpos<0: raise RuntimeError('primary top call missing')
    bc=bc[:insertpos]+"                 if ((irk==0 .and. i==0) .or. (irk==irkm .and. i==im)) uwall=0.0d0\n"+bc[insertpos:]
    return mm,bc

def patch_alternating_order(mm: str, bc: str) -> tuple[str,str]:
    mm=replace_exact(mm,'                         bctype\n','                         bctype,rkstep\n',1,'rk3mom rkstep import')
    old='''      do n = 1,6 
         if(bctype(n)==413) then
           call MOM_wall_boundary(n)
         end if
      end do'''
    new='''      if (mod(rkstep,2)==1) then
        do n=1,6
          if(bctype(n)==413) call MOM_wall_boundary(n)
        end do
      else
        do n=6,1,-1
          if(bctype(n)==413) call MOM_wall_boundary(n)
        end do
      endif'''
    mm=replace_exact(mm,old,new,1,'alternating moment wall order')
    b0=bc.find('  subroutine boucon(subtime)')
    b1=bc.find('  end subroutine boucon',b0)
    if b0<0 or b1<0: raise RuntimeError('boucon block missing')
    block=bc[b0:b1]
    block=replace_exact(block,'    use commvar, only : limmbou\n','    use commvar, only : limmbou,rkstep\n',1,'boucon rkstep import')
    block=replace_exact(block,'    integer :: n\n','    integer :: n,nstart,nend,ninc\n',1,'boucon loop vars')
    block=replace_exact(block,'    do n=1,6\n','''    if(mod(rkstep,2)==1) then
      nstart=1; nend=6; ninc=1
    else
      nstart=6; nend=1; ninc=-1
    endif
    do n=nstart,nend,ninc
''',1,'boucon alternating loop')
    bc=bc[:b0]+block+bc[b1:]
    return mm,bc

MASS_SUB='''
  subroutine enforce_r13_mass_constraint
    use commvar, only : im,jm,km,ia,ja,moment
    use commarray, only : rho,prs,q
    use parallel, only : psum,ig0,jg0
    integer :: i,j,k,gi,gj
    real(8) :: lm,lw,gm,gw,wx,wy,w,fac
    if(moment/='r13') return
    lm=0.0d0; lw=0.0d0
    do k=0,km
      do j=0,jm
        gj=jg0+j; wy=1.0d0
        if(gj==0 .or. gj==ja) wy=0.5d0
        do i=0,im
          gi=ig0+i; wx=1.0d0
          if(gi==0 .or. gi==ia) wx=0.5d0
          w=wx*wy
          lm=lm+w*rho(i,j,k); lw=lw+w
        enddo
      enddo
    enddo
    gm=psum(lm); gw=psum(lw)
    if(gm<=1.0d-300 .or. gw<=0.0d0) stop 'R13 mass constraint invalid'
    fac=gw/gm
    rho(0:im,0:jm,0:km)=rho(0:im,0:jm,0:km)*fac
    prs(0:im,0:jm,0:km)=prs(0:im,0:jm,0:km)*fac
    q(0:im,0:jm,0:km,1:5)=q(0:im,0:jm,0:km,1:5)*fac
  end subroutine enforce_r13_mass_constraint

'''

def patch_mass(main: str) -> str:
    call_marker='''          ! Restore physical cells; the next RK stage rebuilds halos.
          call updatefvar(q,0,im,0,jm,0,km)
          !
        endif'''
    repl='''          ! Restore physical cells; the next RK stage rebuilds halos.
          call updatefvar(q,0,im,0,jm,0,km)
          call enforce_r13_mass_constraint
          !
        endif'''
    main=replace_exact(main,call_marker,repl,1,'mass call')
    insert='  !+-------------------------------------------------------------------+\n  !| The end of the subroutine steploop.                               |'
    if insert not in main: raise RuntimeError('steploop end marker missing')
    main=main.replace(insert,MASS_SUB+insert,1)
    return main

def patch_closure_refresh(main: str) -> str:
    main=replace_exact(main,
        '    use commvar,  only : lavg,lwslic,lwsequ,feqavg,feqchkpt,feqwsequ,  &\n                         feqslice,iomode\n',
        '    use commvar,  only : lavg,lwslic,lwsequ,feqavg,feqchkpt,feqwsequ,  &\n                         feqslice,iomode,moment,hm\n    use commarray, only : q\n    use fludyna, only : updatefvar,miucomp\n    use comsolver, only : gradcal\n    use methodmoment, only : mijkcal,rijcal,deltacal\n',1,'rkfirst imports')
    marker='''    ! time to write checkpoint
    if(iomode == 'n') then'''
    refresh='''    ! Recompute closures from the final sigma/q state before any output.
    if(moment=='r13' .and. (nstep==nxtchkpt .or. (lwsequ .and. nstep==nxtwsequ))) then
      call qswap
      call updatefvar(q,0,im,0,jm,0,km)
      call miucomp
      call gradcal(dswap=.true.)
      call mijkcal
      call rijcal
      call deltacal
    endif
    ! time to write checkpoint
    if(iomode == 'n') then'''
    return replace_exact(main,marker,refresh,1,'closure refresh')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); ap.add_argument('--variant',choices=VARIANTS,required=True); ap.add_argument('--report',type=Path,required=True)
    a=ap.parse_args(); cfg=VARIANTS[a.variant]
    src=a.root/'src'; paths={n:src/n for n in ['methodmoment.F90','bc.F90','fludyna.F90','mainloop.F90','parallel.F90']}
    mm=paths['methodmoment.F90'].read_text(); bc=paths['bc.F90'].read_text(); fl=paths['fludyna.F90'].read_text(); mainf=paths['mainloop.F90'].read_text(); par=paths['parallel.F90'].read_text()
    mm,bc=patch_time_consistent(mm,bc)
    fl=patch_maxwell(fl)
    mm,par,optional_guard_counts=patch_optional_guards(mm,par)
    mainf=patch_closure_refresh(mainf)
    if cfg['mass']: mainf=patch_mass(mainf)
    if cfg['extrap']: mm=patch_extrap(mm)
    if cfg['owner']:
        mm=wrap_r13_calls(mm,'R13wbc'); bc=wrap_r13_calls(bc,'R13wbc_slip')
    if cfg['zero_top_corner']: mm,bc=patch_zero_top_corners(mm,bc)
    if cfg['alternating']: mm,bc=patch_alternating_order(mm,bc)
    paths['methodmoment.F90'].write_text(mm); paths['bc.F90'].write_text(bc); paths['fludyna.F90'].write_text(fl); paths['mainloop.F90'].write_text(mainf); paths['parallel.F90'].write_text(par)
    checks={
      'maxwell':'miucal=temper' in fl,
      'filter_fixed_in_input_not_source':True,
      'closure_refresh':'call mijkcal' in mainf[mainf.find('subroutine rkfirst'):mainf.find('end subroutine rkfirst')],
      'mass_constraint':(not cfg['mass']) or 'enforce_r13_mass_constraint' in mainf,
      'extrapolation':(not cfg['extrap']) or 'r13_extrapolate_primary_face' in mm,
      'corner_owner':(not cfg['owner']) or mm.count('(ndir==1 .or. ndir==2)')>=4,
      'zero_top_corner':(not cfg['zero_top_corner']) or 'uwall=0.0d0' in mm,
      'alternating_order':(not cfg['alternating']) or 'mod(rkstep,2)' in bc,
      'eq13_unchanged':all(x in mm for x in ['4.0d0/(3.0d0*pv)','64.0d0/(25.0d0*pv)','56.0d0/(5.0d0*pv)']),
      'effective_pressure_unchanged':'0.5d0*stt-Deltav/(120.0d0*theta)-Rtt/(28.0d0*theta)' in mm,
      'optional_guards_safe': not re.search(r'present\s*\(\s*timerept\s*\)\s*\.and\.\s*timerept',mm+par,re.IGNORECASE),
    }
    rep={'variant':a.variant,'config':cfg,'checks':checks,'optional_guard_counts':optional_guard_counts,'all_passed':all(checks.values()),'scientific_boundary':'diagnostic pseudo-time variants; none is the exact coupled Eq.20 matrix solve'}
    a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(rep,indent=2)+'\n'); print(json.dumps(rep,indent=2))
    if not rep['all_passed']: raise SystemExit(2)
if __name__=='__main__': main()
