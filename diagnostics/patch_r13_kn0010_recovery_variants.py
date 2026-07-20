#!/usr/bin/env python3
from pathlib import Path
import argparse, json, re

VARIANTS={
 'owner_nomass':dict(owner=True,mass=False,extrap=False,zero_top_corner=False,alternating=False),
 'owner_mass':dict(owner=True,mass=True,extrap=False,zero_top_corner=False,alternating=False),
 'owner_mass_extrap':dict(owner=True,mass=True,extrap=True,zero_top_corner=False,alternating=False),
 'zero_corner_mass_extrap':dict(owner=True,mass=True,extrap=True,zero_top_corner=True,alternating=False),
 'alternating_mass_extrap':dict(owner=False,mass=True,extrap=True,zero_top_corner=False,alternating=True),
}

def replace_exact(text,old,new,count,label):
    n=text.count(old)
    if n!=count: raise RuntimeError(f'{label}: expected {count}, found {n}')
    return text.replace(old,new)

def patch_time_consistent(mm: str, bc: str) -> tuple[str,str]:
    mm=replace_exact(mm,'real(8), parameter :: bc_relax_primary=0.05d0','real(8) :: bc_relax_primary\n    real(8), parameter :: tau_bc=1.0d-3',2,'wall relaxation declarations')
    mm=replace_exact(mm,'    epsp=1.0d-14\n    if (subdeltat <= 0.0d0 .or. deltat <= 0.0d0 .or. &','    epsp=1.0d-14\n    bc_relax_primary=1.0d0-exp(-deltat/tau_bc)\n    if (subdeltat <= 0.0d0 .or. deltat <= 0.0d0 .or. &',1,'moment wall relaxation')
    mm=replace_exact(mm,'    epsp=1.0d-14\n\n    call rana2013_wall_tensors','    epsp=1.0d-14\n    bc_relax_primary=1.0d0-exp(-deltat/tau_bc)\n\n    call rana2013_wall_tensors',1,'primary wall relaxation')
    mm=replace_exact(mm,'use commvar,   only : const2,nstep,rkstep\n','use commvar,   only : const2,nstep,rkstep,deltat\n',1,'deltat import')
    old='min(1.0d0,dble(nstep)/1000.0d0)'; new='min(1.0d0,dble(nstep)*deltat/5.0d-2)'
    if old not in mm or old not in bc: raise RuntimeError('lid ramp fingerprints missing')
    return mm.replace(old,new),bc.replace(old,new)

def patch_maxwell(fl: str) -> str:
    old='miucal=temper*sqrt(temper)*tempconst1/(temper+tempconst)'
    return replace_exact(fl,old,'miucal=temper',1,'Maxwell viscosity')

def patch_optional_guards(mm: str, par: str) -> tuple[str,str,dict]:
    # Fortran does not guarantee short-circuit evaluation of .and.; therefore
    # an absent optional logical must never appear in the same expression as
    # PRESENT(). Patch every inline and block occurrence and fail unless no
    # unsafe expression remains.
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
        n_inline = len(list(inline.finditer(text)))
        text = inline.sub(
            lambda m: (
                f"{m.group('indent')}if (present(timerept)) then\n"
                f"{m.group('indent')}  if (timerept) time_beg=ptime()\n"
                f"{m.group('indent')}end if"
            ),
            text,
        )
        n_block = len(list(block.finditer(text)))
        text = block.sub(
            lambda m: (
                f"{m.group('indent')}if (present(timerept)) then\n"
                f"{m.group('indent')}  if (timerept) then\n"
                f"{m.group('body')}"
                f"{m.group('indent')}  end if\n"
                f"{m.group('indent')}end if"
            ),
            text,
        )
        remaining = len(unsafe.findall(text))
        if remaining:
            raise RuntimeError(f"{name}: {remaining} unsafe optional-logical expressions remain")
        return text, {"inline": n_inline, "block": n_block, "remaining": remaining}

    mm, mm_counts = patch_one("methodmoment", mm)
    par, par_counts = patch_one("parallel", par)
    if par_counts["inline"] < 1 or par_counts["block"] < 1:
        raise RuntimeError(f"parallel optional-guard fingerprints missing: {par_counts}")
    return mm, par, {"methodmoment": mm_counts, "parallel": par_counts}

EXTRAP_HELPERS='''
  ! Diagnostic paper-like one-sided extrapolation for unconstrained face data.
  subroutine r13_extrapolate_moment_face(nface,iw,jw,kw,ip,jp,kp)
    use commarray, only : sigmaxx,sigmaxy,sigmaxz,sigmayy,sigmayz,sigmazz,qx,qy,qz
    integer,intent(in)::nface,iw,jw,kw,ip,jp,kp
    integer::i2,j2,k2
    i2=2*ip-iw; j2=2*jp-jw; k2=2*kp-kw
    sigmaxx(iw,jw,kw)=2.d0*sigmaxx(ip,jp,kp)-sigmaxx(i2,j2,k2)
    sigmaxy(iw,jw,kw)=2.d0*sigmaxy(ip,jp,kp)-sigmaxy(i2,j2,k2)
    sigmaxz(iw,jw,kw)=2.d0*sigmaxz(ip,jp,kp)-sigmaxz(i2,j2,k2)
    sigmayy(iw,jw,kw)=2.d0*sigmayy(ip,jp,kp)-sigmayy(i2,j2,k2)
    sigmayz(iw,jw,kw)=2.d0*sigmayz(ip,jp,kp)-sigmayz(i2,j2,k2)
    sigmazz(iw,jw,kw)=2.d0*sigmazz(ip,jp,kp)-sigmazz(i2,j2,k2)
    qx(iw,jw,kw)=2.d0*qx(ip,jp,kp)-qx(i2,j2,k2)
    qy(iw,jw,kw)=2.d0*qy(ip,jp,kp)-qy(i2,j2,k2)
    qz(iw,jw,kw)=2.d0*qz(ip,jp,kp)-qz(i2,j2,k2)
  end subroutine r13_extrapolate_moment_face
  subroutine r13_extrapolate_primary_face(iw,jw,kw,ip,jp,kp)
    use commarray, only : rho,prs,tem,velx,vely,velz
    integer,intent(in)::iw,jw,kw,ip,jp,kp
    integer::i2,j2,k2
    i2=2*ip-iw; j2=2*jp-jw; k2=2*kp-kw
    rho(iw,jw,kw)=2.d0*rho(ip,jp,kp)-rho(i2,j2,k2)
    prs(iw,jw,kw)=2.d0*prs(ip,jp,kp)-prs(i2,j2,k2)
    tem(iw,jw,kw)=2.d0*tem(ip,jp,kp)-tem(i2,j2,k2)
    velx(iw,jw,kw)=2.d0*velx(ip,jp,kp)-velx(i2,j2,k2)
    vely(iw,jw,kw)=2.d0*vely(ip,jp,kp)-vely(i2,j2,k2)
    velz(iw,jw,kw)=2.d0*velz(ip,jp,kp)-velz(i2,j2,k2)
  end subroutine r13_extrapolate_primary_face
'''

def patch_extrap(mm: str) -> str:
    marker='contains\n'
    mm=replace_exact(mm,marker,marker+EXTRAP_HELPERS,1,'methodmoment contains')
    mm=mm.replace("if(moment=='r13') then\n                 call R13wbc(","if(moment=='r13') then\n                 call r13_extrapolate_moment_face(ndir,i,j,k,ip,jp,kp)\n                 call R13wbc(")
    mm=mm.replace("if(moment == 'r13') then\n                    call R13wbc_slip(","if(moment == 'r13') then\n                    call r13_extrapolate_primary_face(i,j,k,ip,jp,kp)\n                    call R13wbc_slip(")
    return mm

def wrap_r13_calls(text: str, call_name: str) -> str:
    pat=re.compile(rf"(?P<lead>^(?P<indent>[ \t]*)if\s*\(\s*moment\s*==\s*'r13'\s*\)\s*then\s*\n)(?P<body>(?:[ \t]*call\s+r13_extrapolate_[^\n]+\n)?[ \t]*call\s+{re.escape(call_name)}\([^\n]*&\s*\n[ \t]*uwall[^\n]*\)\s*\n)(?P<end>[ \t]*end\s*if\s*)$",re.I|re.M)
    ms=list(pat.finditer(text))
    if len(ms)!=4: raise RuntimeError(f'{call_name}: expected four wall blocks, found {len(ms)}')
    def repl(m):
      ind=m.group('indent'); body=''.join(ind+'  '+x[len(ind):] if x.startswith(ind) else ind+'  '+x for x in m.group('body').splitlines(True))
      return m.group('lead')+ind+"  if (.not. ((ndir==1 .or. ndir==2) .and. &\n"+ind+"       ((jrk==0 .and. j==0) .or. (jrk==jrkm .and. j==jm)))) then\n"+body+ind+'  end if\n'+m.group('end')
    return pat.sub(repl,text)

def patch_zero_top_corners(mm: str,bc: str) -> tuple[str,str]:
    old='uwall=uwall*min(1.0d0,dble(nstep)*deltat/5.0d-2)'
    new=old+"\n      if ((irk==0 .or. irk==irkm) .and. (jrk==jrkm .and. j==jm)) uwall=0.0d0"
    return mm.replace(old,new),bc.replace(old,new)

def patch_alternating_order(mm: str,bc: str) -> tuple[str,str]:
    old="if (moment=='r13') then"
    mm=mm.replace(old,"if (moment=='r13') then\n                 if (mod(rkstep,2)==0 .or. ndir==3 .or. ndir==4) then",4)
    mm=mm.replace("                 call R13wbc(ndir, n1, n2, n3, i, j, k, ip, jp, kp, &\n                         uwall, vwall, wwall, twall, alpha)\n              end if","                 call R13wbc(ndir, n1, n2, n3, i, j, k, ip, jp, kp, &\n                         uwall, vwall, wwall, twall, alpha)\n                 end if\n              end if",4)
    old2="if (moment == 'r13') then"
    bc=bc.replace(old2,"if (moment == 'r13') then\n                    if (mod(rkstep,2)==0 .or. ndir==3 .or. ndir==4) then",4)
    bc=bc.replace("                    call R13wbc_slip(ndir, n1, n2, n3, i, j, k, ip, jp, kp,   &\n                         uwall, vwall, wwall, twall, alpha)\n                   end if","                    call R13wbc_slip(ndir, n1, n2, n3, i, j, k, ip, jp, kp,   &\n                         uwall, vwall, wwall, twall, alpha)\n                    end if\n                   end if",4)
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
        gj=jg0+j; wy=1.0d0; if(gj==0 .or. gj==ja) wy=0.5d0
        do i=0,im
          gi=ig0+i; wx=1.0d0; if(gi==0 .or. gi==ia) wx=0.5d0
          w=wx*wy; lm=lm+w*rho(i,j,k); lw=lw+w
        enddo
      enddo
    enddo
    gm=psum(lm); gw=psum(lw); fac=gw/max(gm,1.0d-300)
    rho=rho*fac; prs=prs*fac; q(1,:,:,:,:)=rho; q(5,:,:,:,:)=prs
  end subroutine enforce_r13_mass_constraint
'''

def patch_mass(main: str) -> str:
    main=replace_exact(main,'contains\n','contains\n'+MASS_SUB,1,'mainloop contains')
    marker='    call boundvar()\n'
    main=replace_exact(main,marker,marker+'    call enforce_r13_mass_constraint()\n',3,'mass correction stages')
    return main

def patch_closure_refresh(main: str) -> str:
    marker='    call writeflowfield()\n'
    refresh="    if(moment=='r13') then\n      call mijkcal()\n      call Rijcal()\n      call Deltacal()\n    endif\n"
    return replace_exact(main,marker,refresh+marker,1,'closure refresh')

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
      'optional_guards_safe': not re.search(r'present\s*\(\s*timerept\s*\)\s*\.and\.\s*timerept', mm+par, re.IGNORECASE),
    }
    rep={'variant':a.variant,'config':cfg,'checks':checks,'optional_guard_counts':optional_guard_counts,'all_passed':all(checks.values()),'scientific_boundary':'diagnostic pseudo-time variants; none is the exact coupled Eq.20 matrix solve'}
    a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(rep,indent=2)+'\n'); print(json.dumps(rep,indent=2))
    if not rep['all_passed']: raise SystemExit(2)
if __name__=='__main__': main()
