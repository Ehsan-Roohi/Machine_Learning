#!/usr/bin/env python3
from __future__ import annotations
import argparse, difflib, json, re
from pathlib import Path

MIJK = r'''  subroutine mijkcal

    use constdef
    use commvar,   only : im,jm,km,const2
    use commarray, only : tmp,prs,sigma,qflux,miu,rho
    use comsolver, only : grad
    use parallel,  only : dataswap

    implicit none
    integer :: i,j,k,a,b,c,l
    real(8) :: s(3,3),qv(3),gs(3,3,3),glnp(3)
    real(8) :: raw(3,3,3),sym3(3,3,3),stf3(3,3,3),tr3(3)
    real(8) :: rawq(3,3,3),symq(3,3,3),stfq(3,3,3),trq(3)
    real(8) :: mfull(3,3,3),rrho,pv

    call dataswap(sigma)
    call dataswap(qflux)
    call dataswap(prs)
    dsigmaxx=grad(sigma(:,:,:,1))
    dsigmaxy=grad(sigma(:,:,:,2))
    dsigmaxz=grad(sigma(:,:,:,3))
    dsigmayy=grad(sigma(:,:,:,4))
    dsigmayz=grad(sigma(:,:,:,5))
    dsigmazz=-dsigmaxx-dsigmayy
    dprs=grad(prs)

    do k=0,km
    do j=0,jm
    do i=0,im
      pv=prs(i,j,k)
      rrho=1.0d0/rho(i,j,k)
      s=0.0d0
      s(1,1)=sigma(i,j,k,1); s(1,2)=sigma(i,j,k,2)
      s(2,1)=s(1,2);          s(1,3)=sigma(i,j,k,3)
      s(3,1)=s(1,3);          s(2,2)=sigma(i,j,k,4)
      s(2,3)=sigma(i,j,k,5); s(3,2)=s(2,3)
      s(3,3)=-s(1,1)-s(2,2)
      qv=(/qflux(i,j,k,1),qflux(i,j,k,2),qflux(i,j,k,3)/)
      glnp=(/dprs(i,j,k,1),dprs(i,j,k,2),dprs(i,j,k,3)/)/pv
      gs=0.0d0
      gs(1,1,:)=dsigmaxx(i,j,k,:); gs(1,2,:)=dsigmaxy(i,j,k,:)
      gs(2,1,:)=gs(1,2,:);          gs(1,3,:)=dsigmaxz(i,j,k,:)
      gs(3,1,:)=gs(1,3,:);          gs(2,2,:)=dsigmayy(i,j,k,:)
      gs(2,3,:)=dsigmayz(i,j,k,:); gs(3,2,:)=gs(2,3,:)
      gs(3,3,:)=dsigmazz(i,j,k,:)

      do a=1,3; do b=1,3; do c=1,3
        raw(a,b,c)=gs(a,b,c)-s(a,b)*glnp(c)
        rawq(a,b,c)=qv(a)*s(b,c)
      enddo; enddo; enddo
      do a=1,3; do b=1,3; do c=1,3
        sym3(a,b,c)=(raw(a,b,c)+raw(a,c,b)+raw(b,a,c)+raw(b,c,a)+raw(c,a,b)+raw(c,b,a))/6.0d0
        symq(a,b,c)=(rawq(a,b,c)+rawq(a,c,b)+rawq(b,a,c)+rawq(b,c,a)+rawq(c,a,b)+rawq(c,b,a))/6.0d0
      enddo; enddo; enddo
      tr3=0.0d0; trq=0.0d0
      do c=1,3; do l=1,3
        tr3(c)=tr3(c)+sym3(l,l,c)
        trq(c)=trq(c)+symq(l,l,c)
      enddo; enddo
      do a=1,3; do b=1,3; do c=1,3
        stf3(a,b,c)=sym3(a,b,c)
        stfq(a,b,c)=symq(a,b,c)
        if(a==b) then
          stf3(a,b,c)=stf3(a,b,c)-tr3(c)/5.0d0
          stfq(a,b,c)=stfq(a,b,c)-trq(c)/5.0d0
        endif
        if(a==c) then
          stf3(a,b,c)=stf3(a,b,c)-tr3(b)/5.0d0
          stfq(a,b,c)=stfq(a,b,c)-trq(b)/5.0d0
        endif
        if(b==c) then
          stf3(a,b,c)=stf3(a,b,c)-tr3(a)/5.0d0
          stfq(a,b,c)=stfq(a,b,c)-trq(a)/5.0d0
        endif
      enddo; enddo; enddo

      ! Rana, Torrilhon & Struchtrup (2013), Eq. (13):
      ! m_ijk = 4/3 q_<i sigma_jk>/p - 2 mu theta d_<i(sigma_jk>/p)
      mfull=(4.0d0/(3.0d0*pv))*stfq-2.0d0*miu(i,j,k)*rrho*stf3
      mijk(i,j,k,1)=mfull(1,1,1)
      mijk(i,j,k,2)=mfull(1,1,2)
      mijk(i,j,k,3)=mfull(1,1,3)
      mijk(i,j,k,4)=mfull(1,2,2)
      mijk(i,j,k,5)=mfull(2,2,2)
      mijk(i,j,k,6)=mfull(2,2,3)
      mijk(i,j,k,7)=mfull(1,2,3)
    enddo
    enddo
    enddo
    call dataswap(mijk,timerept=ltimrpt)
  end subroutine mijkcal'''

RIJ = r'''  subroutine rijcal

    use commvar,   only : im,jm,km
    use commarray, only : prs,miu,rho,sigma,qflux
    use comsolver, only : grad
    use parallel,  only : dataswap

    implicit none
    integer :: i,j,k,a,b,l
    real(8) :: s(3,3),qv(3),gq(3,3),glnp(3)
    real(8) :: ss(3,3),qq(3,3),gd(3,3),rfull(3,3)
    real(8) :: trss,trqq,trgd,rrho,pv

    call dqfluxcal
    call dataswap(sigma)
    call dataswap(prs)
    dprs=grad(prs)
    do k=0,km
    do j=0,jm
    do i=0,im
      pv=prs(i,j,k)
      rrho=1.0d0/rho(i,j,k)
      s=0.0d0
      s(1,1)=sigma(i,j,k,1); s(1,2)=sigma(i,j,k,2)
      s(2,1)=s(1,2);          s(1,3)=sigma(i,j,k,3)
      s(3,1)=s(1,3);          s(2,2)=sigma(i,j,k,4)
      s(2,3)=sigma(i,j,k,5); s(3,2)=s(2,3)
      s(3,3)=-s(1,1)-s(2,2)
      qv=(/qflux(i,j,k,1),qflux(i,j,k,2),qflux(i,j,k,3)/)
      gq(1,:)=dqx(i,j,k,:); gq(2,:)=dqy(i,j,k,:); gq(3,:)=dqz(i,j,k,:)
      glnp=(/dprs(i,j,k,1),dprs(i,j,k,2),dprs(i,j,k,3)/)/pv
      ss=0.0d0
      do a=1,3; do b=1,3; do l=1,3
        ss(a,b)=ss(a,b)+s(l,a)*s(b,l)
      enddo; enddo; enddo
      do a=1,3; do b=1,3
        qq(a,b)=qv(a)*qv(b)
        gd(a,b)=0.5d0*((gq(a,b)-qv(a)*glnp(b))+(gq(b,a)-qv(b)*glnp(a)))
      enddo; enddo
      trss=(ss(1,1)+ss(2,2)+ss(3,3))/3.0d0
      trqq=(qq(1,1)+qq(2,2)+qq(3,3))/3.0d0
      trgd=(gd(1,1)+gd(2,2)+gd(3,3))/3.0d0
      do a=1,3
        ss(a,a)=ss(a,a)-trss
        qq(a,a)=qq(a,a)-trqq
        gd(a,a)=gd(a,a)-trgd
      enddo

      ! Rana et al. (2013), Eq. (13):
      ! R_ij = 20/7 sigma_k<i sigma_j>k/rho + 64/25 q_<i q_j>/p
      !        - 24/5 mu theta d_<i(q_j>/p)
      rfull=(20.0d0/7.0d0)*rrho*ss+(64.0d0/(25.0d0*pv))*qq &
            -(24.0d0/5.0d0)*miu(i,j,k)*rrho*gd
      rij(i,j,k,1)=rfull(1,1)
      rij(i,j,k,2)=rfull(1,2)
      rij(i,j,k,3)=rfull(1,3)
      rij(i,j,k,4)=rfull(2,2)
      rij(i,j,k,5)=rfull(2,3)
    enddo
    enddo
    enddo
    call dataswap(rij,timerept=ltimrpt)
  end subroutine rijcal'''

DELTA = r'''  subroutine deltacal

    use commvar,   only : im,jm,km
    use commarray, only : miu,prs,rho,sigma,qflux
    use comsolver, only : grad
    use parallel,  only : dataswap

    implicit none
    integer :: i,j,k
    real(8) :: sxx,sxy,sxz,syy,syz,szz,qx,qy,qz
    real(8) :: sig2,q2,divq,qg,pv,rrho,gpx,gpy,gpz

    call dataswap(prs)
    dprs=grad(prs)
    do k=0,km
    do j=0,jm
    do i=0,im
      pv=prs(i,j,k); rrho=1.0d0/rho(i,j,k)
      sxx=sigma(i,j,k,1); sxy=sigma(i,j,k,2)
      sxz=sigma(i,j,k,3); syy=sigma(i,j,k,4)
      syz=sigma(i,j,k,5); szz=-sxx-syy
      qx=qflux(i,j,k,1); qy=qflux(i,j,k,2); qz=qflux(i,j,k,3)
      sig2=sxx*sxx+syy*syy+szz*szz+2.0d0*(sxy*sxy+sxz*sxz+syz*syz)
      q2=qx*qx+qy*qy+qz*qz
      divq=dqx(i,j,k,1)+dqy(i,j,k,2)+dqz(i,j,k,3)
      gpx=dprs(i,j,k,1)/pv; gpy=dprs(i,j,k,2)/pv; gpz=dprs(i,j,k,3)/pv
      qg=qx*gpx+qy*gpy+qz*gpz
      ! Rana et al. (2013), Eq. (13):
      ! Delta = 5 sigma:sigma/rho + 56/5 q.q/p - 12 mu theta div(q/p)
      delta(i,j,k)=5.0d0*rrho*sig2+(56.0d0/(5.0d0*pv))*q2 &
                   -12.0d0*miu(i,j,k)*rrho*(divq-qg)
    enddo
    enddo
    enddo
  end subroutine deltacal'''

R13WBC = "  subroutine R13wbc(nface, n1, n2, n3, iw, jw, kw, ip, jp, kp, &\n                    uwall, vwall, wwall, twall, alpha)\n\n    use commvar,   only : const2,nstep,rkstep\n    use commarray, only : vel,sigma,qflux,tmp,prs\n    use parallel,  only : mpirank,irk,jrk,krk,ig0,jg0,kg0\n    use, intrinsic :: ieee_arithmetic, only : ieee_is_finite\n    use, intrinsic :: iso_fortran_env, only : error_unit\n\n    implicit none\n    integer, intent(in) :: iw,jw,kw,ip,jp,kp,nface\n    real(8), intent(in) :: n1,n2,n3,uwall,vwall,wwall,twall,alpha\n    integer :: a,b\n    real(8) :: nv(3),tv(3),sv(3),wv(3)\n    real(8) :: theta,thetaw,Cinv,P,Vt,Tjump\n    real(8) :: snn,stt,stn,qn,qt,Rnn,Rtt,Rtn,mtnn,mnnn,mttn,Deltav\n    real(8) :: snn_new,stt_new,stn_new,qn_new,qt_new\n    real(8) :: Snew(3,3),qnew(3),epsp\n    real(8), parameter :: bc_relax=0.05d0\n\n    if (abs(n3) > 1.0d-12) stop 'Rana2013 R13 wall map is the exact 2D paper model'\n    nv=(/n1,n2,0.0d0/)\n    if (abs(n1) > 0.5d0) then\n      tv=(/0.0d0,1.0d0,0.0d0/)\n    else\n      tv=(/1.0d0,0.0d0,0.0d0/)\n    endif\n    sv=(/0.0d0,0.0d0,1.0d0/)\n    wv=(/uwall,vwall,wwall/)\n\n    theta=max(tmp(iw,jw,kw)/const2,1.0d-14)\n    thetaw=twall/const2\n    Cinv=(2.0d0-alpha)/alpha*sqrt(0.5d0*pi*theta)\n    epsp=1.0d-14\n\n    call rana2013_wall_tensors(iw,jw,kw,nv,tv,sv,snn,stt,stn, &\n         Rnn,Rtt,Rtn,mtnn,mnnn,mttn,qn,qt)\n    Deltav=delta(iw,jw,kw)\n    P=prs(iw,jw,kw)+0.5d0*stt-Deltav/(120.0d0*theta)-Rtt/(28.0d0*theta)\n    if (.not. ieee_is_finite(P) .or. P <= epsp) then\n      write(error_unit,*) 'RANA_P_FAILURE map=moment'\n      write(error_unit,*) 'rank=',mpirank,' decomp=',irk,jrk,krk, &\n           ' step=',nstep,' rk=',rkstep,' face=',nface\n      write(error_unit,*) 'local_ijk=',iw,jw,kw,' global_ijk=', &\n           ig0+iw,jg0+jw,kg0+kw\n      write(error_unit,*) 'P=',P,' prs=',prs(iw,jw,kw), &\n           ' theta=',theta,' stt=',stt\n      write(error_unit,*) 'Delta=',Deltav,' Rtt=',Rtt\n      write(error_unit,*) 'P_terms=',prs(iw,jw,kw),0.5d0*stt, &\n           -Deltav/(120.0d0*theta),-Rtt/(28.0d0*theta)\n      flush(error_unit)\n      stop 'Rana2013 moment wall effective P invalid'\n    endif\n\n    Vt=(vel(iw,jw,kw,1)-uwall)*tv(1)+(vel(iw,jw,kw,2)-vwall)*tv(2)\n    Tjump=theta-thetaw\n\n    ! Exact Rana et al. (2013), Eqs. (7b)-(7f), rearranged for\n    ! sigma_tn, q_n, q_t, sigma_nn, sigma_tt. No extra wall model.\n    stn_new=-(P*Vt+qt/5.0d0+mtnn/2.0d0)/Cinv\n    qn_new=-(2.0d0*P*Tjump-0.5d0*P*Vt*Vt+0.5d0*theta*snn &\n              +Deltav/15.0d0+5.0d0*Rnn/28.0d0)/Cinv\n    qt_new=(5.0d0/(11.0d0*theta))*(6.0d0*P*Tjump*Vt+P*theta*Vt &\n             -P*Vt**3-0.5d0*theta*mtnn-Cinv*Rtn)\n    snn_new=(5.0d0/(7.0d0*theta))*(0.4d0*P*Tjump-0.6d0*P*Vt*Vt &\n             +Deltav/75.0d0-Rnn/14.0d0-Cinv*mnnn)\n    stt_new=(-Cinv*mttn-0.2d0*P*Tjump+0.8d0*P*Vt*Vt-Rtt/14.0d0 &\n             +theta*snn_new/5.0d0-Deltav/150.0d0)/theta\n\n    Snew=0.0d0\n    do a=1,3; do b=1,3\n      Snew(a,b)=snn_new*nv(a)*nv(b)+stt_new*tv(a)*tv(b) &\n                 +stn_new*(tv(a)*nv(b)+nv(a)*tv(b))\n    enddo; enddo\n    Snew(3,3)=-snn_new-stt_new\n    qnew=qn_new*nv+qt_new*tv\n\n    ! Damped fixed-point iteration of the exact wall equations. The\n    ! fixed point is unchanged, while all iterated fields are checkpointed.\n    sigma(iw,jw,kw,1)=sigma(iw,jw,kw,1)+bc_relax*(Snew(1,1)-sigma(iw,jw,kw,1))\n    sigma(iw,jw,kw,2)=sigma(iw,jw,kw,2)+bc_relax*(Snew(1,2)-sigma(iw,jw,kw,2))\n    sigma(iw,jw,kw,3)=sigma(iw,jw,kw,3)+bc_relax*(0.0d0-sigma(iw,jw,kw,3))\n    sigma(iw,jw,kw,4)=sigma(iw,jw,kw,4)+bc_relax*(Snew(2,2)-sigma(iw,jw,kw,4))\n    sigma(iw,jw,kw,5)=sigma(iw,jw,kw,5)+bc_relax*(0.0d0-sigma(iw,jw,kw,5))\n    qflux(iw,jw,kw,1)=qflux(iw,jw,kw,1)+bc_relax*(qnew(1)-qflux(iw,jw,kw,1))\n    qflux(iw,jw,kw,2)=qflux(iw,jw,kw,2)+bc_relax*(qnew(2)-qflux(iw,jw,kw,2))\n    qflux(iw,jw,kw,3)=qflux(iw,jw,kw,3)+bc_relax*(0.0d0-qflux(iw,jw,kw,3))\n    qrhs_mom(iw,jw,kw,1:8)=0.0d0\n    call fvar2qmom(q=q_mom(iw,jw,kw,:),stress=sigma(iw,jw,kw,:), &\n                   heatflux=qflux(iw,jw,kw,:))\n  end subroutine R13wbc\n"

R13WBC_SLIP = "  subroutine R13wbc_slip(nface,n1,n2,n3,iw,jw,kw,ip,jp,kp, &\n                    uwall,vwall,wwall,twall,alpha)\n\n    use commvar,   only : const2,nstep,rkstep\n    use commarray, only : q,rho,vel,sigma,qflux,tmp,prs,qrhs\n    use fludyna,   only : thermal,fvar2q\n    use parallel,  only : mpirank,irk,jrk,krk,ig0,jg0,kg0\n    use, intrinsic :: ieee_arithmetic, only : ieee_is_finite\n    use, intrinsic :: iso_fortran_env, only : error_unit\n\n    implicit none\n    integer, intent(in) :: iw,jw,kw,ip,jp,kp,nface\n    real(8), intent(in) :: n1,n2,n3,uwall,vwall,wwall,twall,alpha\n    real(8) :: nv(3),tv(3),sv(3),wv(3),vnew(3)\n    real(8) :: theta,thetaw,Cinv,P,Vt,Tjump,Deltav,epsp,target_tmp\n    real(8), parameter :: bc_relax=0.05d0\n    real(8) :: snn,stt,stn,qn,qt,Rnn,Rtt,Rtn,mtnn,mnnn,mttn\n\n    if (abs(n3) > 1.0d-12) stop 'Rana2013 R13 slip map is the exact 2D paper model'\n    nv=(/n1,n2,0.0d0/)\n    if (abs(n1) > 0.5d0) then\n      tv=(/0.0d0,1.0d0,0.0d0/)\n    else\n      tv=(/1.0d0,0.0d0,0.0d0/)\n    endif\n    sv=(/0.0d0,0.0d0,1.0d0/)\n    wv=(/uwall,vwall,wwall/)\n    theta=max(tmp(iw,jw,kw)/const2,1.0d-14)\n    thetaw=twall/const2\n    Cinv=(2.0d0-alpha)/alpha*sqrt(0.5d0*pi*theta)\n    epsp=1.0d-14\n\n    call rana2013_wall_tensors(iw,jw,kw,nv,tv,sv,snn,stt,stn, &\n         Rnn,Rtt,Rtn,mtnn,mnnn,mttn,qn,qt)\n    Deltav=delta(iw,jw,kw)\n    P=prs(iw,jw,kw)+0.5d0*stt-Deltav/(120.0d0*theta)-Rtt/(28.0d0*theta)\n    if (.not. ieee_is_finite(P) .or. P <= epsp) then\n      write(error_unit,*) 'RANA_P_FAILURE map=primary_slip'\n      write(error_unit,*) 'rank=',mpirank,' decomp=',irk,jrk,krk, &\n           ' step=',nstep,' rk=',rkstep,' face=',nface\n      write(error_unit,*) 'local_ijk=',iw,jw,kw,' global_ijk=', &\n           ig0+iw,jg0+jw,kg0+kw\n      write(error_unit,*) 'P=',P,' prs=',prs(iw,jw,kw), &\n           ' theta=',theta,' stt=',stt\n      write(error_unit,*) 'Delta=',Deltav,' Rtt=',Rtt\n      write(error_unit,*) 'P_terms=',prs(iw,jw,kw),0.5d0*stt, &\n           -Deltav/(120.0d0*theta),-Rtt/(28.0d0*theta)\n      flush(error_unit)\n      stop 'Rana2013 primary-slip wall effective P invalid'\n    endif\n\n    ! Exact Eqs. (7a)-(7c), solved for v_n, V_t and theta-theta_w.\n    Vt=(-Cinv*stn-qt/5.0d0-mtnn/2.0d0)/P\n    Tjump=(-Cinv*qn+0.5d0*P*Vt*Vt-0.5d0*theta*snn &\n           -Deltav/15.0d0-5.0d0*Rnn/28.0d0)/(2.0d0*P)\n    vnew=wv+Vt*tv\n    target_tmp=max(twall+const2*Tjump,1.0d-12)\n    ! Damped, checkpoint-invariant fixed-point iteration.\n    vel(iw,jw,kw,1)=vel(iw,jw,kw,1)+bc_relax*(vnew(1)-vel(iw,jw,kw,1))\n    vel(iw,jw,kw,2)=vel(iw,jw,kw,2)+bc_relax*(vnew(2)-vel(iw,jw,kw,2))\n    vel(iw,jw,kw,3)=vel(iw,jw,kw,3)+bc_relax*(0.0d0-vel(iw,jw,kw,3))\n    tmp(iw,jw,kw)=tmp(iw,jw,kw)+bc_relax*(target_tmp-tmp(iw,jw,kw))\n    rho(iw,jw,kw)=thermal(pressure=prs(iw,jw,kw),temperature=tmp(iw,jw,kw))\n    call fvar2q(q=q(iw,jw,kw,:),density=rho(iw,jw,kw), &\n                velocity=vel(iw,jw,kw,:),temperature=tmp(iw,jw,kw))\n    qrhs(iw,jw,kw,1:5)=0.0d0\n  end subroutine R13wbc_slip\n"

RANA_HELPER = '  subroutine rana2013_wall_tensors(iw,jw,kw,nv,tv,sv,snn,stt,stn, &\n               Rnn,Rtt,Rtn,mtnn,mnnn,mttn,qn,qt)\n    use commarray, only : sigma,qflux\n    implicit none\n    integer,intent(in) :: iw,jw,kw\n    real(8),intent(in) :: nv(3),tv(3),sv(3)\n    real(8),intent(out):: snn,stt,stn,Rnn,Rtt,Rtn,mtnn,mnnn,mttn,qn,qt\n    integer :: a,b,c\n    real(8) :: S(3,3),RR(3,3),M(3,3,3),qv(3)\n    S=0.0d0; RR=0.0d0; M=0.0d0\n    S(1,1)=sigma(iw,jw,kw,1); S(1,2)=sigma(iw,jw,kw,2); S(2,1)=S(1,2)\n    S(1,3)=sigma(iw,jw,kw,3); S(3,1)=S(1,3)\n    S(2,2)=sigma(iw,jw,kw,4); S(2,3)=sigma(iw,jw,kw,5); S(3,2)=S(2,3)\n    S(3,3)=-S(1,1)-S(2,2)\n    RR(1,1)=rij(iw,jw,kw,1); RR(1,2)=rij(iw,jw,kw,2); RR(2,1)=RR(1,2)\n    RR(1,3)=rij(iw,jw,kw,3); RR(3,1)=RR(1,3)\n    RR(2,2)=rij(iw,jw,kw,4); RR(2,3)=rij(iw,jw,kw,5); RR(3,2)=RR(2,3)\n    RR(3,3)=-RR(1,1)-RR(2,2)\n    M(1,1,1)=mijk(iw,jw,kw,1)\n    M(1,1,2)=mijk(iw,jw,kw,2); M(1,2,1)=M(1,1,2); M(2,1,1)=M(1,1,2)\n    M(1,1,3)=mijk(iw,jw,kw,3); M(1,3,1)=M(1,1,3); M(3,1,1)=M(1,1,3)\n    M(1,2,2)=mijk(iw,jw,kw,4); M(2,1,2)=M(1,2,2); M(2,2,1)=M(1,2,2)\n    M(2,2,2)=mijk(iw,jw,kw,5)\n    M(2,2,3)=mijk(iw,jw,kw,6); M(2,3,2)=M(2,2,3); M(3,2,2)=M(2,2,3)\n    M(1,2,3)=mijk(iw,jw,kw,7); M(1,3,2)=M(1,2,3); M(2,1,3)=M(1,2,3)\n    M(2,3,1)=M(1,2,3); M(3,1,2)=M(1,2,3); M(3,2,1)=M(1,2,3)\n    M(1,3,3)=-M(1,1,1)-M(1,2,2); M(3,1,3)=M(1,3,3); M(3,3,1)=M(1,3,3)\n    M(2,3,3)=-M(1,1,2)-M(2,2,2); M(3,2,3)=M(2,3,3); M(3,3,2)=M(2,3,3)\n    M(3,3,3)=-M(1,1,3)-M(2,2,3)\n    qv=(/qflux(iw,jw,kw,1),qflux(iw,jw,kw,2),qflux(iw,jw,kw,3)/)\n    snn=0.0d0;stt=0.0d0;stn=0.0d0;Rnn=0.0d0;Rtt=0.0d0;Rtn=0.0d0\n    mtnn=0.0d0;mnnn=0.0d0;mttn=0.0d0\n    do a=1,3; do b=1,3\n      snn=snn+nv(a)*S(a,b)*nv(b); stt=stt+tv(a)*S(a,b)*tv(b)\n      stn=stn+tv(a)*S(a,b)*nv(b)\n      Rnn=Rnn+nv(a)*RR(a,b)*nv(b); Rtt=Rtt+tv(a)*RR(a,b)*tv(b)\n      Rtn=Rtn+tv(a)*RR(a,b)*nv(b)\n      do c=1,3\n        mtnn=mtnn+tv(a)*nv(b)*nv(c)*M(a,b,c)\n        mnnn=mnnn+nv(a)*nv(b)*nv(c)*M(a,b,c)\n        mttn=mttn+tv(a)*tv(b)*nv(c)*M(a,b,c)\n      enddo\n    enddo; enddo\n    qn=dot_product(qv,nv); qt=dot_product(qv,tv)\n  end subroutine rana2013_wall_tensors\n'

def replace_sub(text: str, name: str, new: str) -> str:
    pat=re.compile(rf"(?ims)^\s*subroutine\s+{name}\b.*?^\s*end\s+subroutine\s+{name}\b")
    out,n=pat.subn(new,text,count=1)
    if n!=1: raise RuntimeError(f"Expected one {name}, found {n}")
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('astr',type=Path)
    ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--patch-output',type=Path,required=True)
    a=ap.parse_args()
    mm=a.astr/'src/methodmoment.F90'; bc=a.astr/'src/bc.F90'
    mm0=mm.read_text(); bc0=bc.read_text()
    text=replace_sub(mm0,'mijkcal',MIJK)
    text=replace_sub(text,'rijcal',RIJ)
    text=replace_sub(text,'deltacal',DELTA)
    text=replace_sub(text,'R13wbc',R13WBC)
    text=replace_sub(text,'R13wbc_slip',R13WBC_SLIP)
    pos=text.lower().find('  subroutine r13wbc(')
    if pos<0: raise RuntimeError('R13wbc insertion point not found')
    text=text[:pos]+RANA_HELPER+'\n'+text[pos:]
    n1=0
    text,n2=re.subn(r'uwall\s*=\s*dble\(nstep\)\*deltat\s*\n\s*if \(uwall > 1\.0d0\) uwall = 1\.0d0',
                    'uwall = min(1.0d0,dble(nstep)/1000.0d0)',text)
    bc1,n3=re.subn(r'uwall\s*=\s*dble\(nstep\)\*deltat(?:\*10\.0d0)?\s*\n\s*if \(uwall > 1\.0d0\) uwall = 1\.0d0',
                   'uwall = min(1.0d0,dble(nstep)/1000.0d0)',bc0)
    if n2<1 or n3<1: raise RuntimeError(f'Lid-homotopy patches missing: method={n2}, bc={n3}')
    mm.write_text(text); bc.write_text(bc1)
    patch=''.join(difflib.unified_diff(mm0.splitlines(True),text.splitlines(True),
        fromfile='astr/src/methodmoment.F90.upstream',tofile='astr/src/methodmoment.F90.rana2013'))
    patch+=''.join(difflib.unified_diff(bc0.splitlines(True),bc1.splitlines(True),
        fromfile='astr/src/bc.F90.upstream',tofile='astr/src/bc.F90.rana2013'))
    a.patch_output.parent.mkdir(parents=True,exist_ok=True); a.patch_output.write_text(patch)
    checks={
      'eq13_m_qsigma_4_over_3':'4.0d0/(3.0d0*pv)' in text,
      'eq13_m_quotient_gradient':'gs(a,b,c)-s(a,b)*glnp(c)' in text,
      'eq13_R_sigmasigma_20_over_7':'20.0d0/7.0d0' in text,
      'eq13_R_qq_64_over_25':'64.0d0/(25.0d0*pv)' in text,
      'eq13_R_quotient_gradient':'gq(a,b)-qv(a)*glnp(b)' in text,
      'eq13_Delta_sigmasigma_5':'5.0d0*rrho*sig2' in text,
      'eq13_Delta_qq_56_over_5':'56.0d0/(5.0d0*pv)' in text,
      'eq13_Delta_div_q_over_p':'divq-qg' in text,
      'correct_divq_z':'dqz(i,j,k,3)' in text,
      'eq7_exact_effective_P_tau':'0.5d0*stt-Deltav/(120.0d0*theta)-Rtt/(28.0d0*theta)' in text,
      'effective_P_guard_finite':text.count('ieee_is_finite(P)')==2,
      'effective_P_guard_diagnostics':'RANA_P_FAILURE map=moment' in text and 'RANA_P_FAILURE map=primary_slip' in text,
      'eq7b_exact':'stn_new=-(P*Vt+qt/5.0d0+mtnn/2.0d0)/Cinv' in text,
      'eq7c_exact':'qn_new=-(2.0d0*P*Tjump-0.5d0*P*Vt*Vt+0.5d0*theta*snn' in text,
      'eq7d_exact':'qt_new=(5.0d0/(11.0d0*theta))' in text,
      'eq7e_exact':'snn_new=(5.0d0/(7.0d0*theta))' in text,
      'eq7f_exact':'stt_new=(-Cinv*mttn-0.2d0*P*Tjump' in text,
      'eq7a_zero_normal_velocity':'vnew=wv+Vt*tv' in text,
      'lid_homotopy_method':n2>=1 and 'min(1.0d0,dble(nstep)/1000.0d0)' in text,
      'lid_homotopy_bc':n3>=1 and 'min(1.0d0,dble(nstep)/1000.0d0)' in bc1,
      'exact_wall_fixed_point_damped':'bc_relax=0.05d0' in text and 'Damped fixed-point iteration' in text,
    }
    report={'reference':'Rana, Torrilhon & Struchtrup, JCP 236 (2013), Eqs. (3),(4),(7),(13)',
            'model':'full nonlinear transformed Maxwell R13 Eq. (13), damped checkpoint-invariant fixed-point iteration of exact Eq. (7)',
            'checks':checks,'methodmoment_lid_replacements':n2,'bc_lid_replacements':n3,
            'all_passed':all(checks.values())}
    a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if not report['all_passed']: raise SystemExit(2)
if __name__=='__main__': main()
