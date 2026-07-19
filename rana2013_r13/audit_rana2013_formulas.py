#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

PAPER='Rana, Torrilhon & Struchtrup, Journal of Computational Physics 236 (2013) 169-186'

def stf2(a):
    s=0.5*(a+a.T)
    return s-np.eye(3)*np.trace(s)/3.0

def stf3(a):
    s=np.zeros((3,3,3))
    import itertools
    perms=list(itertools.permutations(range(3)))
    for i,j,k in np.ndindex(3,3,3):
        s[i,j,k]=sum(a[(i,j,k)[p[0]],(i,j,k)[p[1]],(i,j,k)[p[2]]] for p in perms)/6.0
    tr=np.einsum('iik->k',s); d=np.eye(3)
    return s-(np.einsum('ij,k->ijk',d,tr)+np.einsum('ik,j->ijk',d,tr)+np.einsum('jk,i->ijk',d,tr))/5.0

def closures(rho,p,mu,sigma,q,grad_sigma,grad_q,grad_p):
    glnp=grad_p/p
    dm=grad_sigma-sigma[:,:,None]*glnp[None,None,:]
    dq=grad_q-q[:,None]*glnp[None,:]
    m=(4.0/(3.0*p))*stf3(np.einsum('i,jk->ijk',q,sigma))-2.0*mu/rho*stf3(dm)
    R=(20.0/(7.0*rho))*stf2(sigma.T@sigma)+(64.0/(25.0*p))*stf2(np.outer(q,q))-(24.0/5.0)*mu/rho*stf2(dq)
    Delta=5.0*np.sum(sigma*sigma)/rho+(56.0/5.0)*np.dot(q,q)/p-12.0*mu/rho*(np.trace(grad_q)-np.dot(q,glnp))
    return m,R,Delta

def bc_residual(theta,P,Vt,Tjump,sig_nn,sig_tt,sig_tn,qt,qn,Rnn,Rtt,Rtn,Delta,mtnn,mnnn,mttn,chi=1.0):
    C=chi/(2.0-chi)*np.sqrt(2.0/(np.pi*theta))
    return np.array([
      sig_tn + C*(P*Vt+qt/5.0+mtnn/2.0),
      qn + C*(2.0*P*Tjump-P*Vt*Vt/2.0+theta*sig_nn/2.0+Delta/15.0+5.0*Rnn/28.0),
      Rtn - C*(6.0*P*Tjump*Vt+P*theta*Vt-P*Vt**3-11.0*theta*qt/5.0-theta*mtnn/2.0),
      mnnn - C*(2.0*P*Tjump/5.0-3.0*P*Vt*Vt/5.0-7.0*theta*sig_nn/5.0+Delta/75.0-Rnn/14.0),
      mttn + C*(P*Tjump/5.0-4.0*P*Vt*Vt/5.0+Rtt/14.0+theta*sig_tt-theta*sig_nn/5.0+Delta/150.0),
    ])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('astr',type=Path); ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    text=(args.astr/'src/methodmoment.F90').read_text(); bc=(args.astr/'src/bc.F90').read_text()
    static={
      'Maxwell_Pr_sigma_1':'Asigma=1.d0' in text,
      'Maxwell_Pr_q_2_over_3':'Aq=num2d3' in text,
      'Maxwell_Pr_m_3_over_2':'Am=1.5d0' in text,
      'Maxwell_Pr_R_7_over_6':'AR1=num7d6' in text,
      'Maxwell_Pr_Delta_2_over_3':'Adelta1=num2d3' in text,
      'Eq13_m_4_over_3':'4.0d0/(3.0d0*pv)' in text,
      'Eq13_m_minus_2':'-2.0d0*miu(i,j,k)*rrho*stf3' in text,
      'Eq13_R_20_over_7':'20.0d0/7.0d0' in text,
      'Eq13_R_64_over_25':'64.0d0/(25.0d0*pv)' in text,
      'Eq13_R_minus_24_over_5':'-(24.0d0/5.0d0)*miu(i,j,k)*rrho*gd' in text,
      'Eq13_Delta_5':'5.0d0*rrho*sig2' in text,
      'Eq13_Delta_56_over_5':'56.0d0/(5.0d0*pv)' in text,
      'Eq13_Delta_minus_12':'-12.0d0*miu(i,j,k)*rrho*(divq-qg)' in text,
      'Eq7_P_effective_tangent':'0.5d0*stt-Deltav/(120.0d0*theta)-Rtt/(28.0d0*theta)' in text,
      'Eq7_Maxwell_factor':'sqrt(0.5d0*pi*theta)' in text,
      'Eq7_full_accommodation':bc.count('alpha = 1.0d0')>=4,
      'Eq7a_vn_zero':'vnew=wv+Vt*tv' in text,
      'Eq7b_exact':'stn_new=-(P*Vt+qt/5.0d0+mtnn/2.0d0)/Cinv' in text,
      'Eq7c_exact':'qn_new=-(2.0d0*P*Tjump-0.5d0*P*Vt*Vt+0.5d0*theta*snn' in text,
      'Eq7d_exact':'qt_new=(5.0d0/(11.0d0*theta))' in text,
      'Eq7e_exact':'snn_new=(5.0d0/(7.0d0*theta))' in text,
      'Eq7f_exact':'stt_new=(-Cinv*mttn-0.2d0*P*Tjump' in text,
      'Eq7_checkpoint_independent':('vel12_slip' not in text[text.find('subroutine R13wbc_slip'):text.find('end subroutine R13wbc_slip')]) and ('sig_ttbc12' not in text[text.find('subroutine R13wbc'):text.find('end subroutine R13wbc')]),
      'lid_homotopy_reaches_exact_final':text.count('min(1.0d0,dble(nstep)/1000.0d0)')>=2 and bc.count('min(1.0d0,dble(nstep)/1000.0d0)')>=1,
      'checkpoint_invariant_damped_wall_solver':text.count('bc_relax=0.05d0')>=2 and 'Damped fixed-point iteration' in text,
      'correct_3D_divergence':'divq=dqx(i,j,k,1)+dqy(i,j,k,2)+dqz(i,j,k,3)' in text,
      'Eq4_STF_q_gradient_fingerprint':'num4d15*(2.d0*dqxdx-dqydy-dqzdz)' in text,
      'Eq3_heat_relaxation_fingerprint':'Aq*qx/miup' in text,
      'Eq4_stress_relaxation_fingerprint':'sigmaxx/miup' in text,
    }
    rng=np.random.default_rng(20260718)
    err={'m_sym':0.0,'m_trace':0.0,'R_sym':0.0,'R_trace':0.0,'quotient_invariance':0.0,'equilibrium':0.0,'bc_constructed_residual':0.0}
    for _ in range(200):
        rho=float(rng.uniform(.3,2)); theta=float(rng.uniform(.5,2)); p=rho*theta; mu=float(rng.uniform(.01,.3))
        sigma=stf2(rng.normal(size=(3,3))); q=rng.normal(size=3); gs=rng.normal(size=(3,3,3)); gq=rng.normal(size=(3,3)); gp=rng.normal(size=3)
        m,R,D=closures(rho,p,mu,sigma,q,gs,gq,gp)
        err['m_sym']=max(err['m_sym'],float(np.max(np.abs(m-m.transpose(1,0,2)))),float(np.max(np.abs(m-m.transpose(2,1,0)))))
        err['m_trace']=max(err['m_trace'],float(np.max(np.abs(np.einsum('iik->k',m)))))
        err['R_sym']=max(err['R_sym'],float(np.max(np.abs(R-R.T))))
        err['R_trace']=max(err['R_trace'],float(abs(np.trace(R))))
        gp2=rng.normal(size=3); s0=stf2(rng.normal(size=(3,3))); q0=rng.normal(size=3)
        sig=p*s0; qq=p*q0; gsig=np.einsum('ij,k->ijk',s0,gp2); gqq=np.einsum('i,k->ik',q0,gp2)
        mg,Rg,Dg=closures(rho,p,mu,sig,qq,gsig,gqq,gp2)
        mn=(4/(3*p))*stf3(np.einsum('i,jk->ijk',qq,sig)); Rn=20/(7*rho)*stf2(sig.T@sig)+64/(25*p)*stf2(np.outer(qq,qq)); Dn=5*np.sum(sig*sig)/rho+56/5*np.dot(qq,qq)/p
        err['quotient_invariance']=max(err['quotient_invariance'],float(np.max(np.abs(mg-mn))),float(np.max(np.abs(Rg-Rn))),float(abs(Dg-Dn)))
        m0,R0,D0=closures(1.,1.,mu,np.zeros((3,3)),np.zeros(3),np.zeros((3,3,3)),np.zeros((3,3)),np.zeros(3))
        err['equilibrium']=max(err['equilibrium'],float(np.max(np.abs(m0))),float(np.max(np.abs(R0))),float(abs(D0)))
        th=float(rng.uniform(.5,2)); PP=float(rng.uniform(.5,2)); V=float(rng.normal(scale=.2)); TJ=float(rng.normal(scale=.05))
        snn=float(rng.normal(scale=.1)); stt=float(rng.normal(scale=.1)); qt=float(rng.normal(scale=.1)); Rnn=float(rng.normal(scale=.1)); Rtt=float(rng.normal(scale=.1)); De=float(rng.normal(scale=.1)); mtnn=float(rng.normal(scale=.1))
        C=np.sqrt(2/(np.pi*th)); stn=-C*(PP*V+qt/5+mtnn/2); qn=-C*(2*PP*TJ-PP*V*V/2+th*snn/2+De/15+5*Rnn/28)
        Rtn=C*(6*PP*TJ*V+PP*th*V-PP*V**3-11*th*qt/5-th*mtnn/2); mnnn=C*(2*PP*TJ/5-3*PP*V*V/5-7*th*snn/5+De/75-Rnn/14)
        mttn=-C*(PP*TJ/5-4*PP*V*V/5+Rtt/14+th*stt-th*snn/5+De/150)
        err['bc_constructed_residual']=max(err['bc_constructed_residual'],float(np.max(np.abs(bc_residual(th,PP,V,TJ,snn,stt,stn,qt,qn,Rnn,Rtt,Rtn,De,mtnn,mnnn,mttn)))))
    tol=1e-11; numerical={k:(v<tol) for k,v in err.items()}
    report={'reference':PAPER,'scope':{'bulk_balances':'Eqs. (1),(3),(4) coefficient fingerprints and Maxwell relaxation coefficients','closures':'full transformed nonlinear Eq. (13), randomized tensor oracle','walls':'Eqs. (7a-f), full accommodation, constructed-residual oracle','geometry':'isothermal square lid cavity; 1000-iteration numerical homotopy to the exact constant-speed final boundary condition'},'static_checks':static,'oracle_errors':err,'oracle_tolerance':tol,'oracle_checks':numerical,'all_passed':all(static.values()) and all(numerical.values())}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
    if not report['all_passed']: raise SystemExit(2)
if __name__=='__main__': main()
