#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import h5py
import numpy as np

GAMMA=5.0/3.0

def arr(h,key):
    a=np.asarray(h[key],dtype=float)
    return a[0] if a.ndim==3 else a

def tensor_fields(h):
    s=np.zeros(arr(h,'sigmaxx').shape+(3,3))
    s[...,0,0]=arr(h,'sigmaxx'); s[...,0,1]=s[...,1,0]=arr(h,'sigmaxy')
    s[...,0,2]=s[...,2,0]=arr(h,'sigmaxz'); s[...,1,1]=arr(h,'sigmayy')
    s[...,1,2]=s[...,2,1]=arr(h,'sigmayz'); s[...,2,2]=arr(h,'sigmazz')
    R=np.zeros_like(s)
    R[...,0,0]=arr(h,'Rxx'); R[...,0,1]=R[...,1,0]=arr(h,'Rxy')
    R[...,0,2]=R[...,2,0]=arr(h,'Rxz'); R[...,1,1]=arr(h,'Ryy')
    R[...,1,2]=R[...,2,1]=arr(h,'Ryz'); R[...,2,2]=-R[...,0,0]-R[...,1,1]
    M=np.zeros(s.shape[:-2]+(3,3,3))
    vals={(0,0,0):arr(h,'mxxx'),(0,0,1):arr(h,'mxxy'),(0,0,2):arr(h,'mxxz'),
          (0,1,1):arr(h,'mxyy'),(1,1,1):arr(h,'myyy'),(1,1,2):arr(h,'myyz'),
          (0,1,2):arr(h,'mxyz')}
    import itertools
    for inds,v in vals.items():
        for p in set(itertools.permutations(inds)): M[...,p[0],p[1],p[2]]=v
    m133=-M[...,0,0,0]-M[...,0,1,1]
    m233=-M[...,0,0,1]-M[...,1,1,1]
    m333=-M[...,0,0,2]-M[...,1,1,2]
    for inds,v in [((0,2,2),m133),((1,2,2),m233),((2,2,2),m333)]:
        for p in set(itertools.permutations(inds)): M[...,p[0],p[1],p[2]]=v
    return s,R,M

def contractions(S,R,M,q,n,t):
    snn=np.einsum('i,...ij,j->...',n,S,n); stt=np.einsum('i,...ij,j->...',t,S,t)
    stn=np.einsum('i,...ij,j->...',t,S,n)
    Rnn=np.einsum('i,...ij,j->...',n,R,n); Rtt=np.einsum('i,...ij,j->...',t,R,t)
    Rtn=np.einsum('i,...ij,j->...',t,R,n)
    mtnn=np.einsum('i,j,k,...ijk->...',t,n,n,M)
    mnnn=np.einsum('i,j,k,...ijk->...',n,n,n,M)
    mttn=np.einsum('i,j,k,...ijk->...',t,t,n,M)
    qn=np.einsum('...i,i->...',q,n); qt=np.einsum('...i,i->...',q,t)
    return snn,stt,stn,Rnn,Rtt,Rtn,mtnn,mnnn,mttn,qn,qt

def relative_residual(lhs,*rhs_terms):
    rhs=sum(rhs_terms)
    den=np.abs(lhs)+sum(np.abs(x) for x in rhs_terms)+1e-30
    return np.abs(lhs-rhs)/den

def wall_residuals(h,const2):
    S,R,M=tensor_fields(h)
    q=np.stack([arr(h,'qx'),arr(h,'qy'),arr(h,'qz')],axis=-1)
    vel=np.stack([arr(h,'u1'),arr(h,'u2'),arr(h,'u3')],axis=-1)
    theta=arr(h,'t')/const2; p=arr(h,'p'); D=arr(h,'Delta')
    walls=[('left',(slice(1,-1),0),np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.zeros(3)),
           ('right',(slice(1,-1),-1),np.array([-1.,0.,0.]),np.array([0.,1.,0.]),np.zeros(3)),
           ('bottom',(0,slice(1,-1)),np.array([0.,1.,0.]),np.array([1.,0.,0.]),np.zeros(3)),
           ('top',(-1,slice(1,-1)),np.array([0.,-1.,0.]),np.array([1.,0.,0.]),np.array([1.,0.,0.]))]
    out={}
    for name,sl,n,t,vw in walls:
        ss=S[sl]; rr=R[sl]; mm=M[sl]; qq=q[sl]; vv=vel[sl]
        th=theta[sl]; pp=p[sl]; de=D[sl]
        snn,stt,stn,Rnn,Rtt,Rtn,mtnn,mnnn,mttn,qn,qt=contractions(ss,rr,mm,qq,n,t)
        P=pp+0.5*stt-de/(120*th)-Rtt/(28*th)
        V=np.einsum('...i,i->...',vv-vw,t); vn=np.einsum('...i,i->...',vv-vw,n)
        T=th-1.0/const2; C=np.sqrt(2.0/(np.pi*th))
        terms={
          '7a_vn':np.abs(vn)/(np.abs(np.einsum('...i,i->...',vv,t))+1e-30),
          '7b_sigma_tn':relative_residual(stn,-C*(P*V+qt/5+mtnn/2)),
          '7c_qn':relative_residual(qn,-C*(2*P*T-0.5*P*V*V+0.5*th*snn+de/15+5*Rnn/28)),
          '7d_Rtn':relative_residual(Rtn,C*(6*P*T*V+P*th*V-P*V**3-11*th*qt/5-th*mtnn/2)),
          '7e_mnnn':relative_residual(mnnn,C*(2*P*T/5-3*P*V*V/5-7*th*snn/5+de/75-Rnn/14)),
          '7f_mttn':relative_residual(mttn,-C*(P*T/5-4*P*V*V/5+Rtt/14+th*stt-th*snn/5+de/150))}
        out[name]={k:{'max':float(np.nanmax(v)),'mean':float(np.nanmean(v))} for k,v in terms.items()}
        out[name]['P_min']=float(np.nanmin(P))
    return out

def checkpoint_change(current,previous):
    if previous is None or not previous.exists(): return None
    keys=['ro','p','t','u1','u2','qx','qy','sigmaxx','sigmaxy','sigmayy','Rxx','Rxy','Ryy','Delta']
    out={}
    with h5py.File(current,'r') as a,h5py.File(previous,'r') as b:
        for k in keys:
            x=np.asarray(a[k],float); y=np.asarray(b[k],float)
            out[k]=float(100*np.sqrt(np.mean((x-y)**2))/(np.sqrt(np.mean(y*y))+1e-30))
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('case',type=Path); ap.add_argument('--reference',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    meta=json.loads((a.case/'rana2013_case_metadata.json').read_text()); ref=json.loads(a.reference.read_text())
    f=a.case/'outdat/flowfield.h5'; prev=a.case/'bakup/flowfield.h5'
    with h5py.File(f,'r') as h, h5py.File(a.case/'datin/grid.h5','r') as g:
        fields={k:arr(h,k) for k in ['ro','p','t','u1','u2','qx','qy','sigmaxy','Delta']}
        for k,x in fields.items():
            if not np.isfinite(x).all(): raise FloatingPointError(k)
        if np.min(fields['ro'])<=0 or np.min(fields['p'])<=0 or np.min(fields['t'])<=0: raise FloatingPointError('non-positive thermodynamic field')
        x=np.asarray(g['x'])[0,0,:]; y=np.asarray(g['y'])[0,:,0]
        const2=GAMMA*meta['Mach']**2
        signed_D=float(np.trapz(const2*fields['sigmaxy'][-1,:],x)); Dabs=abs(signed_D)
        ix=int(np.argmin(abs(x-0.5))); G=float(np.trapz(np.abs(fields['u1'][:,ix]),y))
        wall=wall_residuals(h,const2)
        nstep=int(h['nstep'][0]); time=float(h['time'][0])
        iy=int(np.argmin(abs(y-0.5)))
        rows=[]
        for j,Y in enumerate(y): rows.append({'line':'vertical_x_0p5','coordinate':float(Y),'u_over_lid':float(fields['u1'][j,ix]),'v_over_lid':float(fields['u2'][j,ix]),'T_over_T0':float(fields['t'][j,ix])})
        for i,X in enumerate(x): rows.append({'line':'horizontal_y_0p5','coordinate':float(X),'u_over_lid':float(fields['u1'][iy,i]),'v_over_lid':float(fields['u2'][iy,i]),'T_over_T0':float(fields['t'][iy,i])})
    with (a.output_dir/'centerlines.csv').open('w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=rows[0].keys()); w.writeheader();w.writerows(rows)
    key=f"{meta['paper_kn']:.3f}"; target=ref['global_metrics'].get(key)
    comparison=None
    if target:
        comparison={'paper':target,'D_percent_error':100*(Dabs-target['D_R13'])/target['D_R13'],'G_percent_error':100*(G-target['G_R13'])/target['G_R13']}
    conv=checkpoint_change(f,prev)
    summary={'case_metadata':meta,'nstep':nstep,'time':time,'metrics':{'D_signed':signed_D,'D_abs':Dabs,'G':G},'paper_comparison':comparison,'wall_equation_relative_residuals':wall,'last_checkpoint_relative_RMS_change_percent':conv,'field_ranges':{k:{'min':float(v.min()),'max':float(v.max())} for k,v in fields.items()}}
    (a.output_dir/'rana2013_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
