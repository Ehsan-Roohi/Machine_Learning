#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import h5py, numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter

FIELDS=("ro","p","t","u1","u2","qx","qy","sigmaxx","sigmaxy","sigmayy","Rxx","Rxy","Ryy","Delta")
AF_FIELDS=("t","qx","qy","Rxx","Rxy","Ryy","Delta")
MILESTONES=(40000,60000,80000,100000)

def scalar(h,k): return float(np.asarray(h[k]).reshape(-1)[0])
def f2(h,k):
    a=np.asarray(h[k],dtype=float)
    if a.ndim==3: a=a[0]
    if a.ndim!=2: raise ValueError(f"{k}: {a.shape}")
    return a

def health(path:Path):
    out={'path':str(path),'exists':path.is_file(),'bad':[],'nstep':None,'time':None,'positive':{}}
    if not path.is_file(): return out
    with h5py.File(path,'r') as h:
        def visit(name,obj):
            if isinstance(obj,h5py.Dataset) and obj.dtype.kind in 'fc':
                if not np.isfinite(np.asarray(obj[...])).all(): out['bad'].append(name)
        h.visititems(visit)
        out['nstep']=int(scalar(h,'nstep')); out['time']=scalar(h,'time')
        for k in ('ro','p','t'):
            a=np.asarray(h[k],dtype=float); out['positive'][k]={'min':float(a.min()),'max':float(a.max()),'ok':bool(a.min()>0)}
    out['finite']=not out['bad']; out['positive_rho_p_t']=all(v['ok'] for v in out['positive'].values())
    return out

def rel(new,old):
    rn=float(np.sqrt(np.mean(new*new))); ro=float(np.sqrt(np.mean(old*old)))
    if max(rn,ro)<1e-14: return 0.0
    return float(100*np.sqrt(np.mean((new-old)**2))/(0.5*(rn+ro)+1e-300))

def compare(a:Path,b:Path):
    with h5py.File(a,'r') as ha,h5py.File(b,'r') as hb:
        return {k:rel(f2(hb,k),f2(ha,k)) for k in FIELDS}

def interp(fields,x,y,n=160):
    xc=(np.arange(n)+0.5)/n; yc=(np.arange(n)+0.5)/n; xx,yy=np.meshgrid(xc,yc)
    pts=np.column_stack((yy.ravel(),xx.ravel())); out={}
    for k,v in fields.items(): out[k]=RegularGridInterpolator((y,x),v,bounds_error=False,fill_value=None)(pts).reshape(n,n)
    return out,xc,yc

def af_metrics(flow:Path,grid:Path):
    with h5py.File(flow,'r') as h: fields={k:f2(h,k) for k in AF_FIELDS}
    with h5py.File(grid,'r') as h:
        X=np.asarray(h['x'],dtype=float); Y=np.asarray(h['y'],dtype=float)
        if X.ndim==3: X=X[0];Y=Y[0]
        x=X[0,:]; y=Y[:,0]
    a,xc,yc=interp(fields,x,y); sm=lambda z:uniform_filter(z,size=7,mode='nearest')
    T=sm(a['t']); qx=sm(a['qx']); qy=sm(a['qy']); dTy,dTx=np.gradient(T,yc,xc,edge_order=2)
    qm=np.hypot(qx,qy); gm=np.hypot(dTx,dTy); den=qm*gm; ia=np.full_like(T,np.nan)
    valid=np.isfinite(den)&(den>1e-14); ia[valid]=(qx[valid]*dTx[valid]+qy[valid]*dTy[valid])/den[valid]
    active=valid&(qm>=0.05*np.nanmax(qm))&(gm>=0.05*np.nanmax(gm)); af=active&(ia>0)
    if not np.any(af): raise ValueError('empty AF set')
    rxx,rxy,ryy=(sm(a[k]) for k in ('Rxx','Rxy','Ryy')); D=sm(a['Delta'])
    drxxy,drxxx=np.gradient(rxx,yc,xc,edge_order=2); drxyy,drxyx=np.gradient(rxy,yc,xc,edge_order=2); dryyy,_=np.gradient(ryy,yc,xc,edge_order=2); dDy,dDx=np.gradient(D,yc,xc,edge_order=2)
    divx=drxxx+drxyy; divy=drxyx+dryyy; ok=qm>1e-14; pr=np.full_like(T,np.nan); pd=np.full_like(T,np.nan)
    pr[ok]=(qx[ok]*divx[ok]+qy[ok]*divy[ok])/qm[ok]; pd[ok]=(qx[ok]*dDx[ok]+qy[ok]*dDy[ok])/(3*qm[ok])
    mask=af&np.isfinite(pr)&np.isfinite(pd); rp=float(np.sqrt(np.mean(pr[mask]**2))); rd=float(np.sqrt(np.mean(pd[mask]**2))); chi=np.abs(pd[mask])/(np.abs(pr[mask])+np.abs(pd[mask])+1e-300)
    return {'f_AF_active':float(af.sum()/active.sum()),'mean_IAF_AF':float(np.mean(ia[af])),'PDelta_over_PR':float(rd/rp),'mean_chiDelta':float(np.mean(chi)),'active_cells':int(active.sum()),'anti_fourier_cells':int(af.sum())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--snapshots',type=Path,required=True); ap.add_argument('--grid',type=Path,required=True); ap.add_argument('--final-flow',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    report={'case':'nonlinear Rana-R13 U100 Kn0.05 N32 continuation from validated 20k checkpoint','restart_seed_step':20000,'dt':1e-5,'milestones':{},'cross_milestone_changes':{},'caveats':['restart continuation, although this Rana wall fixed-point is checkpoint-invariant','cavity corner treatment remains order-dependent and is not publication-grade','target-step completion is not proof of convergence']}
    prev=20000; prev_path=None
    for target in MILESTONES:
        paths=[a.snapshots/f'flowfield_{s}.h5' for s in (target-2000,target-1000,target)]
        hs=[health(p) for p in paths]
        if [h['nstep'] for h in hs] != [target-2000,target-1000,target]: raise SystemExit(f'bad checkpoint sequence at {target}: {hs}')
        if not all(h.get('finite') and h.get('positive_rho_p_t') for h in hs): raise SystemExit(f'bad HDF5 health at {target}')
        i1=compare(paths[0],paths[1]); i2=compare(paths[1],paths[2]); steady=all(i1[k]<=0.1 and i2[k]<=0.1 for k in FIELDS) and all(i2[k]<=max(1.1*i1[k],1e-6) for k in FIELDS)
        report['milestones'][str(target)]={'time':hs[-1]['time'],'health':hs[-1],'last_two_1000_step_intervals':[{str(target-2000)+'_to_'+str(target-1000):i1},{str(target-1000)+'_to_'+str(target):i2}],'steady_three_checkpoint_gate':bool(steady),'anti_fourier_metrics':af_metrics(paths[-1],a.grid)}
        if prev_path is not None: report['cross_milestone_changes'][f'{prev}_to_{target}']=compare(prev_path,paths[-1])
        prev=target; prev_path=paths[-1]
    final=health(a.final_flow); report['final_current_health']=final
    report['completed_all_milestones']=final.get('nstep')==100000 and final.get('finite') and final.get('positive_rho_p_t')
    report['scientific_status']='stable continuation complete; convergence must be judged milestone by milestone; not publication-grade while corner treatment remains unresolved'
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
    if not report['completed_all_milestones']: raise SystemExit(2)
if __name__=='__main__': main()
