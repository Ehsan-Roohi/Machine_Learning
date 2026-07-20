#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import h5py, numpy as np

PAPER_D=0.1585; PAPER_G=0.1893; GAMMA=5.0/3.0
FIELDS=('ro','p','t','u1','u2','qx','qy','sigmaxx','sigmaxy','sigmayy','Rxx','Rxy','Ryy','Delta')

def scalar(h,k): return float(np.asarray(h[k]).reshape(-1)[0])
def field(h,k):
    a=np.asarray(h[k],dtype=float)
    return a[0] if a.ndim==3 else a

def read_state(path:Path):
    with h5py.File(path,'r') as h:
        step=int(scalar(h,'nstep')); tm=scalar(h,'time')
        data={k:field(h,k) for k in FIELDS if k in h}
        bad=[]
        def visit(n,o):
            if isinstance(o,h5py.Dataset) and o.dtype.kind in 'fc' and not np.isfinite(o[...]).all(): bad.append(n)
        h.visititems(visit)
    return step,tm,data,bad

def nrms(a,b): return float(np.sqrt(np.mean((a-b)**2))/max(np.sqrt(np.mean(a*a)),1e-300))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--case',type=Path,required=True); ap.add_argument('--variant',required=True); ap.add_argument('--target',type=int,required=True); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); case=a.case
    cur=case/'outdat/flowfield.h5'; bak=case/'bakup/flowfield.h5'
    st,tm,f,bad=read_state(cur); bst,bt,bf,bbad=read_state(bak)
    with h5py.File(case/'datin/grid.h5','r') as g:
        X=np.asarray(g['x'],float);Y=np.asarray(g['y'],float);X=X[0] if X.ndim==3 else X;Y=Y[0] if Y.ndim==3 else Y;x=X[0];y=Y[:,0]
    meta=json.loads((case/'rana2013_case_metadata.json').read_text())
    const2=GAMMA*float(meta['Mach'])**2
    signed=float(np.trapezoid(const2*f['sigmaxy'][-1,:],x)); D=-signed
    umid=np.asarray([np.interp(0.5,x,row) for row in f['u1']]); G=float(np.trapezoid(np.abs(umid),y))
    mass=float(np.trapezoid(np.trapezoid(f['ro'],x,axis=1),y))
    changes={k:nrms(f[k],bf[k]) for k in FIELDS if k in f and k in bf}
    log='\n'.join((p.read_text(errors='replace') if p.exists() else '') for p in [case/'logs/astr.log',case/'logs/astr_time.log'])
    pats=['RANA_P_FAILURE','SIGFPE','IEEE_INVALID_FLAG','Floating-point exception','COMPUTATION CRASHED','Segmentation fault','NaN','Infinity']
    markers={p:len(re.findall(re.escape(p),log,re.I)) for p in pats}
    status=json.loads((case/'analysis/run_status.json').read_text()) if (case/'analysis/run_status.json').exists() else {}
    thermo={k:{'min':float(np.min(f[k])),'max':float(np.max(f[k]))} for k in ['ro','p','t']}
    extrema={k:{'min':float(np.min(f[k])),'max':float(np.max(f[k])),'maxabs':float(np.max(np.abs(f[k])))} for k in ['qx','qy','Rxx','Rxy','Ryy','Delta'] if k in f}
    finite=(not bad and not bbad); positive=all(thermo[k]['min']>0 for k in thermo)
    completed=(st==a.target and int(status.get('return_code',1))==0 and finite and positive and all(v==0 for v in markers.values()))
    report={
      'variant':a.variant,'target':a.target,'actual':st,'time':tm,'backup_step':bst,'backup_time':bt,
      'solver_status':status,'markers':markers,'finite':finite,'positive':positive,'thermodynamic_extrema':thermo,'higher_moment_extrema':extrema,
      'mass':mass,'mass_defect_percent':100*(mass-1.0),
      'D':D,'D_signed_integral':signed,'D_paper':PAPER_D,'D_error_percent':100*(D-PAPER_D)/PAPER_D,
      'G':G,'G_paper':PAPER_G,'G_error_percent':100*(G-PAPER_G)/PAPER_G,
      'current_vs_backup_normalized_RMS':changes,'max_current_vs_backup_normalized_RMS':max(changes.values()) if changes else None,
      'completed_cleanly':completed,
      'scientific_status':'diagnostic pseudo-time variant; not the exact Rana Eq.(20) coupled matrix discretization',
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
