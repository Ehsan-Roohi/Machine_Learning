#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np

# Ubuntu 24.04's apt NumPy is older than 2.0.  Keep the analyzer compatible
# with both APIs instead of letting a successful solver run fail in postprocess.
_trapezoid = getattr(np, 'trapezoid', np.trapz)

PAPER_D=0.1585; PAPER_G=0.1893; GAMMA=5.0/3.0
FIELDS=('ro','p','t','u1','u2','qx','qy','sigmaxx','sigmaxy','sigmayy','Rxx','Rxy','Ryy','Delta')

def scalar(h,k): return float(np.asarray(h[k]).reshape(-1)[0])
def field(h,k):
    a=np.asarray(h[k],dtype=float)
    return a[0] if a.ndim==3 else a

def read_state(path:Path):
    import h5py
    with h5py.File(path,'r') as h:
        step=int(scalar(h,'nstep')); tm=scalar(h,'time')
        data={k:field(h,k) for k in FIELDS if k in h}
        bad=[]
        def visit(n,o):
            if isinstance(o,h5py.Dataset) and o.dtype.kind in 'fc' and not np.isfinite(o[...]).all(): bad.append(n)
        h.visititems(visit)
    return step,tm,data,bad

def nrms(a,b): return float(np.sqrt(np.mean((a-b)**2))/max(np.sqrt(np.mean(a*a)),1e-300))

def global_metrics(sigmaxy,u1,x,y,mach,speed_ratio,gamma=GAMMA):
    """Return Rana's reduced drag D and centerline flow rate G.

    ASTR stores stress on the dynamic-pressure scale.  Rana/Sharipov's
    low-speed drag coefficient uses sqrt(2)*(sigma_xy/p0)/U*, where
    U*=U_lid/sqrt(theta0).  The former 1/(sqrt(2)*U*) implementation was
    exactly a factor of two too small.
    """
    sigmaxy=np.asarray(sigmaxy,dtype=float); u1=np.asarray(u1,dtype=float)
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    if sigmaxy.shape != (y.size,x.size):
        raise ValueError(f'sigmaxy shape {sigmaxy.shape} != {(y.size,x.size)}')
    if u1.shape != sigmaxy.shape:
        raise ValueError(f'u1 shape {u1.shape} != sigmaxy shape {sigmaxy.shape}')
    if not (np.all(np.diff(x)>0) and np.all(np.diff(y)>0)):
        raise ValueError('x and y must be strictly increasing')
    const2=float(gamma)*float(mach)**2
    speed_ratio=float(speed_ratio)
    if speed_ratio<=0:
        raise ValueError('speed_ratio must be positive')
    if not np.isclose(const2,speed_ratio**2,rtol=2e-8,atol=1e-14):
        raise ValueError(
            f'inconsistent nondimensionalization: gamma*Ma^2={const2}, '
            f'U*^2={speed_ratio**2}'
        )
    iy_top=int(np.argmax(y))
    raw_integral=float(_trapezoid(sigmaxy[iy_top,:],x))
    sigma_over_p0_signed=const2*raw_integral
    reduced_factor=np.sqrt(2.0)/speed_ratio
    reduced_signed=reduced_factor*sigma_over_p0_signed
    umid=np.asarray([np.interp(0.5,x,row) for row in u1])
    G=float(_trapezoid(np.abs(umid),y))
    return {
      'D_reduced_signed':reduced_signed,
      'D_reduced_abs':abs(reduced_signed),
      'D_sigma_over_p0_signed':sigma_over_p0_signed,
      'D_raw_ASTR_integral':raw_integral,
      'D_reduced_stress_factor':reduced_factor,
      'const2':const2,
      'speed_ratio_U_over_sqrt_theta0':speed_ratio,
      'G':G,
    }

def main():
    import h5py
    ap=argparse.ArgumentParser(); ap.add_argument('--case',type=Path,required=True); ap.add_argument('--variant',required=True); ap.add_argument('--target',type=int,required=True); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); case=a.case
    cur=case/'outdat/flowfield.h5'; bak=case/'bakup/flowfield.h5'
    st,tm,f,bad=read_state(cur); bst,bt,bf,bbad=read_state(bak)
    with h5py.File(case/'datin/grid.h5','r') as g:
        X=np.asarray(g['x'],float);Y=np.asarray(g['y'],float);X=X[0] if X.ndim==3 else X;Y=Y[0] if Y.ndim==3 else Y;x=X[0];y=Y[:,0]
    meta=json.loads((case/'rana2013_case_metadata.json').read_text())
    metrics=global_metrics(
        f['sigmaxy'],f['u1'],x,y,meta['Mach'],
        meta['speed_ratio_U_over_sqrt_theta0'],
    )
    D=metrics['D_reduced_abs']; G=metrics['G']
    mass=float(_trapezoid(_trapezoid(f['ro'],x,axis=1),y))
    changes={k:nrms(f[k],bf[k]) for k in FIELDS if k in f and k in bf}
    log='\n'.join((p.read_text(errors='replace') if p.exists() else '') for p in [case/'logs/astr.log',case/'logs/astr_time.log'])
    pats=['RANA_P_FAILURE','SIGFPE','IEEE_INVALID_FLAG','Floating-point exception','COMPUTATION CRASHED','Segmentation fault','NaN','Infinity']
    markers={p:len(re.findall(re.escape(p),log,re.I)) for p in pats}
    status=json.loads((case/'analysis/run_status.json').read_text()) if (case/'analysis/run_status.json').exists() else {}
    thermo={k:{'min':float(np.min(f[k])),'max':float(np.max(f[k]))} for k in ['ro','p','t']}
    extrema={k:{'min':float(np.min(f[k])),'max':float(np.max(f[k])),'maxabs':float(np.max(np.abs(f[k])))} for k in ['qx','qy','Rxx','Rxy','Ryy','Delta'] if k in f}
    finite=(not bad and not bbad); positive=all(thermo[k]['min']>0 for k in thermo)
    completed=(st==a.target and int(status.get('return_code',1))==0 and finite and positive and all(v==0 for v in markers.values()))
    max_change=max(changes.values()) if changes else None
    convergence_threshold=1.0e-3
    converged=bool(max_change is not None and max_change<=convergence_threshold)
    report={
      'variant':a.variant,'target':a.target,'actual':st,'time':tm,'backup_step':bst,'backup_time':bt,
      'solver_status':status,'markers':markers,'finite':finite,'positive':positive,'thermodynamic_extrema':thermo,'higher_moment_extrema':extrema,
      'mass':mass,'mass_defect_percent':100*(mass-1.0),
      'D':D,'D_reduced_signed':metrics['D_reduced_signed'],
      'D_sigma_over_p0_signed':metrics['D_sigma_over_p0_signed'],
      'D_raw_ASTR_integral':metrics['D_raw_ASTR_integral'],
      'D_reduced_stress_factor':metrics['D_reduced_stress_factor'],
      'D_paper':PAPER_D,'D_error_percent':100*(D-PAPER_D)/PAPER_D,
      'G':G,'G_paper':PAPER_G,'G_error_percent':100*(G-PAPER_G)/PAPER_G,
      'current_vs_backup_normalized_RMS':changes,'max_current_vs_backup_normalized_RMS':max_change,
      'convergence_threshold_normalized_RMS':convergence_threshold,'converged':converged,
      'completed_cleanly':completed,'publication_grade':False,
      'scientific_status':(
        'clean-completion and convergence are separate gates; this is a '
        'diagnostic pseudo-time variant, not the exact Rana Eq.(20) coupled matrix discretization'
      ),
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
