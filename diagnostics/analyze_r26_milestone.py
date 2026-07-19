#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import h5py,numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter
FIELDS=('ro','p','t','u1','u2','qx','qy','sigmaxx','sigmaxy','sigmayy','Rxx','Rxy','Ryy','Delta')
AF=('t','qx','qy','Rxx','Rxy','Ryy','Delta')
def s(h,k):return float(np.asarray(h[k]).reshape(-1)[0])
def f(h,k):
 a=np.asarray(h[k],float); return a[0] if a.ndim==3 else a
def read(p):
 with h5py.File(p,'r') as h:return int(s(h,'nstep')),s(h,'time'),{k:f(h,k) for k in FIELDS}
def rel(n,o):return float(np.sqrt(np.mean((n-o)**2))/max(np.sqrt(np.mean(n*n)),1e-30))
def metrics(p,grid):
 with h5py.File(p,'r') as h:fs={k:f(h,k) for k in AF}
 with h5py.File(grid,'r') as h:
  X=np.asarray(h['x'],float);Y=np.asarray(h['y'],float);X=X[0] if X.ndim==3 else X;Y=Y[0] if Y.ndim==3 else Y;x=X[0];y=Y[:,0]
 n=160;xc=(np.arange(n)+.5)/n;yc=(np.arange(n)+.5)/n;xx,yy=np.meshgrid(xc,yc);pts=np.column_stack((yy.ravel(),xx.ravel()));a={k:RegularGridInterpolator((y,x),v,bounds_error=False,fill_value=None)(pts).reshape(n,n) for k,v in fs.items()};sm=lambda z:uniform_filter(z,7,mode='nearest')
 T=sm(a['t']);qx=sm(a['qx']);qy=sm(a['qy']);dTy,dTx=np.gradient(T,yc,xc,edge_order=2);qm=np.hypot(qx,qy);gm=np.hypot(dTx,dTy);den=qm*gm;ia=np.full_like(T,np.nan);v=np.isfinite(den)&(den>1e-14);ia[v]=(qx[v]*dTx[v]+qy[v]*dTy[v])/den[v];act=v&(qm>=.05*np.nanmax(qm))&(gm>=.05*np.nanmax(gm));af=act&(ia>0)
 Rxx,Rxy,Ryy=(sm(a[k]) for k in ('Rxx','Rxy','Ryy'));D=sm(a['Delta']);Rxx_y,Rxx_x=np.gradient(Rxx,yc,xc,edge_order=2);Rxy_y,Rxy_x=np.gradient(Rxy,yc,xc,edge_order=2);Ryy_y,_=np.gradient(Ryy,yc,xc,edge_order=2);D_y,D_x=np.gradient(D,yc,xc,edge_order=2);dx=Rxx_x+Rxy_y;dy=Rxy_x+Ryy_y;ok=qm>1e-14;pr=np.full_like(T,np.nan);pd=np.full_like(T,np.nan);pr[ok]=(qx[ok]*dx[ok]+qy[ok]*dy[ok])/qm[ok];pd[ok]=(qx[ok]*D_x[ok]+qy[ok]*D_y[ok])/(3*qm[ok]);m=af&np.isfinite(pr)&np.isfinite(pd);chi=np.abs(pd[m])/(np.abs(pr[m])+np.abs(pd[m])+1e-300)
 return {'f_AF_active':float(af.sum()/act.sum()),'mean_IAF_AF':float(np.mean(ia[af])),'PDelta_over_PR':float(np.sqrt(np.mean(pd[m]**2))/np.sqrt(np.mean(pr[m]**2))),'mean_chiDelta':float(np.mean(chi)),'active_cells':int(act.sum()),'af_cells':int(af.sum())}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--snapshots',type=Path,required=True);ap.add_argument('--grid',type=Path,required=True);ap.add_argument('--target',type=int,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();steps=[a.target-3000,a.target-2000,a.target-1000,a.target]; rec=[];states=[]
 for z in steps:
  p=a.snapshots/f'step_{z:06d}.h5'; st,t,x=read(p); assert st==z; rec.append({'nstep':st,'time':t,'metrics':metrics(p,a.grid),'rho_mean':float(np.mean(x['ro']))});states.append(x)
 intervals=[]
 for i in range(3):
  fr={k:rel(states[i+1][k],states[i][k]) for k in FIELDS};mc={k:abs(rec[i+1]['metrics'][k]-rec[i]['metrics'][k])/max(abs(rec[i+1]['metrics'][k]),1e-12) for k in ('f_AF_active','mean_IAF_AF','PDelta_over_PR','mean_chiDelta')};intervals.append({'from':steps[i],'to':steps[i+1],'field_normalized_rms':fr,'max_field_normalized_rms':max(fr.values()),'density_mean_relative_change':abs(rec[i+1]['rho_mean']/rec[i]['rho_mean']-1),'metric_relative_change':mc,'max_metric_relative_change':max(mc.values())})
 report={'target':a.target,'time':rec[-1]['time'],'records':rec,'intervals':intervals,'steady_convergence_criterion_passed':all(x['max_field_normalized_rms']<=1e-3 and x['density_mean_relative_change']<=1e-5 and x['max_metric_relative_change']<=1e-2 for x in intervals),'scientific_status':'diagnostic continuation only; relaxed non-serialized wall memory resets at each job and double-owner corners remain unresolved'};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
