#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,shutil
from pathlib import Path
import h5py,numpy as np
GAMMA=5.0/3.0
R_ARGON=208.13
T0=273.0
ULID=50.0
MAXTOK=16

# The ASTR time integrator is explicit, whereas Rana et al. solve the steady
# nonlinear system implicitly with QMR.  The regularizing terms contain one
# additional spatial derivative and impose a parabolic grid/Kn restriction.
# N=75, Kn=0.0798, dt=1e-5 is the anchor configuration tested before the
# production matrix.  For finer grids or larger Kn, --dt is treated as an
# upper bound and reduced with dx^2/Kn scaling.  This changes only the
# pseudo-time path, never the steady R13 equations or wall fixed point.
DT_ANCHOR=1.0e-5
N_ANCHOR=75.0
KN_ANCHOR=0.0798

def fd(x):
    s=f'{x:.8e}'.replace('e','d')
    if len(s)>MAXTOK: raise ValueError(s)
    return s

def replace(lines,prefix,value):
    for i,l in enumerate(lines):
        if l.strip().startswith(prefix): lines[i+1]=value; return
    raise RuntimeError(prefix)

def decomp(cells):
    return (2,2) if cells >= 32 else (1,1)

def partitions(n, parts):
    q,r=divmod(n,parts); sizes=[q+(i<r) for i in range(parts)]
    starts=[]; cur=0
    for z in sizes: starts.append(cur); cur+=z
    assert cur==n
    return sizes,starts

def stable_dt(requested_dt:float,cells:int,paper_kn:float)->tuple[float,float]:
    if requested_dt <= 0:
        raise ValueError('dt must be positive')
    grid_kn_cap=DT_ANCHOR*(N_ANCHOR/float(cells))**2*(KN_ANCHOR/paper_kn)
    # Never make a coarser/easier case larger than the already validated
    # anchor step, and never exceed the caller's requested value.
    cap=min(DT_ANCHOR,grid_kn_cap)
    used=min(requested_dt,cap)
    if not math.isfinite(used) or used <= 0:
        raise ValueError(f'invalid stable dt: requested={requested_dt}, cap={cap}')
    return used,cap

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--template-case',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--paper-kn',type=float,required=True)
    ap.add_argument('--grid-points',type=int,default=75,help='Paper mesh size N: cells/intervals per direction')
    ap.add_argument('--maxstep',type=int,default=100000)
    ap.add_argument('--dt',type=float,default=2.5e-5,help='maximum explicit pseudo-time step; grid/Kn stability cap is applied automatically')
    ap.add_argument('--checkpoint-every',type=int,default=5000)
    a=ap.parse_args()
    if a.paper_kn<=0 or a.grid_points<20 or a.maxstep<=0: raise ValueError('invalid case')
    # Rana et al. describe the convergence reference as 500x500 cells and
    # report mesh sizes N=40,...,400. ASTR stores N+1 physical nodes.
    cells=a.grid_points
    dt_used,dt_cap=stable_dt(a.dt,cells,a.paper_kn)
    px,py=decomp(cells)
    out=a.output.resolve(); shutil.rmtree(out,ignore_errors=True); shutil.copytree(a.template_case,out)
    for n in ['outdat','bakup','testout','monitor','islice','jslice','kslice']:
        shutil.rmtree(out/n,ignore_errors=True); (out/n).mkdir(exist_ok=True)
    dat=out/'datin'; inp=next(dat.glob('input.astr*')); ctrl=dat/'controller'
    speed_ratio=ULID/math.sqrt(R_ARGON*T0)
    mach=ULID/math.sqrt(GAMMA*R_ARGON*T0)
    reynolds=speed_ratio/a.paper_kn
    meanfree_kn=math.sqrt(math.pi/2.0)*a.paper_kn
    lines=inp.read_text().splitlines()
    replace(lines,'# im,jm,km',f'{cells},{cells},0')
    replace(lines,'# lihomo,ljhomo,lkhomo','f,f,t')
    replace(lines,'# lrestar','f')
    replace(lines,'# alfa_filter, kcutoff',f'0.49d0, {max(8,int(.75*cells))}')
    replace(lines,'# ref_t,reynolds,mach,gamma',f'{T0:.1f}d0, {fd(reynolds)}, {fd(mach)}, {fd(GAMMA)}')
    replace(lines,'# turbmode,iomode, moment','none,h,r13')
    replace(lines,'# ninit','0')
    inp.write_text('\n'.join(lines)+'\n')
    cl=ctrl.read_text().splitlines()
    replace(cl,'# lwsequ,lwslic,lavg,lcracon','       f,     f,   f,      f')
    replace(cl,'# maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg',f'{a.maxstep:9d}, {a.checkpoint_every:7d}, {a.checkpoint_every:7d},    1000,     500,   500')
    replace(cl,'# deltat',fd(dt_used)); ctrl.write_text('\n'.join(cl)+'\n')
    c=np.linspace(0.,1.,cells+1); X,Y=np.meshgrid(c,c,indexing='xy'); X=X[None];Y=Y[None];Z=np.zeros_like(X)
    with h5py.File(dat/'grid.h5','w') as h: h['x']=X;h['y']=Y;h['z']=Z
    xs,x0=partitions(cells,px); ys,y0=partitions(cells,py)
    pl=['    isize     jsize     ksize',f'{px:9d}{py:10d}{1:10d}','     Rank       Irk       Jrk       Krk        IM        JM        KM        I0        J0        K0']
    rank=0
    for jy in range(py):
      for ix in range(px):
        pl.append(f'{rank:9d}{ix:10d}{jy:10d}{0:10d}{xs[ix]:10d}{ys[jy]:10d}{0:10d}{x0[ix]:10d}{y0[jy]:10d}{0:10d}');rank+=1
    (dat/'parallel.info').write_text('\n'.join(pl)+'\n')
    meta={
      'paper':'Rana et al. JCP 236 (2013) 169-186',
      'paper_kn':a.paper_kn,
      'mean_free_path_kn_convention':meanfree_kn,
      'paper_mesh_N_cells_per_direction':a.grid_points,
      'ASTR_grid_nodes_per_direction':cells+1,
      'ASTR_intervals_im_jm':cells,
      'mpi_decomposition':[px,py,1],
      'mpi_ranks':px*py,
      'T0_K':T0,
      'lid_velocity_m_per_s':ULID,
      'specific_gas_constant_J_kgK':R_ARGON,
      'gas_mapping':'monatomic argon, matching the DSMC reference family cited by Rana',
      'speed_ratio_U_over_sqrt_theta0':speed_ratio,
      'Mach':mach,
      'Reynolds':reynolds,
      'paper_kn_identity':'Kn=sqrt(gamma)*Ma/Re=mu/(rho*sqrt(theta)*L)',
      'dt_requested_upper_bound':a.dt,
      'dt_grid_kn_cap':dt_cap,
      'dt':dt_used,
      'dt_scaling':'explicit R13 stability cap: 1e-5*(75/N)^2*(0.0798/Kn), capped at 1e-5',
      'maxstep':a.maxstep,
      'pseudo_time_horizon':dt_used*a.maxstep,
      'initial_state':'global equilibrium at rest',
      'moment':'full nonlinear transformed Maxwell R13 Eq. (13)',
      'walls':'Rana Eq. (7a-f), chi=1, isothermal; 1000-iteration homotopy then exact Uwall=50 m/s final condition',
    }
    (out/'rana2013_case_metadata.json').write_text(json.dumps(meta,indent=2)); print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
