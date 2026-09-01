#!/usr/bin/env python3
"""Reduce off-wall particle snapshots to blockwise incoming half-range moments."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

DEPTHS = np.array([0.25, 0.5, 1.0, 2.0])
NAMES = np.array(["n_over_ninf", "incoming_fraction", "J0", "Jnn", "Jnt",
                  "Jnnn", "Jn_tt", "Jn_zz", "Jnn_t", "Jn_energy"])


def surface_elements(path: Path):
    lines=[x.strip() for x in path.read_text().splitlines()]; points={}; elems=[]; sec=None
    for line in lines:
        if line=="Points": sec="p"; continue
        if line=="Lines": sec="l"; continue
        f=line.split()
        try:
            if sec=="p" and len(f)==3: points[int(f[0])]=(float(f[1]),float(f[2]))
            elif sec=="l" and len(f)>=4 and int(f[1])==2:
                elems.append((int(f[0]), points[int(f[2])], points[int(f[3])]))
        except (ValueError, KeyError): pass
    elems.sort()
    sid=np.array([e[0] for e in elems]); v1=np.array([e[1] for e in elems]); v2=np.array([e[2] for e in elems])
    tangent=v2-v1; length=np.linalg.norm(tangent,axis=1); tangent/=length[:,None]
    normal=np.column_stack((-tangent[:,1],tangent[:,0]))
    tcommon=tangent.copy(); flip=tcommon[:,0]<0; tcommon[flip]*=-1
    return sid,0.5*(v1+v2),tcommon,normal,length


def snapshots(path: Path):
    with gzip.open(path,"rt") as stream:
        while True:
            line=stream.readline()
            if not line: return
            if not line.startswith("ITEM: TIMESTEP"): continue
            step=int(stream.readline()); stream.readline(); n=int(stream.readline())
            stream.readline(); stream.readline(); stream.readline(); stream.readline()
            header=stream.readline().split()[2:]
            rows=np.array([[float(x) for x in stream.readline().split()] for _ in range(n)])
            yield step,header,rows


def main():
    p=argparse.ArgumentParser(); p.add_argument("case_dir",type=Path); p.add_argument("--label",default="offwall_half_range")
    p.add_argument("--block-steps",type=int,default=1000); p.add_argument("--steps",type=int,default=5000); a=p.parse_args()
    meta=json.loads((a.case_dir/"metadata.json").read_text()); phys=meta["physics"]
    lam=float(phys["mean_free_path_m"]); uinf=float(phys["stream_speed_m_per_s"])
    ninf=float(phys["number_density_m_minus_3"]); fnum=float(phys["fnum_particles_per_simulator"])
    sid,mid,tangent,normal,length=surface_elements(a.case_dir/"wall.surf")
    nblock=a.steps//a.block_steps; shape=(nblock,len(DEPTHS),len(sid))
    count=np.zeros(shape); incoming=np.zeros(shape); sums=np.zeros(shape+(8,)); nsnap=np.zeros(nblock,int)
    half_n=max(0.25*lam,0.5*min(float(phys["dx_m"]),float(phys["dy_m"])))
    probes=[mid+d*lam*normal for d in DEPTHS]; trees=[cKDTree(x) for x in probes]
    dump=a.case_dir/"output"/a.label/"particles.gz"
    for step,header,rows in snapshots(dump):
        block=min((step-1)//a.block_steps,nblock-1); nsnap[block]+=1; col={x:i for i,x in enumerate(header)}
        xy=rows[:,[col["x"],col["y"]]]; vel=rows[:,[col["vx"],col["vy"],col["vz"]]]
        for k,tree in enumerate(trees):
            _,idx=tree.query(xy); delta=xy-probes[k][idx]
            dt=np.einsum("ij,ij->i",delta,tangent[idx]); dn=np.einsum("ij,ij->i",delta,normal[idx])
            accept=(np.abs(dt)<=0.5*length[idx])&(np.abs(dn)<=half_n)
            ii=idx[accept]; vv=vel[accept]; cn=np.einsum("ij,ij->i",vv[:,:2],normal[ii]); ct=np.einsum("ij,ij->i",vv[:,:2],tangent[ii]); cz=vv[:,2]
            np.add.at(count[block,k],ii,1); inc=cn<0; jj=ii[inc]; cn=cn[inc]/uinf; ct=ct[inc]/uinf; cz=cz[inc]/uinf; w=-cn
            np.add.at(incoming[block,k],jj,1)
            vals=np.column_stack((w,w*(-cn),w*ct,w*cn*cn,w*ct*ct,w*cz*cz,w*cn*ct,w*(cn*cn+ct*ct+cz*cz)))
            for q in range(vals.shape[1]): np.add.at(sums[block,k,:,q],jj,vals[:,q])
    if np.any(nsnap==0): raise RuntimeError(f"empty block(s): {nsnap}")
    area=length[None,None,:]*(2*half_n); avg_count=count/nsnap[:,None,None]
    density=avg_count*fnum/(area*ninf); features=np.zeros(shape+(len(NAMES),))
    features[...,0]=density; features[...,1]=incoming/np.maximum(count,1)
    features[...,2:]=sums/np.maximum(count[...,None],1)
    out=a.case_dir/"output"/a.label/"offwall_incoming_moments.npz"
    np.savez_compressed(out,features=features,feature_names=NAMES,depths_lambda=DEPTHS,
                        surface_id=sid,midpoint=mid,tangent=tangent,normal=normal,
                        block_snapshots=nsnap,case_id=meta["case_id"],normal_halfwidth_m=half_n)
    if not np.isfinite(features).all(): raise RuntimeError("non-finite reduced features")
    print(f"OFFWALL_REDUCTION_PASS={out} shape={features.shape} snapshots={nsnap.tolist()}")

if __name__=="__main__": main()
