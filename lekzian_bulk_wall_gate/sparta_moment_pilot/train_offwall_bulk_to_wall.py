#!/usr/bin/env python3
"""Train and blind-test the finite-distance incoming bulk-to-wall surrogate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from analyze_moment_gate import load_case, nrmse

SEEDS=(11,29,47,71,101); FEATURES=("S0","S1","S2","Soff"); TARGETS=("Cp","Cf")


def case_ids(root: Path):
    return [x.strip() for x in (root/"case_list.txt").read_text().splitlines()
            if x.strip() and (x.startswith("BWD_") or x.startswith("FWD_"))]


def load(root: Path, case_id: str, label: str):
    case=load_case(root/case_id); mask=case.region["protrusion"]; sid=case.surface_id[mask]
    path=root/case_id/"output"/label/"offwall_incoming_moments.npz"
    with np.load(path,allow_pickle=False) as z:
        order={int(x):i for i,x in enumerate(z["surface_id"])}
        idx=np.array([order[int(x)] for x in sid]); off=z["features"].mean(axis=0)[:,idx,:]
    off=np.moveaxis(off,0,1).reshape(len(sid),-1)
    if not np.isfinite(off).all(): raise ValueError(f"non-finite off-wall features in {case_id}")
    return case,mask,{"S0":case.features["S0"][mask],"S1":case.features["S1"][mask],
                      "S2":case.features["S2"][mask],"Soff":np.column_stack((case.features["S2"][mask],off))}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--train-root",type=Path,required=True); p.add_argument("--test-root",type=Path,required=True)
    p.add_argument("--label",default="offwall_half_range"); p.add_argument("--out",type=Path,required=True); a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    train=[load(a.train_root,c,a.label) for c in case_ids(a.train_root)]; test=[load(a.test_root,c,a.label) for c in case_ids(a.test_root)]
    rows=[]; saved={}
    for seed in SEEDS:
        for f in FEATURES:
            x=np.vstack([d[f] for _,_,d in train]); y=np.vstack([c.targets[m] for c,m,_ in train])
            model=ExtraTreesRegressor(n_estimators=800,min_samples_leaf=2,max_features=0.85,n_jobs=-1,random_state=seed)
            model.fit(x,y)
            for c,m,d in test:
                pred=model.predict(d[f]); saved.setdefault((f,c.case_id),[]).append(pred)
                for j,t in enumerate(TARGETS): rows.append({"seed":seed,"feature":f,"case_id":c.case_id,"target":t,"nrmse":nrmse(c.targets[m,j],pred[:,j])})
    with (a.out/"offwall_surrogate_metrics.csv").open("w",newline="") as s:
        w=csv.DictWriter(s,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    arrays={}; summary={}
    for (f,cid),values in saved.items():
        arrays[f"{f}__{cid}"]=np.stack(values); summary.setdefault(cid,{})[f]={}
        truth=next(c.targets[m] for c,m,_ in test if c.case_id==cid)
        for j,t in enumerate(TARGETS):
            errs=[nrmse(truth[:,j],v[:,j]) for v in values]
            summary[cid][f][t]={"mean_nrmse":float(np.mean(errs)),"std_nrmse":float(np.std(errs,ddof=1))}
    np.savez_compressed(a.out/"offwall_surrogate_predictions.npz",**arrays)
    (a.out/"offwall_surrogate_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(f"OFFWALL_SURROGATE_COMPLETE={a.out}")

if __name__=="__main__": main()
