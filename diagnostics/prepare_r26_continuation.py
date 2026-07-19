#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,shutil,subprocess,sys,zipfile
from pathlib import Path
import h5py,numpy as np

def find_one(root:Path,name:str,kind='file')->Path:
    items=[p for p in root.rglob(name) if (p.is_file() if kind=='file' else p.is_dir())]
    if len(items)!=1: raise RuntimeError(f'expected one {name}, found {items}')
    return items[0]

def h5_gate(path:Path,step:int):
    with h5py.File(path,'r') as h:
        assert int(np.asarray(h['nstep']).reshape(-1)[0])==step
        bad=[]
        def visit(name,obj):
            if isinstance(obj,h5py.Dataset) and obj.dtype.kind in 'fc' and not np.isfinite(np.asarray(obj[...])).all(): bad.append(name)
        h.visititems(visit); assert not bad,bad
        for k in ('ro','p','t'):
            a=np.asarray(h[k],float); assert a.min()>0 and np.isfinite(a).all()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source-artifact',type=Path,required=True); ap.add_argument('--seed-artifact',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    out=a.output.resolve(); shutil.rmtree(out,ignore_errors=True); out.mkdir(parents=True)
    source_zip=find_one(a.source_artifact,'ASTR_NONLINEAR_R13_JFM_CAVITY_U100_KN005.zip')
    src_unpack=out.parent/'source_unpack'; shutil.rmtree(src_unpack,ignore_errors=True); src_unpack.mkdir()
    with zipfile.ZipFile(source_zip) as z:z.extractall(src_unpack)
    shutil.copytree(src_unpack/'astr',out/'astr')
    shutil.rmtree(out/'astr/build',ignore_errors=True)
    subprocess.run([sys.executable,'astr_r26_stage1/patch_r26_stage1.py',str(out/'astr'),'--report',str(out/'stage1_report.json'),'--patch-output',str(out/'stage1.patch')],check=True)
    seed_zip=find_one(a.seed_artifact,'R26_V3_STRICT_FPE_FRESH_U100_KN005_N32_20K_DIAGNOSTIC.zip')
    seed_unpack=out.parent/'seed_unpack'; shutil.rmtree(seed_unpack,ignore_errors=True); seed_unpack.mkdir()
    with zipfile.ZipFile(seed_zip) as z:z.extractall(seed_unpack)
    seed=find_one(seed_unpack,'R26_V3_STRICT_FPE_FRESH_U100_KN005_N32_20K_DIAGNOSTIC','dir')
    verdict=json.loads((seed/'diagnostics/final_verdict.json').read_text())
    assert verdict['clean_strict_fpe_continuous_20k'] is True,verdict
    assert verdict['solver_return_code']==0 and verdict['runtime_marker_lines']==0
    shutil.copy2(seed/'patched_methodmoment.F90',out/'astr/src/methodmoment.F90')
    shutil.copy2(seed/'patched_mainloop.F90',out/'astr/src/mainloop.F90')
    shutil.copy2(seed/'patched_fludyna.F90',out/'astr/src/fludyna.F90')
    flags='-ffpe-trap=invalid -fbacktrace -g'
    subprocess.run(['cmake','-S',str(out/'astr'),'-B',str(out/'astr/build'),'-DCMAKE_BUILD_TYPE=Debug','-DBUILD_TESTING=OFF',f'-DCMAKE_Fortran_FLAGS={flags}'],check=True)
    subprocess.run(['cmake','--build',str(out/'astr/build'),'-j','2'],check=True)
    exe=out/'astr/build/bin/astr'; assert exe.is_file(); exe.chmod(0o755)
    case=out/'case'; (case/'outdat').mkdir(parents=True); (case/'bakup').mkdir()
    shutil.copytree(seed/'case_input',case/'datin')
    shutil.copytree(seed/'current_hdf5',case/'outdat',dirs_exist_ok=True)
    shutil.copytree(seed/'backup_hdf5',case/'bakup',dirs_exist_ok=True)
    shutil.copy2(seed/'case_metadata.json',case/'case_metadata.json')
    inp=next((case/'datin').glob('input.astr*')); lines=inp.read_text().splitlines()
    for i,l in enumerate(lines):
        if l.strip().startswith('# lrestar'): lines[i+1]='t'; break
    else: raise RuntimeError('lrestar not found')
    inp.write_text('\n'.join(lines)+'\n')
    meta=json.loads((case/'case_metadata.json').read_text()); meta.update({'initial_state':'restart from clean strict-FPE R26-v3 step 20000','restart_seed_run':29689454883,'restart_seed_step':20000,'restart_caveat':'R26 relaxed wall-memory arrays are not serialized and reset at each workflow-job restart'})
    (case/'case_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    h5_gate(case/'outdat/flowfield.h5',20000)
    for name in ('audits','diagnostics','patches'):
        if (seed/name).exists(): shutil.copytree(seed/name,out/name,dirs_exist_ok=True)
    (out/'continuation_provenance.json').write_text(json.dumps({'seed_run':29689454883,'seed_artifact':'R26_V3_STRICT_FPE_FRESH_U100_KN005_N32_20K_DIAGNOSTIC','seed_clean':True,'source_policy':'Stage-1 bc plus exact patched methodmoment/mainloop/fludyna from seed artifact'},indent=2)+'\n')
    print(json.dumps({'prepared':str(out),'seed_step':20000,'executable':str(exe)},indent=2))
if __name__=='__main__':main()
