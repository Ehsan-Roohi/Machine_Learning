#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,shutil,subprocess,sys,zipfile
from pathlib import Path
import h5py,numpy as np


def find_one(root:Path,name:str,kind='file')->Path:
    items=[p for p in root.rglob(name) if (p.is_file() if kind=='file' else p.is_dir())]
    if len(items)!=1:
        raise RuntimeError(f'expected one {name}, found {items}')
    return items[0]


def h5_gate(path:Path,step:int):
    with h5py.File(path,'r') as h:
        assert int(np.asarray(h['nstep']).reshape(-1)[0])==step
        bad=[]
        def visit(name,obj):
            if isinstance(obj,h5py.Dataset) and obj.dtype.kind in 'fc' and not np.isfinite(np.asarray(obj[...])).all():
                bad.append(name)
        h.visititems(visit)
        assert not bad,bad
        for k in ('ro','p','t'):
            a=np.asarray(h[k],float)
            assert a.min()>0 and np.isfinite(a).all()


def extract_inner(artifact_root:Path,inner_name:str,destination:Path)->Path:
    inner=find_one(artifact_root,inner_name)
    shutil.rmtree(destination,ignore_errors=True)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(inner) as z:
        z.extractall(destination)
    return destination


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-artifact',type=Path,required=True)
    ap.add_argument('--base-artifact',type=Path,required=True)
    ap.add_argument('--endpoint-artifact',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()

    out=a.output.resolve()
    shutil.rmtree(out,ignore_errors=True)
    out.mkdir(parents=True)

    source_zip=find_one(a.source_artifact,'ASTR_NONLINEAR_R13_JFM_CAVITY_U100_KN005.zip')
    src_unpack=out.parent/'source_unpack'
    shutil.rmtree(src_unpack,ignore_errors=True)
    src_unpack.mkdir()
    with zipfile.ZipFile(source_zip) as z:
        z.extractall(src_unpack)
    shutil.copytree(src_unpack/'astr',out/'astr')
    shutil.rmtree(out/'astr/build',ignore_errors=True)

    # The original diagnostic contains the exact case input, metadata, audits,
    # and patched source provenance.  The recovered endpoint contains the actual
    # clean nstep=20000 state produced by PR 37.
    base_unpack=extract_inner(
        a.base_artifact,
        'R26_V3_STRICT_FPE_FRESH_U100_KN005_N32_20K_DIAGNOSTIC.zip',
        out.parent/'base_unpack',
    )
    base=find_one(base_unpack,'R26_V3_STRICT_FPE_FRESH_U100_KN005_N32_20K_DIAGNOSTIC','dir')

    endpoint_unpack=extract_inner(
        a.endpoint_artifact,
        'R26_V3_RESTART_11K_TO_20K_DIAGNOSTIC.zip',
        out.parent/'endpoint_unpack',
    )
    endpoint=find_one(endpoint_unpack,'R26_V3_RESTART_11K_TO_20K_DIAGNOSTIC','dir')
    verdict=json.loads((endpoint/'diagnostics/final_verdict.json').read_text())
    assert verdict['completed'] is True,verdict
    assert int(verdict['actual'])==20000 and int(verdict['solver_return_code'])==0,verdict
    assert verdict['finite'] is True and verdict['positive'] is True,verdict
    assert all(int(v)==0 for v in verdict['markers'].values()),verdict

    # Use the exact patched sources stored by the recovered endpoint.
    shutil.copy2(endpoint/'methodmoment.F90',out/'astr/src/methodmoment.F90')
    shutil.copy2(endpoint/'mainloop.F90',out/'astr/src/mainloop.F90')
    shutil.copy2(endpoint/'fludyna.F90',out/'astr/src/fludyna.F90')

    flags='-ffpe-trap=invalid -fbacktrace -g'
    subprocess.run([
        'cmake','-S',str(out/'astr'),'-B',str(out/'astr/build'),
        '-DCMAKE_BUILD_TYPE=Debug','-DBUILD_TESTING=OFF',
        f'-DCMAKE_Fortran_FLAGS={flags}',
    ],check=True)
    subprocess.run(['cmake','--build',str(out/'astr/build'),'-j','2'],check=True)
    exe=out/'astr/build/bin/astr'
    assert exe.is_file()
    exe.chmod(0o755)

    case=out/'case'
    (case/'outdat').mkdir(parents=True)
    (case/'bakup').mkdir()
    shutil.copytree(base/'case_input',case/'datin')
    shutil.copytree(endpoint/'outdat',case/'outdat',dirs_exist_ok=True)
    shutil.copytree(endpoint/'bakup',case/'bakup',dirs_exist_ok=True)
    shutil.copy2(base/'case_metadata.json',case/'case_metadata.json')

    inp=next((case/'datin').glob('input.astr*'))
    lines=inp.read_text().splitlines()
    for i,l in enumerate(lines):
        if l.strip().startswith('# lrestar'):
            lines[i+1]='t'
            break
    else:
        raise RuntimeError('lrestar not found')
    inp.write_text('\n'.join(lines)+'\n')

    meta=json.loads((case/'case_metadata.json').read_text())
    meta.update({
        'initial_state':'restart from recovered clean R26-v3 step 20000',
        'restart_seed_run':29707336380,
        'restart_seed_step':20000,
        'restart_caveat':'legacy R26 relaxed wall-memory arrays are not serialized and reset at every workflow-job restart',
    })
    (case/'case_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')

    h5_gate(case/'outdat/flowfield.h5',20000)
    backup_step=None
    with h5py.File(case/'bakup/flowfield.h5','r') as h:
        backup_step=int(np.asarray(h['nstep']).reshape(-1)[0])
    assert 19000 <= backup_step <= 20000,backup_step
    h5_gate(case/'bakup/flowfield.h5',backup_step)

    for name in ('audits','patches'):
        if (base/name).exists():
            shutil.copytree(base/name,out/name,dirs_exist_ok=True)
    shutil.copytree(endpoint/'diagnostics',out/'diagnostics',dirs_exist_ok=True)

    provenance={
        'base_run':29689454883,
        'endpoint_run':29707336380,
        'endpoint_artifact':'R26_V3_RESTART_11K_TO_20K_DIAGNOSTIC',
        'seed_step':20000,
        'backup_step':backup_step,
        'seed_clean':True,
        'source_policy':'exact patched methodmoment/mainloop/fludyna stored by recovered endpoint',
        'scientific_caveat':'diagnostic segmented continuation because legacy R26 wall-memory arrays reset at each job boundary',
    }
    (out/'continuation_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'prepared':str(out),'seed_step':20000,'backup_step':backup_step,'executable':str(exe)},indent=2))


if __name__=='__main__':
    main()
