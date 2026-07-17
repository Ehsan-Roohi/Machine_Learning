#!/bin/bash
set -euxo pipefail
cd /work
rm -rf src output
mkdir -p output
git clone --depth 1 https://github.com/lamBOOO/fenicsR13.git src
cd src
python3 -m pip install --user -e .
export PATH="$HOME/.local/bin:$PATH"
cd examples/lid_driven_cavity
geoToH5 lid.geo lid5.h5 "-setnumber p 5"
geoToH5 lid.geo lid6.h5 "-setnumber p 6"
cp /work/input_jfm_kn005_u100.yml .
PYTHONPATH=/work/src python3 /work/run_and_postprocess.py 2>&1 | tee /work/output/run.log
sed -i 's/Walls use the repository-standard chi_tilde=1.0 and theta_w=1.0./Fully diffuse Maxwell walls use chi=1, mapped to chi_tilde=sqrt(2\/pi)=0.797884560803; theta_w=1.0./' /work/output/RUN_NOTES.txt
cp /work/input_jfm_kn005_u100.yml /work/output/
test -s /work/output/run.log
test -s /work/output/summary_all_meshes.json
