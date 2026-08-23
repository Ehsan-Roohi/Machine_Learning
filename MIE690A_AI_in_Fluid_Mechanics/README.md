# MIE 690A - AI in Fluid Mechanics

Public computational materials from the six-week graduate course **MIE 690A: AI in Fluid Mechanics**, University of Massachusetts Amherst, Summer 2026.

The course treats scientific machine learning as an end-to-end computational workflow: generate or audit numerical data, define a physically meaningful learning task, separate development/validation/blind cases, compare against simple non-neural baselines, evaluate physical fidelity, and document failure modes and reproducibility.

## Six final project-track notebooks

1. `P1_Re_Generalization.ipynb` - Reynolds-number interpolation/extrapolation and failure-boundary analysis.
2. `P2_Physics_Guided_DNN.ipynb` - physics-guided coordinate DNNs, wall weighting, and divergence-aware diagnostics.
3. `P3_POD_Study.ipynb` - POD/reduced-order learning, rank sensitivity, modal-coefficient prediction, and structure preservation.
4. `P4_Uncertainty_Study.ipynb` - uncertainty/data-sufficiency analysis and robustness checks.
5. `P5_Rarefied_Cavity.ipynb` - rarefied cavity modeling, particle sampling, and kinetic-field validation.
6. `P6_FP_Cavity_Closure.ipynb` - neural closure acceleration for a Fokker-Planck particle solver and closed-loop fidelity testing.

`common/P0_Project_Setup.ipynb` is provided as a shared setup/audit notebook and is not counted as a project track.

## Common physical benchmark

A major course thread is the lid-driven cavity. Continuum cavity fields are generated with a streamfunction-vorticity finite-difference solver and checked against the classical Ghia et al. benchmark where applicable. The same problem is then revisited through data-driven surrogates, POD/reduced-order representations, particle sampling, DSMC-style modeling, and Fokker-Planck closure acceleration.

## Reproducibility principles

- Split by complete physical cases, not random grid points, when case-wise generalization is the scientific question.
- Fit scalers and choose hyperparameters using development/training data only.
- Freeze model-selection rules before opening blind cases.
- Compare neural models with transparent baselines such as interpolation or linear coefficient prediction.
- Report physical diagnostics in addition to aggregate error norms.
- Negative results and failure modes are valid outcomes when the experimental protocol is clean.

## Repository structure

- `notebooks/` - the six final project-track notebooks.
- `common/P0_Project_Setup.ipynb` - the shared setup/audit notebook.
- The first public commit focuses on the six final project notebooks and the shared setup notebook. The helper modules, reference datasets, and Track-6 solver scripts remain in the archived course package while licensing and release packaging are checked; they will be added before the archival paper release.

## Associated manuscript

A companion manuscript is being prepared under the working title:

**From CFD to Scientific Machine Learning: A Reproducible Jupyter Curriculum for AI in Fluid Mechanics**

The manuscript explains the six-week course design, lecture sequence, numerical and ML workflow, physical-validation philosophy, representative student projects, and how the course grew out of the instructor's own research transition into scientific machine learning for computational fluid dynamics.

## Student work and permissions

Student project results shown in the manuscript are attributed by name and will be included in the final submitted version only after explicit permission is obtained from the students. Student submissions are not included in this public repository.

## License

Copyright (c) 2026 Ehsan Roohi. Public release for inspection and educational use. A formal code/content license will be selected before the final archival release associated with the paper.

## Contact

Ehsan Roohi  
Mechanical and Industrial Engineering  
University of Massachusetts Amherst  
roohie@umass.edu
