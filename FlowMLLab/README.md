# FlowMLLab

FlowMLLab is an installable orchestration and validation layer for reproducible
computational fluid dynamics (CFD), scientific machine learning (SciML), and
Direct Simulation Monte Carlo (DSMC) experiments. Version 1.0.0 exposes a
stable command-line interface while retaining the audited numerical evidence in
the adjacent `MIE690A_AI_in_Fluid_Mechanics/` tree.

## Install

```bash
python -m pip install -e ./FlowMLLab[test]
```

## Validate and reproduce

```bash
flowmllab validate
flowmllab reproduce continuum
flowmllab reproduce pod-deeponet
flowmllab reproduce dsmc
```

Pass `--root /path/to/Machine_Learning` when the checkout cannot be discovered
from the working directory. Add `--recompute` to the continuum command to
regenerate the two retained Reynolds-number 1000 CFD solutions before
rebuilding the pressure-validation figure.

The CLI calls versioned programs; it does not depend on hidden notebook state.
Each workflow writes both figures and machine-readable comma-separated value
(CSV), JavaScript Object Notation (JSON), or NumPy archive (NPZ) evidence.

## Software layers

- `flowmllab.continuum`: continuum solver and pressure-validation orchestration;
- `flowmllab.surrogate`: proper orthogonal decomposition (POD) and DeepONet study;
- `flowmllab.kinetic`: DSMC wall-pressure validation;
- `flowmllab.validation`: release, data-contract, and numerical-anchor checks;
- `flowmllab.cli`: stable public command-line interface.

Tutorial notebooks remain examples of the public software and are not required
to invoke the CLI. The project is distributed under the MIT License.
