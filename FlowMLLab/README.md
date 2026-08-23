# FlowMLLab

The installable FlowMLLab framework, tutorials, fixed evidence, and validation
programs are maintained in
[`MIE690A_AI_in_Fluid_Mechanics/`](../MIE690A_AI_in_Fluid_Mechanics/).

```bash
cd MIE690A_AI_in_Fluid_Mechanics
python -m pip install -e ".[test]"
flowmllab smoke --root .
flowmllab qa --root .
flowmllab figures all --root .
```

This stable top-level entry point makes the software identity explicit while
preserving existing links to the complete tutorial release.
