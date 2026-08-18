# Machine Learning / ASTR moment-method research

Research workflows used to audit ASTR R13 and R26 cavity solvers and compare
them with kinetic references.  The R13/R26 material supports an active JFM
article and is therefore treated as protected scientific code.

## Repository map

- [`astr_r26_stage1/`](astr_r26_stage1/) — conservative short-gate audit of
  the existing ASTR R26 path.
- [`astr_r26_stage2/`](astr_r26_stage2/) — fixed continuation/convergence
  tooling.
- [`astr_r26_stage3/`](astr_r26_stage3/) — full-bulk closure patch record.
- [`.github/workflows/`](.github/workflows/) — reproducible R13/R26 build,
  continuation, and validation campaigns.

Several later diagnostics and held experiments remain on their original
research branches and pull requests.  They are evidence records, not an
instruction to replace the default implementation.

## Protected JFM code

Read [`JFM_R13_R26_PROTECTION.md`](JFM_R13_R26_PROTECTION.md) before changing
any R13/R26 source, patch, workflow, reference data, or numerical constants.
Repository cleanup must not rewrite, combine, or relabel those scientific
assets.  Documentation-only maintenance is intentionally kept separate.

## Status language

Use the status labels recorded by each stage.  In particular, an audited,
diagnostic, held, semi-linear, or short-gate result must not be described as a
fully validated nonlinear R13/R26 result.

## License and citation

No repository-wide license or citation metadata is declared yet.  These must
be chosen only after the article authorship and release scope are confirmed.
