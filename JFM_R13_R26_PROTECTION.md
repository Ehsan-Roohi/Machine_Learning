# R13/R26 JFM code-protection policy

The R13 and R26 implementations, patches, workflows, checkpoints, and
reference summaries in this repository are scientific assets for an active
JFM article.  They are **read-only during repository cleanup**.

## Protected scope

- `astr_r26_stage*/**`
- `.github/workflows/*r13*` and `.github/workflows/*r26*`
- R13/R26 solver patches, generated cases, validators, checkpoints, reference
  summaries, and restart/continuation scripts on research branches
- numerical constants, boundary conditions, nondimensionalization, closure
  terms, convergence thresholds, and stage labels used by those assets

## Rules

1. Do not reformat, rename, move, squash, delete, or opportunistically merge
   protected files during general cleanup.
2. Do not combine competing scientific branches merely to reduce the pull
   request count.
3. A scientific change requires its own narrowly scoped pull request with the
   exact baseline commit, equation/term justification, reproducible command,
   and numerical comparison against the previous result.
4. Preserve negative, held, and failed results; do not relabel them as passed.
5. Keep large HPC outputs external unless a small artifact is necessary to
   audit a published numerical claim.

This policy protects code integrity; it does not assert that every historical
branch is scientifically accepted.
