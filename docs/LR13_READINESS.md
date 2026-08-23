# Linearized R13 readiness contract

## Current scientific verdict

The audited ASTR source implements a **nonlinear R13** path named `r13`.  It
does not contain a separate, formula-complete linearized R13 (`lr13`) model.
The 2,000-step smoke and 20,000-step diagnostic are therefore deliberately
blocked.  Running the existing executable with two nonlinear source blocks
disabled would not make it LR13.

The CFD discretization and domain-decode fixes from the audited ASTR solver may
be retained.  What cannot be retained under an LR13 label is the nonlinear
mathematical model.

## Gates required before any LR13 run

1. Introduce an explicit `lr13` input label, state layout, initialization,
   restart metadata, output metadata, and dispatch path.  It must never alias
   `r13` silently.
2. Apply one documented `O(epsilon)` perturbation convention around a uniform
   equilibrium to **all five** primary balances and to all stress/heat-flux
   balances.  Terms of order `epsilon^2` must be absent, including moment
   advection products.
3. Derive every R13 regularizing closure under the same convention and evaluate
   transport/relaxation coefficients at the declared equilibrium state.  Local
   products such as `mu*T/p` need an explicit first-order derivation rather than
   an informal toggle.
4. Choose and implement one complete, authoritative linear boundary model.
   Either derive the first-order limit of the classical Maxwell/Rana wall
   relations consistently, or implement the full Onsager LR13 system.  These
   must not be mixed opportunistically.
5. Derive and document the cavity-corner system.  A single-owner face rule is a
   numerical convention, not a paper-exact corner condition.  Equation (20) of
   the Rana cavity paper is a coupled steady matrix solve, not a local corner
   wall formula.
6. Add a symbolic first-order oracle that perturbs every retained equation,
   checks cancellation of the equilibrium residual, and verifies all retained
   coefficients and signs against the chosen source.
7. Only after gates 1-6 pass: strict-FPE compile and a fresh-equilibrium 2k
   smoke.  Only after the smoke passes: the matched `U=100 m/s`, `Kn=0.05`,
   `N=32`, `dt=2.5e-5`, continuous 20k diagnostic with 1k checkpoints.
8. Scientific acceptance still requires finite/positive HDF5 state, no runtime
   floating-point markers, sustained convergence across at least three
   checkpoints, and all four predeclared anti-Fourier metrics.  A fixed-step
   completion alone is not convergence.

## Why the current source fails the contract

The automated audit records line-level evidence for each of these facts:

- there is no `lr13` token or dispatch path;
- the base solver retains nonlinear compressible fluxes;
- moment convection contains products of moment variables and velocity;
- unconditional nonlinear source blocks remain active;
- regularizer coefficients depend on the local state;
- nonlinear wall products such as squared slip velocity remain;
- no LR13 corner derivation or symbolic `O(epsilon)` oracle exists.

## Authoritative equation sources

- Rana, Torrilhon & Struchtrup (2013), [nonlinear R13 equations, closures, and
  wall relations](https://www.engr.uvic.ca/~struchtr/2013_JCP_Lidcavity.pdf).
- Lin et al. (2025), [time-dependent R13 with Onsager boundary conditions in
  the linear regime](https://doi.org/10.1017/jfm.2025.215).
- Cai, Torrilhon & Yang (2024), [well-posed linear regularized 13-moment
  equations](https://doi.org/10.1137/23M1556472).

This audit is a safety/readiness artifact, not an LR13 implementation and not a
physics result.
