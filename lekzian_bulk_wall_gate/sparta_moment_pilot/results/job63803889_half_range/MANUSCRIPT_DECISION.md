# Manuscript decision after cross-geometry half-range gate

## Result

The old information-horizon manuscript should not be resubmitted unchanged.
The new evidence supports a different paper centered on finite full-range
moment insufficiency and incident half-range sufficiency.

- The blind, equal-capacity interpolation gate still fails as a transferable
  wall closure.  Corrected gas-side sampling gives Cp NRMSE values of 3.20%
  (parameter baseline), 8.87% (S0), 6.11% (S1), and 5.84% (S2).  Thus stress
  moments materially improve pressure relative to primitives, but do not beat
  the parameter baseline.
- For signed Cf the corresponding errors are 26.37%, 35.66%, 43.38%, and
  45.76%.  Full-range stress and heat-flux moments do not provide a robust
  cross-Kn closure for shear.
- S3 uses incident pre-collision velocities plus the known 300 K diffuse-wall
  kernel.  Across ISO, BWD, and FWD, its concurrent Cp NRMSE is 0.15--0.28%
  and signed-Cf NRMSE is 1.30--7.35%.  The pre/post impulse control gives
  approximately 0.056% Cp error in every case, validating surface IDs,
  normalization, and geometry-aware normals.
- Against a separate 40,000-step wall reference, BWD remains accurate.  FWD
  signed-Cf differences are 16.65--25.09%, but the independent DSMC block SEM
  is already 13.93--20.70%; this is a precision limitation, not evidence that
  the concurrent S3 identity failed.
- A constructive velocity-space example supplies two non-negative
  distributions with every full-range monomial moment through degree three
  identical to machine precision, while incident pressure and shear differ.
  This proves pointwise non-uniqueness for the finite S0/S1/S2 hierarchy.

## Relation to the PoF rejection

The redesign directly removes the three weaknesses identified by the referee:

1. The model-dependent information horizon is no longer the principal claim.
2. Parameter, S0, S1, and S2 use the same model class, width, seeds, training
   budget, and padded input dimension in a blind intermediate-Kn test.
3. Absolute wall-load accuracy is reported and S3 is validated against the
   direct DSMC impulse control, not against a full-domain neural reference.

The likely new referee objection is that wall flux being a half-range kinetic
functional is known.  Novelty must therefore rest on the combination of a
fair blind hierarchy, corrected overhang geometry, cross-geometry quantitative
gates, and a constructive non-uniqueness result—not on S3 accuracy alone.

## Remaining run

One targeted precision run is recommended before manuscript submission:
continue only FWD at Kn=0.2 and 0.4 for 50,000 sampled steps in ten 5,000-step
blocks.  The existing 5,000-step data remain untouched by using the label
`half_range_fwd_extended`.  No additional ISO or BWD DSMC is justified.

The submission gate is:

- independent-window Cp NRMSE below 3%;
- independent-window signed-Cf NRMSE below 15%, or a difference statistically
  indistinguishable from zero using the combined DSMC/S3 block uncertainty;
- full pre/post Cp control below 0.2%;
- corrected gas-facing surface normals retained.
