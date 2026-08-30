# Manuscript revision v12

This directory records the evidence used for the Physics of Fluids revision of
the bulk-to-wall observability study.  The revision preserves the manuscript's
section structure while changing its central claim from a universal spatial
radius to a quantity-dependent kinetic-information hierarchy.

## Evidence hierarchy

- `S0`: primitive bulk fields.
- `S1`: `S0` plus the full momentum-flux tensor.
- `S2`: `S1` plus the heat-flux vector.
- `S_HR`: the incident half-range molecular state together with the prescribed
  diffuse wall-scattering kernel.  This replaces the preliminary label
  "S3 incident" because it is distribution-level information, not merely a
  third finite full-range moment set.

The blind, equal-capacity intermediate-Knudsen-number test shows that adding
`P_ij` reduces the pressure error relative to primitives, but no finite
full-range state improves transferable signed shear.  A constructive positive-
distribution example proves why: two distributions can share every two-
dimensional full-range monomial moment through total degree three while having
different incident pressure flux and opposite incident shear flux.

The co-temporal half-range reconstruction gives pressure NRMSE of 0.15--0.28%
and signed-shear NRMSE of 1.30--7.35% over ISO, BWD and FWD geometries.  The
independent-window test retains the pressure conclusion (1.68--2.85% NRMSE).
The noisier FWD shear cases are reported with DSMC sampling uncertainty and are
not used to make a stronger claim than the evidence supports.

## Figure policy

All revision figures are landscape, use embedded serif fonts, omit a redundant
figure-level title, and place a compact common legend outside the data panels.
For wall profiles, raw binned values remain visible as markers.  A light
Savitzky--Golay curve is used only for display, independently on each physical
face; smoothing never crosses the apex.  Every reported metric is evaluated on
the unsmoothed arrays.

## Files

- `main.tex`: revised full manuscript.
- `response_to_reviewer.tex`: point-by-point response.
- `cover_letter_to_editor.tex`: resubmission letter.
- `figures/`: central vector figures used in the revision.

The extended FWD independent-window continuation (job array `63805923`) was
still active when this evidence freeze was made, so its outcome is intentionally
excluded from the manuscript claims.
