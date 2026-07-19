# Maxwell R26 full-bulk closure v3: equation-level audit

## Scope

This patch is applied after the existing Stage-1 and nonlinear-source-v1 patches. It preserves the Maxwell-molecule coefficients, the STF projection routines, the quotient gradients, and the corrected R13 `div(q)` expression, while correcting three equation-level discrepancies found by a direct term-by-term comparison with Gu & Emerson (JFM 636, 2009), equations (24)–(26).

## Corrections relative to provisional v2

1. **Equation (25), velocity-divergence contribution.** The bracket contains both `+8 m_ijk div(u)` and `-6 m_ijk div(u)`. The provisional code represented the `-6` term as an oriented gradient contraction. v3 restores the scalar-divergence term exactly.
2. **Equation (25), collision-product signs.** The leading outer minus multiplies both the `Y2 sigma*m` and `Y3 q*sigma` products. v3 assigns a negative sign to both terms.
3. **Equation (26), `R_ij grad(sigma_jk)` term.** The paper contains `R_ij * d(sigma_jk)/dx_k` only. The provisional code added a pressure-gradient contribution. v3 removes that unsupported addition.

The workflow includes explicit static gates for all three corrections, in addition to the existing coefficient, closure-activation, quotient-gradient, `div(q)`, and STF trace tests.

## Model label

**Maxwell R26 full-bulk-closure v3, term-by-term audited, with relaxed WBC**

## Remaining limitations

- Appendix-C wall boundary formulas are still imposed through temporal relaxation rather than exact algebraic enforcement.
- The 50k restart seed predates v3. The 50k-to-100k continuation is a diagnostic relaxation study; a publication-grade result still requires a from-equilibrium v3 run.
- Grid convergence, restart independence, and the planned Knudsen-number sweep remain separate validation tasks.
