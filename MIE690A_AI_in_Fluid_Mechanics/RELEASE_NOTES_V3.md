# Release v3: article-aligned executable validation

This release closes the gap between the teaching notebooks and the manuscript figures.

- All 16 notebooks carry an explicit article-output contract.
- The Week-1 cavity notebook begins with executed Ghia velocity and Botella--Peyret pressure validation, embeds the current figures/metrics, and provides an optional exact rerun of the $Re=1000$ grids.
- The Week-3 DSMC notebook begins with direct validation of our HS--NTC solver against Mohammadzadeh DSMC wall pressure. It embeds the current comparison/residual figure and provides the exact four-run recomputation protocol.
- `common/article_validation.py` is the shared figure builder for article Figures 2, 5, and 8 in the compiled v3 manuscript.
- `ARTICLE_FIGURE_MAP.md` maps all paper-facing evidence to its owner notebook, retained inputs, outputs, and rerun command.
- `qa/validate_course_release.py` checks notebook synchronization, figure assets, numerical thresholds, lecture PDFs, POD-DeepONet evidence, and the fixed dataset hash.

Recorded validation values:

- Ghia at $Re=400$: $E_u=5.239\%$, $E_v=15.426\%$ for the transparent teaching grid.
- Botella--Peyret pressure at $Re=1000$, $129^2$: vertical $E=5.538\%$, horizontal $E=6.575\%$.
- Mohammadzadeh wall pressure: $60^2$ DSMC relative $L_2=0.772\%$, RMSE $=0.0078p_0$, $40^2\rightarrow60^2$ change $=0.870\%$.
