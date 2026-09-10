# Changes for the fifth review round (September 2026)

## Correction announced in the previous round (Figure 5)
`scripts/04_var_granger_analysis.py` read `fevd.decomp` as (horizon, equation, shock); statsmodels
indexes it as (equation, horizon, shock). The corrected script and the regenerated
`results/var_granger/` files are included here. No table value in the article changed; only
Figure 5 and one sentence did.

## New analyses (revision 5)
- `scripts/09_r5_var_primary.py` - VAR(4) as the primary specification, with VAR(2) and VAR(6)
  as robustness checks. It covers lag-order criteria and diagnostics, joint Granger block tests
  (Holm and residual bootstrap, 1,999 replications), 20 pairwise Wald tests with
  Benjamini-Hochberg q-values, Cholesky FEVD under five orderings, generalized FEVD
  (Pesaran-Shin, row-normalized, raw row sums reported), and IRFs with asymptotic 95% bands.
  It first reproduces the previously published VAR(6) figures as a control.
  Outputs go to `results/r5/`, `figures/fig_04_irf_selected.png` and `figures/fig_05_fevd_ipc.png`.
- `scripts/10_r5_breaks_curation.py` - Chow and Bai-Perron-type break search with simulated
  sup-F critical values, break tests on CPI alone, corpus-composition series, and a fixed panel
  of persistent sources (fixed weights). The break tests and the pre/post-COVID contrasts are
  repeated on the fixed-panel emotion series. The script needs headline-level data, which are
  not distributed; its outputs, including the monthly fixed-panel series, are in `results/r5/`.
