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

## ARIMAX (scripts/12_r5_arimax_multiplicity.py)
- Benjamini-Hochberg and Holm corrections over the 88 emotion coefficients and the 44 dummy
  coefficients of the 22 ARIMAX models, computed from the published p-values.
- The RMSE and MAE reported by 03_arimax_models.py included the first residual of the diffuse
  initialization, which for d = 1 equals the level of the series (e.g. 93.45 for CPI). They are
  recomputed without the initialization observations (`results/r5/arimax_rmse_corrected.csv`),
  and Figure 3 is regenerated without that residual (`figures/fig_03_arimax_global.png`).
- `scripts/11_r5_negativity_index.py` - complementary test of the negativity hypothesis: the
  composite negativity index (anger + fear + sadness, as in Fig. 6) in a three-variable VAR
  [CPI, negativity, joy], with Granger tests in both directions (bootstrap, Holm), a lag sweep
  from 1 to 8, the cumulative response of negativity to a CPI shock, and the same tests on the
  fixed panel of sources. It runs from the published monthly data in `data/` and `results/r5/`.
- `scripts/r5_common.py` - functions shared by scripts 09 and 11 (bootstrap Granger test, generalized FEVD).

## Repository location
This repository replaces https://github.com/Jorgejosezamora/sentiment-news-spain-2017-2023, whose
account is no longer accessible. The earlier repository remains online in its March 2026 state but
cannot be updated; this one carries its full commit history plus all later corrections.
