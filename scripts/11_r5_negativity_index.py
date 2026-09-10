"""
R5 (Access-2026-17671): contraste complementario de la hipotesis de negatividad.

El indice compuesto de negatividad (Ira + Miedo + Tristeza, el mismo de la Fig. 6) entra en
un VAR de tres variables [CPI, Negativity, Joy], en primeras diferencias como el VAR principal.
- Orden: la misma regla que el VAR principal (menor orden con residuos blancos a p + 5 rezagos).
- Granger en ambas direcciones: Wald F, p-valor bootstrap bajo H0 y Holm sobre las dos direcciones.
- Barrido de rezagos 1-8 para ver la sensibilidad al orden.
- Signo: respuesta acumulada ortogonalizada de la negatividad a un shock del IPC, con bandas.
- Panel fijo de fuentes (results/fixed_composition_emotions.csv, de r5_breaks.py).
- Control desagregado: las tres emociones negativas por separado dentro del VAR(4) de cinco variables.
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.stats.multitest import multipletests

from r5_common import bootstrap_granger, B

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'econometric_dataset_global.csv'
RES = ROOT / 'results' / 'r5'
FIXED = RES / 'fixed_composition_emotions.csv'
MAX_LAGS = 8
H = 12
NEG_ES = ['Ira', 'Miedo', 'Tristeza']

df = pd.read_csv(DATA, parse_dates=['Fecha'], encoding='utf-8-sig').set_index('Fecha').sort_index()
df.index = pd.DatetimeIndex(df.index).to_period('M').to_timestamp()
ipc_col = next(c for c in df.columns if c.startswith('IPC_ECOICOP') and 'General' in c)
assert np.allclose(df['Negatividad'], df[NEG_ES].sum(axis=1)), "Negatividad no es Ira + Miedo + Tristeza"

fixed = pd.read_csv(FIXED, index_col=0, parse_dates=True)
series = {
    'published': pd.DataFrame({'CPI': df[ipc_col], 'Negativity': df[NEG_ES].sum(axis=1), 'Joy': df['Alegria']}),
    'fixed_panel': pd.DataFrame({'CPI': df[ipc_col], 'Negativity': fixed[['anger', 'fear', 'sadness']].sum(axis=1),
                                 'Joy': fixed['joy']}),
}


def diffed(levels):
    X = levels.dropna().diff().dropna()
    X.index.freq = 'MS'
    return X


summary = {}
sweep_rows = []
for label, lv in series.items():
    X = diffed(lv)
    for p in range(1, MAX_LAGS + 1):
        r = VAR(X).fit(p)
        sweep_rows.append({
            'series': label, 'lag': p, 'portmanteau_p_plus5': r.test_whiteness(nlags=p + 5).pvalue,
            'neg_to_cpi_p': r.test_causality('CPI', ['Negativity'], kind='f').pvalue,
            'cpi_to_neg_p': r.test_causality('Negativity', ['CPI'], kind='f').pvalue,
            'joy_to_cpi_p': r.test_causality('CPI', ['Joy'], kind='f').pvalue})
sweep = pd.DataFrame(sweep_rows)
sweep.round(4).to_csv(RES / 'negativity_lag_sweep.csv', index=False)
print(sweep.round(3).to_string())

pub = sweep[sweep['series'] == 'published']
p_sel = int(pub.loc[pub['portmanteau_p_plus5'] > 0.05, 'lag'].min())
sel = VAR(diffed(series['published'])).select_order(maxlags=MAX_LAGS).selected_orders
print(f"\nOrden por la regla de residuos blancos: {p_sel} | criterios: {sel}")

for label, lv in series.items():
    X = diffed(lv)
    out = {}
    for p in sorted({p_sel, 2, 6}):
        r = VAR(X).fit(p)
        n2c = r.test_causality('CPI', ['Negativity'], kind='f')
        c2n = r.test_causality('Negativity', ['CPI'], kind='f')
        row = {'portmanteau_p_plus5': float(r.test_whiteness(nlags=p + 5).pvalue),
               'neg_to_cpi': {'F': float(n2c.test_statistic), 'df': [int(d) for d in n2c.df], 'p': float(n2c.pvalue)},
               'cpi_to_neg': {'F': float(c2n.test_statistic), 'df': [int(d) for d in c2n.df], 'p': float(c2n.pvalue)}}
        if p == p_sel:
            print(f"   bootstrap ({B} replicas) {label} VAR({p})...")
            row['neg_to_cpi']['p_boot'] = float(bootstrap_granger(X, p, ['CPI'], ['Negativity']))
            row['cpi_to_neg']['p_boot'] = float(bootstrap_granger(X, p, ['Negativity'], ['CPI']))
            ha = multipletests([row['neg_to_cpi']['p'], row['cpi_to_neg']['p']], method='holm')[1]
            hb = multipletests([row['neg_to_cpi']['p_boot'], row['cpi_to_neg']['p_boot']], method='holm')[1]
            row['neg_to_cpi'].update(p_holm=float(ha[0]), p_boot_holm=float(hb[0]))
            row['cpi_to_neg'].update(p_holm=float(ha[1]), p_boot_holm=float(hb[1]))
            irf = r.irf(H)
            cum = irf.orth_cum_effects[:, 1, 0]
            se = irf.cum_effect_stderr(orth=True)[:, 1, 0]
            neg_sd = float(X['Negativity'].std())
            row['cum_irf_cpi_to_neg'] = {
                str(h): {'effect': float(cum[h]), 'lower': float(cum[h] - 1.96 * se[h]),
                         'upper': float(cum[h] + 1.96 * se[h]), 'in_sd_of_monthly_change': float(cum[h] / neg_sd)}
                for h in [1, 2, 3, 6, 12]}
        out[f'VAR({p})'] = row
        print(f"{label:12s} VAR({p}) Portm={row['portmanteau_p_plus5']:.3f} | neg->CPI p={row['neg_to_cpi']['p']:.3f} "
              f"| CPI->neg p={row['cpi_to_neg']['p']:.3f}"
              + (f" boot={row['cpi_to_neg']['p_boot']:.3f} Holm={row['cpi_to_neg']['p_holm']:.3f} "
                 f"boot-Holm={row['cpi_to_neg']['p_boot_holm']:.3f} | neg->CPI boot={row['neg_to_cpi']['p_boot']:.3f}"
                 if p == p_sel else ""))
        if 'cum_irf_cpi_to_neg' in row:
            for h, v in row['cum_irf_cpi_to_neg'].items():
                print(f"      respuesta acumulada de la negatividad a un shock del IPC, h={h}: {v['effect']:+.5f} "
                      f"[{v['lower']:+.5f}, {v['upper']:+.5f}] ({v['in_sd_of_monthly_change']:+.2f} d.t.)")
    summary[label] = out

# Control desagregado dentro del VAR de cinco variables
L5 = pd.DataFrame({'CPI': df[ipc_col], 'Anger': df['Ira'], 'Fear': df['Miedo'], 'Sadness': df['Tristeza'],
                   'Joy': df['Alegria']})
X5 = diffed(L5)
disagg = {}
for p in [2, p_sel, 6]:
    r = VAR(X5).fit(p)
    disagg[f'VAR({p})'] = {
        'three_negative_to_cpi_p': float(r.test_causality('CPI', ['Anger', 'Fear', 'Sadness'], kind='f').pvalue),
        'cpi_to_three_negative_p': float(r.test_causality(['Anger', 'Fear', 'Sadness'], ['CPI'], kind='f').pvalue)}
print("\nDesagregado (VAR de 5 variables):", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in disagg.items()})

summary.update(selected_order=p_sel, criteria={k.upper(): int(v) for k, v in sel.items()},
               lag_sweep_published={
                   'cpi_to_neg_p_range': [float(pub['cpi_to_neg_p'].min()), float(pub['cpi_to_neg_p'].max())],
                   'neg_to_cpi_p_range': [float(pub['neg_to_cpi_p'].min()), float(pub['neg_to_cpi_p'].max())],
                   'cpi_to_neg_significant_lags': [int(x) for x in pub.loc[pub['cpi_to_neg_p'] < 0.05, 'lag']],
                   'neg_to_cpi_significant_lags': [int(x) for x in pub.loc[pub['neg_to_cpi_p'] < 0.05, 'lag']]},
               disaggregated_five_var=disagg)
with open(RES / 'r5_negativity_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)
print("\nResumen escrito en", RES / 'r5_negativity_summary.json')
