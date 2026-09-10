"""
R5 (Access-2026-17671): sensibilidad de la cointegracion al numero de rezagos y tests de
Granger de Toda-Yamamoto, validos con independencia del orden de integracion y de la
cointegracion.

- Engle-Granger (residuos, rezago automatico) y Johansen (traza, r = 0) para CPI y Alegria con
  k = 1..8 diferencias retardadas. El 10.9613 publicado corresponde a k = 5 (VAR(6) en niveles).
- Toda-Yamamoto: VAR en niveles con p + 1 rezagos y Wald sobre los p primeros, para p = 2, 4, 6,
  en el sistema de cinco variables y en el de negatividad.
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'econometric_dataset_global.csv'
RES = ROOT / 'results' / 'r5'
NAMES = ['CPI', 'Anger', 'Fear', 'Sadness', 'Joy']
EMO = NAMES[1:]

df = pd.read_csv(DATA, parse_dates=['Fecha'], encoding='utf-8-sig').set_index('Fecha').sort_index()
df.index = pd.DatetimeIndex(df.index).to_period('M').to_timestamp()
ipc_col = next(c for c in df.columns if c.startswith('IPC_ECOICOP') and 'General' in c)
L = df[[ipc_col, 'Ira', 'Miedo', 'Tristeza', 'Alegria']].dropna()
L.columns = NAMES
L.index.freq = 'MS'

# ─── 1. Cointegracion CPI-Joy ───
eg = coint(L['CPI'], L['Joy'])
rows = []
for k in range(1, 9):
    jo = coint_johansen(L[['CPI', 'Joy']], det_order=0, k_ar_diff=k)
    rows.append({'k_ar_diff': k, 'levels_var_order': k + 1, 'trace_r0': float(jo.lr1[0]), 'cv95_r0': float(jo.cvt[0, 1]),
                 'cv99_r0': float(jo.cvt[0, 2]), 'reject_r0_5pct': bool(jo.lr1[0] > jo.cvt[0, 1])})
J = pd.DataFrame(rows)
J.round(4).to_csv(RES / 'johansen_sensitivity.csv', index=False)
print("Engle-Granger: stat=%.4f p=%.4f cv5=%.4f" % (eg[0], eg[1], eg[2][1]))
print(J.round(3).to_string(index=False))
assert abs(J.loc[J['k_ar_diff'] == 5, 'trace_r0'].iloc[0] - 10.9613) < 0.01, "no reproduce el 10.9613 publicado"

# ─── 2. Toda-Yamamoto ───
def wald_lags(res, cols, caused, causing, lags):
    """Wald conjunto sobre los coeficientes de `causing` (rezagos `lags`) en las ecuaciones de `caused`.

    statsmodels apila cov_params() por coeficiente y, dentro de cada uno, por ecuacion.
    """
    names = list(res.params.index)
    k = len(cols)
    pv = res.params.values.ravel(order='C')
    cov = res.cov_params().values
    sel = [names.index(f'L{l}.{c}') * k + cols.index(e) for e in caused for l in lags for c in causing]
    b = pv[sel]
    V = cov[np.ix_(sel, sel)]
    w = float(b @ np.linalg.solve(V, b))
    return {'chi2': w, 'df': len(sel), 'p': float(1 - stats.chi2.cdf(w, len(sel)))}


# control: con todos los rezagos coincide con test_causality de statsmodels
r4 = VAR(L.diff().dropna()).fit(4)
own = wald_lags(r4, NAMES, ['CPI'], EMO, range(1, 5))
sm_ = r4.test_causality('CPI', EMO, kind='wald')
assert abs(own['chi2'] - sm_.test_statistic) < 1e-6, "la ordenacion de cov_params no coincide"

ty = {}
for p in [2, 4, 6]:
    r = VAR(L).fit(p + 1)
    lags = range(1, p + 1)
    ty[f'p={p}'] = {
        'levels_var_order': p + 1, 'nobs': int(r.nobs),
        'portmanteau_p': float(r.test_whiteness(nlags=p + 6).pvalue),
        'emotions_to_cpi': wald_lags(r, NAMES, ['CPI'], EMO, lags),
        'cpi_to_emotions': wald_lags(r, NAMES, EMO, ['CPI'], lags),
        'joy_to_cpi': wald_lags(r, NAMES, ['CPI'], ['Joy'], lags),
    }
    print(f"TY p={p}: emo->CPI p={ty[f'p={p}']['emotions_to_cpi']['p']:.4f} | CPI->emo p={ty[f'p={p}']['cpi_to_emotions']['p']:.4f} "
          f"| Joy->CPI p={ty[f'p={p}']['joy_to_cpi']['p']:.4f}")

N = pd.DataFrame({'CPI': L['CPI'], 'Negativity': L[EMO[:3]].sum(axis=1), 'Joy': L['Joy']})
N.index.freq = 'MS'
NC = list(N.columns)
ty_neg = {}
for p in [2, 4, 6]:
    r = VAR(N).fit(p + 1)
    lags = range(1, p + 1)
    ty_neg[f'p={p}'] = {'neg_to_cpi': wald_lags(r, NC, ['CPI'], ['Negativity'], lags),
                        'cpi_to_neg': wald_lags(r, NC, ['Negativity'], ['CPI'], lags)}
    print(f"TY negatividad p={p}: neg->CPI p={ty_neg[f'p={p}']['neg_to_cpi']['p']:.4f} | CPI->neg p={ty_neg[f'p={p}']['cpi_to_neg']['p']:.4f}")

summary = {
    'engle_granger': {'stat': float(eg[0]), 'p': float(eg[1]), 'cv5': float(eg[2][1])},
    'johansen': J.to_dict(orient='records'),
    'johansen_reject_ks': [int(k) for k in J.loc[J['reject_r0_5pct'], 'k_ar_diff']],
    'toda_yamamoto': ty, 'toda_yamamoto_negativity': ty_neg,
}
with open(RES / 'r5_cointegration_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)
print("Resumen escrito en", RES / 'r5_cointegration_summary.json')
