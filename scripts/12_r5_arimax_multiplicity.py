"""
R5 (Access-2026-17671): correccion por comparaciones multiples en los modelos ARIMAX y
RMSE sin las observaciones de arranque.

- Los 23 modelos de results/arimax/all_results.json son 22 distintos: el modelo A (global) es
  identico al modelo del IPC General del modelo B.
- BH y Holm por familias (88 coeficientes de emociones, 44 de dummies), sobre los p-valores
  publicados: no cambia ninguna beta ni ningun p del articulo.
- Reajuste con SARIMAX(order, trend='c') como en 03_arimax_models.py. Control: el AIC coincide
  con el JSON. RMSE y MAE se recalculan sin las observaciones de arranque de la inicializacion
  difusa (loglikelihood_burn), que el script original incluia y que con d = 1 valen el nivel
  de la serie.
- Fig. 3 regenerada sin el residuo de arranque.
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.statespace.sarimax import SARIMAX

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
JSON_IN = ROOT / 'results' / 'arimax' / 'all_results.json'
GLOBAL = ROOT / 'data' / 'econometric_dataset_global.csv'
BY_CAT = ROOT / 'data' / 'econometric_dataset_by_category.csv'
RES = ROOT / 'results' / 'r5'
FIG = ROOT / 'figures'
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

EMO = {'Ira': 'Anger', 'Miedo': 'Fear', 'Tristeza': 'Sadness', 'Alegria': 'Joy'}
DUM = {'D_COVID': 'D_COVID', 'D_UCRANIA': 'D_UKRAINE'}
EXOG = list(EMO) + list(DUM)
LABEL = {
    'IPC_ECOICOP Índice General': 'CPI General Index',
    'ECOICOP_Alimentos y bebidas no alcohólicas': 'Food & Non-Alc. Beverages',
    'ECOICOP_Bebidas alcohólicas y tabaco': 'Alcohol & Tobacco',
    'ECOICOP_Vestido y calzado': 'Clothing & Footwear',
    'ECOICOP_Vivienda, agua, electricidad, gas y otros combustibles': 'Housing & Utilities',
    'ECOICOP_Muebles, artículos del hogar y artículos para el mantenimiento corriente del hogar': 'Furnishings & Household',
    'ECOICOP_Sanidad': 'Health',
    'ECOICOP_Transporte': 'Transport',
    'ECOICOP_Comunicaciones': 'Communications',
    'ECOICOP_Ocio y cultura': 'Recreation & Culture',
    'ECOICOP_Enseñanza': 'Education',
    'ECOICOP_Restaurantes y hoteles': 'Restaurants & Hotels',
    'ECOICOP_Otros bienes y servicios': 'Other Goods & Services',
    'ICM': 'ICM', 'ICC': 'ICC', 'IPI': 'IPI',
}
CAT_EN = {'economia': 'economy', 'politica': 'politics', 'sociedad': 'society', 'deporte': 'sports',
          'cultura': 'culture', 'medioambiente': 'environment'}


def label(var):
    if ' -> ' in var:
        cat, ind = var.split(' -> ')
        return f"{CAT_EN[cat]} → {LABEL[ind]}"
    return LABEL[var]


published = json.loads(JSON_IN.read_text(encoding='utf-8'))
g, b = published[0]['coefficients'], published[1]['coefficients']
assert all(abs(g[k]['coefficient'] - b[k]['coefficient']) < 1e-9 for k in g), "el modelo A no es el modelo B del IPC"
models = published[1:]
print(f"Modelos distintos: {len(models)}")

# ─── 1. Correccion multiple por familias sobre los p publicados ───
rows = []
for r in models:
    for k, v in r['coefficients'].items():
        if k in EMO or k in DUM:
            rows.append({'model': label(r['variable']), 'variable': r['variable'], 'term': EMO.get(k) or DUM[k],
                         'family': 'emotion' if k in EMO else 'dummy', 'beta': v['coefficient'], 'p': v['p_value']})
T = pd.DataFrame(rows)
for fam, s in T.groupby('family'):
    T.loc[s.index, 'q_bh'] = multipletests(s['p'], method='fdr_bh')[1]
    T.loc[s.index, 'p_holm'] = multipletests(s['p'], method='holm')[1]
T = T.sort_values(['family', 'p']).reset_index(drop=True)
T.round(4).to_csv(RES / 'arimax_multiplicity.csv', index=False)
mult = {}
for fam, s in T.groupby('family'):
    mult[fam] = {'n_tests': int(len(s)), 'expected_by_chance': float(0.05 * len(s)),
                 'nominal_05': int((s['p'] < .05).sum()), 'bh_05': int((s['q_bh'] < .05).sum()),
                 'holm_05': int((s['p_holm'] < .05).sum()),
                 'nominal': [{'model': r.model, 'term': r.term, 'beta': r.beta, 'p': r.p, 'q_bh': r.q_bh,
                              'p_holm': r.p_holm} for r in s[s['p'] < .05].itertuples()]}
    print(f"\nFamilia {fam}: {mult[fam]['n_tests']} tests | nominales {mult[fam]['nominal_05']} "
          f"(azar ~{mult[fam]['expected_by_chance']:.1f}) | BH {mult[fam]['bh_05']} | Holm {mult[fam]['holm_05']}")
    print(s[s['p'] < .05].round(4).to_string(index=False))

# ─── 2. Reajuste y RMSE sin observaciones de arranque ───
dg = pd.read_csv(GLOBAL, parse_dates=['Fecha'], encoding='utf-8-sig').set_index('Fecha').sort_index()
dg.index = pd.DatetimeIndex(dg.index).to_period('M').to_timestamp()
dc = pd.read_csv(BY_CAT, parse_dates=['Fecha'], encoding='utf-8-sig')


def design(var):
    if ' -> ' in var:
        cat, ind = var.split(' -> ')
        d = dc[dc['categoria'] == cat].set_index('Fecha').sort_index()
        d.index = pd.DatetimeIndex(d.index).to_period('M').to_timestamp()
        d = d[list(EMO)].join(dg[[ind] + list(DUM)], how='inner').dropna()
        return d[ind], d[list(EMO) + list(DUM)]
    m = dg[var].notna() & dg[EXOG].notna().all(axis=1)
    return dg.loc[m, var], dg.loc[m, EXOG]


fit_rows = []
global_fit = None
for r in models:
    y, X = design(r['variable'])
    order = tuple(int(x) for x in r['order'].strip('()').split(','))
    res = SARIMAX(y, exog=X, order=order, trend='c').fit(disp=False)
    burn = int(res.loglikelihood_burn)
    e_all = (y - res.fittedvalues)
    e = e_all.iloc[burn:]
    # Los modelos con terminos AR/MA reproducen el AIC salvo tolerancias del optimizador (< 0.3)
    aic_ok = abs(res.aic - r['aic']) < 0.5
    fit_rows.append({'model': label(r['variable']), 'variable': r['variable'], 'order': r['order'], 'n_obs': len(y),
                     'aic_published': r['aic'], 'aic_refit': round(float(res.aic), 2), 'aic_matches': aic_ok,
                     'burn': burn, 'first_residual': float(e_all.iloc[0]),
                     'rmse_published': r['rmse'], 'rmse_all': float(np.sqrt((e_all ** 2).mean())),
                     'rmse_corrected': float(np.sqrt((e ** 2).mean())), 'mae_published': r['mae'],
                     'mae_corrected': float(e.abs().mean()), 'ljung_box_ok': r['ljung_box_ok']})
    if r['variable'] == 'IPC_ECOICOP Índice General':
        global_fit = (y, res, burn)
F = pd.DataFrame(fit_rows)
F.round(4).to_csv(RES / 'arimax_rmse_corrected.csv', index=False)
print("\nReajuste (AIC publicado vs reajustado, RMSE publicado vs corregido):")
print(F[['model', 'order', 'aic_published', 'aic_refit', 'burn', 'rmse_published', 'rmse_all',
         'rmse_corrected', 'mae_corrected']].round(4).to_string(index=False))
print(f"Max |AIC reajustado - publicado| = {(F['aic_refit'] - F['aic_published']).abs().max():.3f}")
if not F['aic_matches'].all():
    sys.exit("Algun reajuste no reproduce el AIC publicado.")
assert (F['burn'] == (F['order'].str.contains(', 1, ')).astype(int)).all(), "burn distinto de d"
assert np.allclose(F['rmse_all'], F['rmse_published'], rtol=5e-3), "el RMSE publicado no es el de todas las obs"

# ─── 3. Fig. 3 sin el residuo de arranque (mismo diseno que 03_arimax_models.py) ───
y, res, burn = global_fit
y_pred = res.fittedvalues.iloc[burn:]
resid = (y - res.fittedvalues).iloc[burn:]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0, 0].plot(y.index, y, label='Actual', color='blue', lw=1.5)
axes[0, 0].plot(y_pred.index, y_pred, label='Predicted', color='red', ls='--', lw=1.5)
axes[0, 0].set_title('Actual vs Predicted - CPI General Index', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 1].plot(resid.index, resid, color='green', lw=1)
axes[0, 1].axhline(0, color='red', ls='--', lw=1)
axes[0, 1].set_title('Residuals', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[1, 0].hist(resid, bins=20, color='purple', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Residual Distribution', fontweight='bold')
stats.probplot(resid, dist='norm', plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot', fontweight='bold')
plt.suptitle('ARIMAX(0, 1, 0) - CPI General Index', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(FIG / 'fig_03_arimax_global.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)

gl = F[F['variable'] == 'IPC_ECOICOP Índice General'].iloc[0]
summary = {
    'n_models': len(models), 'multiplicity': mult,
    'global': {'rmse_published': float(gl['rmse_published']), 'rmse_corrected': float(gl['rmse_corrected']),
               'mae_corrected': float(gl['mae_corrected']), 'first_residual': float(gl['first_residual']),
               'first_level': float(y.iloc[0]), 'aic': float(gl['aic_refit']),
               'max_abs_residual': float(resid.abs().max()),
               'max_abs_residual_date': resid.abs().idxmax().strftime('%B %Y'),
               'residual_sd': float(resid.std())},
    'rmse_corrected': {r.variable: float(r.rmse_corrected) for r in F.itertuples()},
    'rmse_published_range': [float(F['rmse_published'].min()), float(F['rmse_published'].max())],
}
with open(RES / 'r5_arimax_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("\nResumen escrito en", RES / 'r5_arimax_summary.json')
