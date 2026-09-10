"""
R5 (Access-2026-17671): rupturas estructurales y estabilidad de la curacion de Google News.

Responde a R3-Q4 (un cambio de curacion variable en el tiempo podria confundir las
rupturas) y a R2 (implementacion de las rupturas):
- Control: la agregacion mensual del parquet reproduce econometric_dataset_global.csv.
- Rupturas en el IPC solo (serie oficial del INE, ajena a la curacion).
- Valores criticos sup-F por Monte Carlo para la busqueda secuencial tipo Bai-Perron.
- Series de composicion del corpus (volumen, n.o de fuentes, concentracion, cuotas por categoria).
- Indice de emociones de composicion fija (fuentes presentes todos los anos, pesos fijos):
  se repiten Chow/Bai-Perron, contrastes pre/post-COVID y el Granger conjunto VAR(4).
"""

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.api import VAR

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results' / 'r5'
RES.mkdir(parents=True, exist_ok=True)
# Requiere los datos a nivel de titular, que no se distribuyen por derechos de autor.
RAW = Path(os.environ.get('HEADLINES_PARQUET', ROOT / 'data' / 'dataset_emociones.parquet'))
GLOBAL = ROOT / 'data' / 'econometric_dataset_global.csv'
if not RAW.exists():
    sys.exit('Este script necesita dataset_emociones.parquet (datos a nivel de titular, no publicos). '
             'Indica su ruta con la variable de entorno HEADLINES_PARQUET. Sus resultados estan en results/r5/.')

BREAK_COVID = pd.Timestamp('2020-03-01')
BREAK_UKRAINE = pd.Timestamp('2022-02-01')
BREAK_BP = pd.Timestamp('2021-12-01')
MIN_SEGMENT = 12
N_SIM = 10000
SEED = 42
EMO_EN = ['anger', 'fear', 'sadness', 'joy']
EMO_ES = ['Ira', 'Miedo', 'Tristeza', 'Alegria']


def chow_test(y, X, break_date):
    """Igual que 05_structural_breaks.py."""
    X_full = add_constant(X, has_constant='add')
    rss_full = OLS(y, X_full).fit().ssr
    n, k = len(y), X_full.shape[1]
    pre = y.index < break_date
    rss1 = OLS(y[pre], X_full[pre]).fit().ssr
    rss2 = OLS(y[~pre], X_full[~pre]).fit().ssr
    f = ((rss_full - (rss1 + rss2)) / k) / ((rss1 + rss2) / (n - 2 * k))
    return float(f), float(1 - stats.f.cdf(f, k, n - 2 * k)), int(pre.sum()), int((~pre).sum())


def sup_f(y, X, min_segment=MIN_SEGMENT):
    """Primer paso de la busqueda secuencial de 05_structural_breaks.py (max F sobre fechas candidatas)."""
    best = max(((y.index[i], chow_test(y, X, y.index[i])[0])
                for i in range(min_segment, len(y) - min_segment)), key=lambda t: t[1])
    return best[0], float(best[1])


def sup_f_critical(X, n_sim=N_SIM, seed=SEED):
    """Criticos sup-F bajo H0 de estabilidad con los regresores observados y errores iid N(0,1)."""
    rng = np.random.default_rng(seed)
    idx = X.index
    Xf = add_constant(X, has_constant='add').values
    n, k = Xf.shape
    cand = range(MIN_SEGMENT, n - MIN_SEGMENT)

    def rss(A, b):
        beta = np.linalg.lstsq(A, b, rcond=None)[0]
        r = b - A @ beta
        return r @ r

    sims = np.empty(n_sim)
    for s in range(n_sim):
        e = rng.standard_normal(n)
        full = rss(Xf, e)
        best = 0.0
        for i in cand:
            r12 = rss(Xf[:i], e[:i]) + rss(Xf[i:], e[i:])
            best = max(best, ((full - r12) / k) / (r12 / (n - 2 * k)))
        sims[s] = best
    return {'cv_10': float(np.quantile(sims, 0.90)), 'cv_05': float(np.quantile(sims, 0.95)),
            'cv_01': float(np.quantile(sims, 0.99)), 'n_sim': n_sim, 'k': int(k), 'n': int(n)}


# ─── Datos crudos (mismo filtro y agregacion que 01/02) ───
raw = pd.read_parquet(RAW, columns=['anio', 'mes_pub', 'Medio', 'categoria'] + EMO_EN)
raw = raw[(raw['anio'] >= 2017) & (raw['anio'] <= 2023)].copy()
raw['Fecha'] = pd.to_datetime(raw['anio'].astype(str) + '-' + raw['mes_pub'].astype(str).str.zfill(2) + '-01')
print(f"Titulares 2017-2023: {len(raw):,} | fuentes distintas: {raw['Medio'].nunique():,} | "
      f"sin fuente: {raw['Medio'].isna().sum():,}")

glob = pd.read_csv(GLOBAL, parse_dates=['Fecha'], encoding='utf-8-sig').set_index('Fecha').sort_index()
glob.index = pd.DatetimeIndex(glob.index).to_period('M').to_timestamp()
ipc_col = next(c for c in glob.columns if c.startswith('IPC_ECOICOP') and 'General' in c)
cpi = glob[ipc_col].dropna()

monthly = raw.groupby('Fecha')[EMO_EN].mean()
diff_ctrl = float((monthly.values - glob.loc[monthly.index, EMO_ES].values).__abs__().max())
print(f"Control agregacion: max |diferencia| con econometric_dataset_global = {diff_ctrl:.2e}")
if diff_ctrl > 1e-8:
    sys.exit("La agregacion no reproduce el dataset publicado.")

summary = {'n_headlines': int(len(raw)), 'n_sources': int(raw['Medio'].nunique()),
           'aggregation_control_max_abs_diff': diff_ctrl}

# ─── 1. Especificacion publicada y rupturas del IPC solo ───
E = glob.loc[cpi.index, EMO_ES]
none = pd.DataFrame(index=cpi.index)
spec = {}
for label, X in [('paper: CPI ~ const + 4 emotions', E), ('CPI alone: CPI ~ const', none)]:
    row = {}
    for name, bd in [('chow_covid', BREAK_COVID), ('chow_ukraine', BREAK_UKRAINE), ('chow_dec2021', BREAK_BP)]:
        f, p, n1, n2 = chow_test(cpi, X, bd)
        row[name] = {'F': f, 'p': p, 'n1': n1, 'n2': n2}
    d, f = sup_f(cpi, X)
    row['supF'] = {'date': str(d.date()), 'F': f}
    print(f"  sup-F criticos Monte Carlo para [{label}] ({N_SIM} sim.)...")
    row['supF_critical'] = sup_f_critical(X)
    spec[label] = row
    print(f"{label}: sup-F={f:.2f} en {d.date()} | criticos 5%={row['supF_critical']['cv_05']:.2f} "
          f"1%={row['supF_critical']['cv_01']:.2f} | Chow COVID F={row['chow_covid']['F']:.2f} "
          f"Ucrania F={row['chow_ukraine']['F']:.2f}")
summary['break_tests'] = spec

# ─── 2. Composicion del corpus ───
g = raw.groupby('Fecha')
comp = pd.DataFrame({
    'n_headlines': g.size(),
    'n_sources': g['Medio'].nunique(),
    'top10_share': g['Medio'].apply(lambda s: s.value_counts(normalize=True).iloc[:10].sum()),
    'hhi_sources': g['Medio'].apply(lambda s: (s.value_counts(normalize=True) ** 2).sum()),
})
cat_share = raw.groupby(['Fecha', 'categoria']).size().unstack(fill_value=0)
cat_share = cat_share.div(cat_share.sum(axis=1), axis=0).add_prefix('share_')
comp = comp.join(cat_share)
comp.round(5).to_csv(RES / 'corpus_composition_monthly.csv')

comp_rows = []
for c in comp.columns:
    s = comp[c]
    d, f = sup_f(s, pd.DataFrame(index=s.index))
    comp_rows.append({'series': c, 'mean': s.mean(), 'cv': s.std() / s.mean(),
                      'supF_date': str(d.date()), 'supF': f,
                      'F_covid': chow_test(s, pd.DataFrame(index=s.index), BREAK_COVID)[0],
                      'F_dec2021': chow_test(s, pd.DataFrame(index=s.index), BREAK_BP)[0],
                      'F_ukraine': chow_test(s, pd.DataFrame(index=s.index), BREAK_UKRAINE)[0]})
comp_tab = pd.DataFrame(comp_rows)
comp_tab.round(4).to_csv(RES / 'corpus_composition_breaks.csv', index=False)
print("\nComposicion del corpus (ruptura de media, constante sola):")
print(comp_tab.round(3).to_string())
summary['composition'] = comp_tab.round(4).to_dict(orient='records')

# ─── 3. Indice de emociones de composicion fija ───
years = raw.groupby('Medio')['anio'].nunique()
persistent = years[years == 7].index
sub = raw[raw['Medio'].isin(persistent)]
w = sub['Medio'].value_counts(normalize=True)
sm = sub.groupby(['Fecha', 'Medio'])[EMO_EN].mean().reset_index()
sm['w'] = sm['Medio'].map(w)
fixed = sm.groupby('Fecha').apply(lambda d: pd.Series(
    {e: np.average(d[e], weights=d['w']) for e in EMO_EN}))
fixed = fixed.reindex(cpi.index)
coverage = {'n_persistent_sources': int(len(persistent)),
            'headline_share_persistent': float(len(sub) / len(raw)),
            'min_sources_per_month': int(sm.groupby('Fecha').size().min())}
print(f"\nFuentes presentes los 7 anos: {coverage['n_persistent_sources']} "
      f"({100 * coverage['headline_share_persistent']:.1f}% de los titulares); "
      f"minimo {coverage['min_sources_per_month']} fuentes/mes")

corr = {}
for en, es in zip(EMO_EN, EMO_ES):
    a, b = fixed[en], glob.loc[cpi.index, es]
    corr[en] = {'levels': float(a.corr(b)), 'first_diff': float(a.diff().corr(b.diff()))}
print("Correlacion indice fijo vs publicado:", {k: (round(v['levels'], 3), round(v['first_diff'], 3)) for k, v in corr.items()})

Ef = fixed[EMO_EN].copy()
Ef.columns = EMO_ES
fixed_breaks = {}
for name, bd in [('chow_covid', BREAK_COVID), ('chow_ukraine', BREAK_UKRAINE)]:
    f, p, n1, n2 = chow_test(cpi, Ef, bd)
    fixed_breaks[name] = {'F': f, 'p': p}
d, f = sup_f(cpi, Ef)
fixed_breaks['supF'] = {'date': str(d.date()), 'F': f}
print(f"Rupturas con indice fijo: Chow COVID F={fixed_breaks['chow_covid']['F']:.2f}, "
      f"Ucrania F={fixed_breaks['chow_ukraine']['F']:.2f}, sup-F={f:.2f} en {d.date()}")

pre = cpi.index < BREAK_COVID
prepost = {}
for en, es in zip(EMO_EN, EMO_ES):
    for lab, s in [('published', glob.loc[cpi.index, es]), ('fixed', fixed[en])]:
        prepost.setdefault(en, {})[lab] = {
            'delta_pct': float(100 * (s[~pre].mean() / s[pre].mean() - 1)),
            'mw_p': float(stats.mannwhitneyu(s[pre], s[~pre]).pvalue),
            'ks_p': float(stats.ks_2samp(s[pre], s[~pre]).pvalue)}
print("Pre/post-COVID (delta %, MW p) publicado vs fijo:",
      {k: ((round(v['published']['delta_pct'], 1), round(v['published']['mw_p'], 4)),
           (round(v['fixed']['delta_pct'], 1), round(v['fixed']['mw_p'], 4))) for k, v in prepost.items()})

Xf = pd.concat([cpi.rename('CPI'), Ef.rename(columns=dict(zip(EMO_ES, ['Anger', 'Fear', 'Sadness', 'Joy'])))], axis=1)
Xf = Xf.diff().dropna()
Xf.index.freq = 'MS'
granger_fixed = {}
for p in [2, 4, 6]:
    r = VAR(Xf).fit(p)
    granger_fixed[f'VAR({p})'] = {
        'emotions_to_cpi_p': float(r.test_causality('CPI', ['Anger', 'Fear', 'Sadness', 'Joy'], kind='f').pvalue),
        'cpi_to_emotions_p': float(r.test_causality(['Anger', 'Fear', 'Sadness', 'Joy'], ['CPI'], kind='f').pvalue),
        'portmanteau_p_plus5': float(r.test_whiteness(nlags=p + 5).pvalue)}
print("Granger conjunto con indice fijo:", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in granger_fixed.items()})

summary['fixed_composition'] = {'coverage': coverage, 'correlation_with_published': corr,
                                'breaks': fixed_breaks, 'pre_post_covid': prepost,
                                'joint_granger': granger_fixed}
fixed.round(6).to_csv(RES / 'fixed_composition_emotions.csv')

with open(RES / 'r5_breaks_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("\nResumen escrito en", RES / 'r5_breaks_summary.json')
