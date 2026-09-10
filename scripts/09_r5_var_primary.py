"""
R5 (Access-2026-17671): VAR(4) como especificacion principal, correccion por
comparaciones multiples y FEVD generalizada.

- Control: reproduce las cifras publicadas en R4 con VAR(6) (FEVD 0.6732, GFEVD 39.5 %).
- Criterios de rezago y diagnosticos de VAR(2), VAR(4) y VAR(6).
- Granger: tests conjuntos de bloque (Holm + p-valor bootstrap bajo H0) y 20 tests
  por par dentro del VAR con rezago fijo (q-valores BH, Holm).
- FEVD Cholesky (5 ordenaciones), FEVD generalizada de Pesaran-Shin normalizada por
  filas (Diebold-Yilmaz) e IRF ortogonalizadas con bandas Monte Carlo.
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
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.multitest import multipletests

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'econometric_dataset_global.csv'
RES = ROOT / 'results' / 'r5'
FIG = ROOT / 'figures'
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(exist_ok=True)

PRIMARY = 4
LAGS = [2, 4, 6]
MAX_LAGS = 8
H = 12
B = 1999
SEED = 42

NAMES = ['CPI', 'Anger', 'Fear', 'Sadness', 'Joy']
EMO = NAMES[1:]
ES_EMO = ['Ira', 'Miedo', 'Tristeza', 'Alegria']
ORDERINGS = {
    'baseline': ['CPI', 'Anger', 'Fear', 'Sadness', 'Joy'],
    'joy_first': ['Joy', 'CPI', 'Anger', 'Fear', 'Sadness'],
    'joy_second': ['CPI', 'Joy', 'Anger', 'Fear', 'Sadness'],
    'reversed': ['Joy', 'Sadness', 'Fear', 'Anger', 'CPI'],
    'emotions_first': ['Anger', 'Fear', 'Sadness', 'Joy', 'CPI'],
}

# ─── Datos: igual que 04_var_granger_analysis.py (se diferencian las 5 series) ───
df = pd.read_csv(DATA, parse_dates=['Fecha'], encoding='utf-8-sig').set_index('Fecha').sort_index()
df.index = pd.DatetimeIndex(df.index).to_period('M').to_timestamp()
ipc_col = next(c for c in df.columns if c.startswith('IPC_ECOICOP') and 'General' in c)
levels = df[[ipc_col] + ES_EMO].dropna()
levels.columns = NAMES

adf_rows = []
for c in NAMES:
    a0 = adfuller(levels[c], autolag='AIC')
    a1 = adfuller(levels[c].diff().dropna(), autolag='AIC')
    adf_rows.append({'variable': c, 'adf_level': a0[0], 'p_level': a0[1],
                     'adf_diff': a1[0], 'p_diff': a1[1]})
pd.DataFrame(adf_rows).round(4).to_csv(RES / 'adf_tests.csv', index=False)

X = levels.diff().dropna()
X.index.freq = 'MS'
print(f"Observaciones tras diferenciar: {len(X)} ({X.index[0].date()} a {X.index[-1].date()})")


# ─── Utilidades ───
def gfevd(res, horizon=H):
    """FEVD generalizada (Pesaran-Shin 1998). Devuelve (bruta, normalizada por filas)."""
    sig = np.asarray(res.sigma_u)
    phi = res.ma_rep(horizon - 1)
    k = sig.shape[0]
    raw = np.zeros((k, k))
    for i in range(k):
        den = sum(phi[l][i] @ sig @ phi[l][i] for l in range(horizon))
        for j in range(k):
            num = sum((phi[l][i] @ sig[:, j]) ** 2 for l in range(horizon)) / sig[j, j]
            raw[i, j] = num / den
    return raw, raw / raw.sum(axis=1, keepdims=True)


def lag_design(Y, p):
    """Matriz de regresores [1, y_{t-1}, ..., y_{t-p}] para t = p..T-1."""
    T, k = Y.shape
    cols = [np.ones(T - p)] + [Y[p - l:T - l, j] for l in range(1, p + 1) for j in range(k)]
    return np.column_stack(cols)


def bootstrap_granger(Xdf, p, caused, causing, n_boot=B, seed=SEED):
    """p-valor bootstrap del test de Wald de no causalidad, generando bajo H0.

    Se estima el VAR restringido (sin los rezagos de `causing` en las ecuaciones de
    `caused`), se simulan series recursivamente remuestreando filas de los residuos
    centrados y se recalcula el estadistico del VAR sin restringir en cada replica.
    """
    names = list(Xdf.columns)
    Y = Xdf.values
    T, k = Y.shape
    Z = lag_design(Y, p)
    ci = [names.index(c) for c in caused]
    ki = [names.index(c) for c in causing]
    drop = {1 + (l - 1) * k + j for l in range(1, p + 1) for j in ki}
    coef = np.zeros((Z.shape[1], k))
    for i in range(k):
        keep = [c for c in range(Z.shape[1]) if not (i in ci and c in drop)]
        coef[keep, i] = np.linalg.lstsq(Z[:, keep], Y[p:, i], rcond=None)[0]
    U = Y[p:] - Z @ coef
    U = U - U.mean(axis=0)

    stat = VAR(Xdf).fit(p).test_causality(caused, causing, kind='f').test_statistic
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_boot):
        draw = U[rng.integers(0, len(U), len(U))]
        Ys = np.empty_like(Y)
        Ys[:p] = Y[:p]
        for t in range(p, T):
            z = np.concatenate([[1.0], Ys[t - p:t][::-1].ravel()])
            Ys[t] = z @ coef + draw[t - p]
        s = VAR(pd.DataFrame(Ys, columns=names)).fit(p).test_causality(caused, causing, kind='f').test_statistic
        exceed += s >= stat
    return (exceed + 1) / (n_boot + 1)


# ─── 0. Control: reproducir R4 (VAR(6)) ───
print("\n[0] Control contra las cifras publicadas en R4 (VAR(6))")
r6 = VAR(X).fit(6)
fevd6 = r6.fevd(H).decomp[0, -1, :]
_, g6 = gfevd(r6)
checks = {
    'FEVD CPI own h12 = 0.6732': round(fevd6[0], 4) == 0.6732,
    'FEVD Fear h12 = 0.1154': round(fevd6[2], 4) == 0.1154,
    'GFEVD emotions = 39.5%': round(100 * (1 - g6[0, 0]), 1) == 39.5,
    'Portmanteau VAR(6) = 0.0327': round(r6.test_whiteness(nlags=11).pvalue, 4) == 0.0327,
}
published_x = {'joy_first': 0.6574, 'joy_second': 0.6732, 'reversed': 0.5596, 'emotions_first': 0.5596}
for name, cpi_share in published_x.items():
    rr = VAR(X[ORDERINGS[name]]).fit(6)
    got = rr.fevd(H).decomp[ORDERINGS[name].index('CPI'), -1, ORDERINGS[name].index('CPI')]
    checks[f'Table X {name} CPI = {cpi_share}'] = round(got, 4) == cpi_share
for k_, ok in checks.items():
    print(f"   {'OK ' if ok else 'FALLO'} {k_}")
if not all(checks.values()):
    sys.exit("El control no reproduce R4: revisar antes de continuar.")

# ─── 1. Criterios de rezago ───
print("\n[1] Criterios de seleccion de rezago")
sel = VAR(X).select_order(maxlags=MAX_LAGS)
crit = pd.DataFrame({c.upper(): sel.ics[c] for c in ['aic', 'bic', 'hqic', 'fpe']})
crit.index.name = 'lag'
crit.to_csv(RES / 'lag_selection.csv')
print(crit.round(3).to_string())
print("   Seleccionados:", {k.upper(): int(v) for k, v in sel.selected_orders.items()})

# ─── 2. Diagnosticos y Granger por especificacion ───
summary = {'n_obs_differenced': len(X), 'selected_orders': {k.upper(): int(v) for k, v in sel.selected_orders.items()},
           'control_r4': {k: bool(v) for k, v in checks.items()}, 'models': {}}
t8_rows = []
pair_tables = {}
for p in LAGS:
    print(f"\n[2] VAR({p})")
    r = VAR(X).fit(p)
    diag = {
        'nobs': int(r.nobs),
        'coef_per_equation': int(r.params.shape[0]),
        'portmanteau_p_plus5': float(r.test_whiteness(nlags=p + 5).pvalue),
        'portmanteau_12': float(r.test_whiteness(nlags=12).pvalue),
        'portmanteau_16': float(r.test_whiteness(nlags=16).pvalue),
        'normality_p': float(r.test_normality().pvalue),
        'max_root_modulus': float(np.max(1 / np.abs(r.roots))),
    }

    j_e2c = r.test_causality('CPI', EMO, kind='f')
    j_c2e = r.test_causality(EMO, ['CPI'], kind='f')
    print(f"   bootstrap ({B} replicas) de los tests conjuntos...")
    boot_e2c = bootstrap_granger(X, p, ['CPI'], EMO)
    boot_c2e = bootstrap_granger(X, p, EMO, ['CPI'])
    holm_asym = multipletests([j_e2c.pvalue, j_c2e.pvalue], method='holm')[1]
    holm_boot = multipletests([boot_e2c, boot_c2e], method='holm')[1]
    joint = {
        'emotions_to_cpi': {'F': float(j_e2c.test_statistic), 'df': [int(d) for d in j_e2c.df],
                            'p': float(j_e2c.pvalue), 'p_holm': float(holm_asym[0]),
                            'p_boot': float(boot_e2c), 'p_boot_holm': float(holm_boot[0])},
        'cpi_to_emotions': {'F': float(j_c2e.test_statistic), 'df': [int(d) for d in j_c2e.df],
                            'p': float(j_c2e.pvalue), 'p_holm': float(holm_asym[1]),
                            'p_boot': float(boot_c2e), 'p_boot_holm': float(holm_boot[1])},
    }

    rows = []
    for cause in NAMES:
        for effect in NAMES:
            if cause == effect:
                continue
            t = r.test_causality(effect, [cause], kind='f')
            rows.append({'cause': cause, 'effect': effect, 'F': t.test_statistic,
                         'df_num': int(t.df[0]), 'df_den': int(t.df[1]), 'p': t.pvalue})
    pairs = pd.DataFrame(rows)
    pairs['q_bh'] = multipletests(pairs['p'], method='fdr_bh')[1]
    pairs['p_holm'] = multipletests(pairs['p'], method='holm')[1]
    pairs = pairs.sort_values('p').reset_index(drop=True)
    pair_tables[p] = pairs
    pairs.round(4).to_csv(RES / f'granger_pairs_var{p}.csv', index=False)

    raw, g = gfevd(r)
    summary['models'][f'VAR({p})'] = {
        **diag, 'joint': joint,
        'pairs_nominal_05': int((pairs['p'] < 0.05).sum()),
        'pairs_bh_05': int((pairs['q_bh'] < 0.05).sum()),
        'pairs_holm_05': int((pairs['p_holm'] < 0.05).sum()),
        'nominal_pairs': [f"{a}->{b}" for a, b in pairs.loc[pairs['p'] < 0.05, ['cause', 'effect']].values],
        'gfevd_cpi_h12': dict(zip(NAMES, [float(v) for v in g[0]])),
        'gfevd_emotions_total': float(1 - g[0, 0]),
        'gfevd_cpi_raw_rowsum': float(raw[0].sum()),
        'fevd_cholesky_baseline_h12': dict(zip(NAMES, [float(v) for v in r.fevd(H).decomp[0, -1, :]])),
    }
    t8_rows.append({
        'Model': f'VAR({p})', 'Obs': int(r.nobs), 'Coef./eq.': int(r.params.shape[0]),
        'Portm. p': diag['portmanteau_p_plus5'],
        'Emo→CPI p (boot)': boot_e2c, 'CPI→Emo p (boot)': boot_c2e,
        'Pairs p<.05': int((pairs['p'] < 0.05).sum()), 'Pairs q<.05': int((pairs['q_bh'] < 0.05).sum()),
    })
    print(f"   nobs={r.nobs} coef/ec={r.params.shape[0]} Portm(p+5)={diag['portmanteau_p_plus5']:.4f} "
          f"Portm(12)={diag['portmanteau_12']:.4f} normalidad p={diag['normality_p']:.4f} "
          f"raiz max={diag['max_root_modulus']:.3f}")
    for nm, jj in joint.items():
        print(f"   conjunto {nm:16s} F={jj['F']:.3f} df={jj['df']} p={jj['p']:.4f} Holm={jj['p_holm']:.4f} "
              f"boot={jj['p_boot']:.4f} boot-Holm={jj['p_boot_holm']:.4f}")
    print(f"   pares: nominal {summary['models'][f'VAR({p})']['pairs_nominal_05']}, "
          f"BH {summary['models'][f'VAR({p})']['pairs_bh_05']}, Holm {summary['models'][f'VAR({p})']['pairs_holm_05']}")
    print(f"   GFEVD CPI h12: " + ", ".join(f"{n} {100 * v:.1f}%" for n, v in zip(NAMES, g[0]))
          + f" | emociones {100 * (1 - g[0, 0]):.1f}% | suma bruta fila {raw[0].sum():.3f}")

pd.DataFrame(t8_rows).round(4).to_csv(RES / 'table_VIII_r5.csv', index=False)
pair_tables[PRIMARY].round(4).to_csv(RES / 'table_VII_r5.csv', index=False)
print("\nTabla VII (VAR(4), 20 pares):")
print(pair_tables[PRIMARY].round(4).to_string())

# Metodo anterior (bivariante, mejor rezago por par), solo para documentar en la carta
legacy = []
for p in LAGS:
    for cause in NAMES:
        for effect in NAMES:
            if cause == effect:
                continue
            gc = grangercausalitytests(X[[effect, cause]], maxlag=p, verbose=False)
            ps = [gc[l][0]['ssr_ftest'][1] for l in range(1, p + 1)]
            legacy.append({'var_order': p, 'cause': cause, 'effect': effect,
                           'best_lag': int(np.argmin(ps)) + 1, 'p_min': min(ps)})
legacy = pd.DataFrame(legacy)
legacy['p_holm_20'] = legacy.groupby('var_order')['p_min'].transform(lambda s: multipletests(s, method='holm')[1])
legacy['p_bonf_lag_then_bh'] = legacy.groupby('var_order')['p_min'].transform(
    lambda s: multipletests(np.minimum(1, s * legacy.loc[s.index, 'var_order']), method='fdr_bh')[1])
legacy.round(4).to_csv(RES / 'legacy_bivariate_bestlag.csv', index=False)
summary['legacy_bivariate_bestlag'] = {
    f'VAR({p})': {'nominal': int((g_['p_min'] < .05).sum()), 'holm': int((g_['p_holm_20'] < .05).sum()),
                  'bonf_lag_then_bh': int((g_['p_bonf_lag_then_bh'] < .05).sum())}
    for p, g_ in legacy.groupby('var_order')}

# ─── 3. FEVD, ordenaciones y GFEVD de la especificacion principal ───
print(f"\n[3] FEVD de VAR({PRIMARY})")
rp = VAR(X).fit(PRIMARY)
fevd_full = pd.DataFrame(rp.fevd(H).decomp[0], columns=NAMES, index=range(1, H + 1))
fevd_full.index.name = 'Horizon (months)'
fevd_full.round(4).to_csv(RES / 'fevd_cpi_var4.csv')
fevd_full.loc[[1, 3, 6, 12]].round(4).to_csv(RES / 'table_IX_r5.csv')
print(fevd_full.loc[[1, 3, 6, 12]].round(4).to_string())

t10 = []
for name, order in ORDERINGS.items():
    rr = VAR(X[order]).fit(PRIMARY)
    dec = rr.fevd(H).decomp[order.index('CPI'), -1, :]
    t10.append({'Ordering': name, **{n: dec[order.index(n)] for n in NAMES}})
raw_p, g_p = gfevd(rp)
t10.append({'Ordering': 'Generalized (P-S)', **dict(zip(NAMES, g_p[0]))})
t10 = pd.DataFrame(t10)
t10['Emotions total'] = 1 - t10['CPI']
t10.round(4).to_csv(RES / 'table_X_r5.csv', index=False)
print(t10.round(4).to_string())
chol = t10.iloc[:-1]
summary['primary'] = {
    'order': PRIMARY,
    'cholesky_cpi_own_range': [float(chol['CPI'].min()), float(chol['CPI'].max())],
    'cholesky_emotions_range': [float(chol['Emotions total'].min()), float(chol['Emotions total'].max())],
    'cholesky_joy_range': [float(chol['Joy'].min()), float(chol['Joy'].max())],
    'gfevd_raw_rowsum_cpi': float(raw_p[0].sum()),
}

# ─── 4. IRF ortogonalizadas con bandas ───
print(f"\n[4] IRF de VAR({PRIMARY}) con bandas asintoticas al 95 % (Lutkepohl)")
irf = rp.irf(H)
# errband_mc de statsmodels 0.14.6 devuelve lower == upper (todas las replicas
# iguales), asi que se usan los errores estandar analiticos de las IRF ortogonalizadas.
se = irf.stderr(orth=True)
lo, hi = irf.orth_irfs - 1.96 * se, irf.orth_irfs + 1.96 * se
irf_out = {}
for i, imp in enumerate(NAMES):
    for j, resp in enumerate(NAMES):
        irf_out[f'{imp} -> {resp}'] = {'irf': irf.orth_irfs[:, j, i].tolist(),
                                       'lower': lo[:, j, i].tolist(), 'upper': hi[:, j, i].tolist()}
with open(RES / 'irf_var4.json', 'w', encoding='utf-8') as f:
    json.dump(irf_out, f, indent=1)
irf_desc = {}
for key in [f'{e} -> CPI' for e in EMO] + [f'CPI -> {e}' for e in EMO]:
    d = irf_out[key]
    v = np.array(d['irf'])
    sig_h = [h for h in range(H + 1) if d['lower'][h] > 0 or d['upper'][h] < 0]
    irf_desc[key] = {'h0_3': np.round(v[:4], 4).tolist(), 'peak_h': int(np.argmax(np.abs(v))),
                     'peak': float(v[np.argmax(np.abs(v))]), 'band_excludes_zero_at': sig_h}
    print(f"   {key:16s} h0-6={np.round(v[:7], 3).tolist()} pico h={irf_desc[key]['peak_h']} "
          f"banda excluye 0 en h={sig_h}")
summary['irf_var4'] = irf_desc

# ─── 5. Figuras (estilo de 06_generate_article_figures.py) ───
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                     'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.5})
COL = {'Anger': '#d62728', 'Fear': '#9467bd', 'Joy': '#2ca02c', 'Sadness': '#1f77b4'}

fig, axes = plt.subplots(2, 4, figsize=(7.16, 4))
for j, emo in enumerate(EMO):
    for row, key, title in [(0, f'{emo} -> CPI', f'{emo} -> CPI'), (1, f'CPI -> {emo}', f'CPI -> {emo}')]:
        d = irf_out[key]
        ax = axes[row, j]
        ax.fill_between(range(H + 1), d['lower'], d['upper'], color=COL[emo], alpha=0.15, lw=0)
        ax.plot(range(H + 1), d['irf'], color=COL[emo], lw=1.5)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_title(title, fontsize=8)
        if j == 0:
            ax.set_ylabel('Response')
        if row == 1:
            ax.set_xlabel('Months')
plt.suptitle('Orthogonalized Impulse Response Functions, VAR(4)', fontsize=10, fontweight='bold')
plt.tight_layout()
fig.savefig(FIG / 'fig_04_irf_selected.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3))
bottom = np.zeros(H)
fcol = {'CPI': '#1f77b4', 'Anger': '#d62728', 'Fear': '#9467bd', 'Sadness': '#ff7f0e', 'Joy': '#2ca02c'}
for c in NAMES:
    ax.bar(fevd_full.index, fevd_full[c], bottom=bottom, label=c, color=fcol[c], alpha=0.8, width=0.8)
    bottom += fevd_full[c].values
ax.set_xlabel('Forecast horizon (months)')
ax.set_ylabel('Proportion of variance')
ax.set_title('FEVD of CPI, VAR(4)')
ax.set_xticks(list(fevd_full.index))
ax.legend(fontsize=6, loc='upper center', ncol=5, columnspacing=0.8, handlelength=1.0, frameon=False)
ax.set_ylim(0, 1.20)
plt.tight_layout()
fig.savefig(FIG / 'fig_05_fevd_ipc.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)

with open(RES / 'r5_var_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
print("\nResumen escrito en", RES / 'r5_var_summary.json')
print("Metodo anterior (bivariante, mejor rezago):", summary['legacy_bivariate_bestlag'])
