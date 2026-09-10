"""
Script 06: Generacion de figuras para articulo IEEE Access
Articulo IEEE Access - Jorge Jose Zamora, UPCT

~7-8 figuras en formato IEEE Access (300+ dpi, sans-serif, single/double column)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sys
import os
import json
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

# IEEE Access style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

COLORS = {
    'Ira': '#d62728',      'Miedo': '#9467bd',
    'Alegria': '#2ca02c',   'Tristeza': '#1f77b4',
    'Sorpresa': '#ff7f0e',  'Asco': '#8c564b',
    'Otros': '#7f7f7f',
}

# Translation mappings: Spanish -> English
TR_EMO = {
    'Ira': 'Anger', 'Miedo': 'Fear', 'Tristeza': 'Sadness',
    'Alegria': 'Joy', 'Sorpresa': 'Surprise', 'Asco': 'Disgust',
    'Otros': 'Others',
}
TR_CAT = {
    'cultura': 'Culture', 'deporte': 'Sports', 'economia': 'Economy',
    'medioambiente': 'Environment', 'politica': 'Politics', 'sociedad': 'Society',
}

def tr_emo(name):
    return TR_EMO.get(name, name)

def tr_cat(name):
    return TR_CAT.get(name.lower(), name.capitalize())

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_GLOBAL = BASE_DIR / 'article_outputs' / 'data' / 'econometric_dataset_global.csv'
INPUT_BY_CAT = BASE_DIR / 'article_outputs' / 'data' / 'econometric_dataset_by_category.csv'
INPUT_EMOTIONS = BASE_DIR / 'article_outputs' / 'data' / 'dataset_emociones_2017_2023.parquet'
OUTPUT_DIR = BASE_DIR / 'article_outputs' / 'figures'

# Check for VAR/Granger results
VAR_DIR = BASE_DIR / 'article_outputs' / 'var_granger'
STRUCT_DIR = BASE_DIR / 'article_outputs' / 'structural'

EMOTION_VARS_ES = ['Ira', 'Miedo', 'Alegria', 'Tristeza', 'Sorpresa', 'Asco']
BREAK_COVID = pd.Timestamp('2020-03-01')

os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(fig, name):
    png_path = os.path.join(OUTPUT_DIR, f'{name}.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"   -> {name}.png")


print("=" * 80)
print("GENERACION DE FIGURAS - ARTICULO IEEE ACCESS")
print("=" * 80)

# Cargar datos
print("\n[0] Cargando datos...")
df_global = pd.read_csv(INPUT_GLOBAL, parse_dates=['Fecha'])
df_global = df_global.sort_values('Fecha')

df_by_cat = pd.read_csv(INPUT_BY_CAT, parse_dates=['Fecha'])

if os.path.exists(INPUT_EMOTIONS):
    df_emotions = pd.read_parquet(INPUT_EMOTIONS)
    print(f"   Dataset emociones: {len(df_emotions):,}")
else:
    df_emotions = None

print(f"   Dataset global: {len(df_global)} obs")
print(f"   Dataset por categoria: {len(df_by_cat)} filas")

# ═══ Fig 1: Evolucion mensual de emociones + media movil 12m ═══
print("\n[Fig 1] Evolucion mensual de emociones...")
fig, axes = plt.subplots(2, 1, figsize=(7.16, 6), sharex=True)  # double column width

for emo in ['Ira', 'Miedo', 'Alegria', 'Tristeza']:
    if emo in df_global.columns:
        axes[0].plot(df_global['Fecha'], df_global[emo], label=tr_emo(emo),
                    color=COLORS.get(emo, 'gray'), lw=1, alpha=0.7)

axes[0].axvline(BREAK_COVID, color='red', ls='--', lw=1, alpha=0.6, label='COVID-19')
axes[0].axvline(pd.Timestamp('2022-02-01'), color='orange', ls='--', lw=1, alpha=0.6, label='Ukraine')
axes[0].set_ylabel('Mean monthly probability')
axes[0].set_title('(a) Monthly emotion means')
axes[0].legend(ncol=6, loc='upper left', fontsize=7)

for emo in ['Ira', 'Miedo', 'Alegria', 'Tristeza']:
    if emo in df_global.columns:
        ma12 = df_global[emo].rolling(12, min_periods=6).mean()
        axes[1].plot(df_global['Fecha'], ma12, label=tr_emo(emo),
                    color=COLORS.get(emo, 'gray'), lw=1.5)

axes[1].axvline(BREAK_COVID, color='red', ls='--', lw=1, alpha=0.6)
axes[1].axvline(pd.Timestamp('2022-02-01'), color='orange', ls='--', lw=1, alpha=0.6)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('12-month moving average')
axes[1].set_title('(b) 12-month moving average')
axes[1].legend(ncol=4, loc='upper left', fontsize=7)

for ax in axes:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
save_fig(fig, 'fig_01_emotion_evolution')

# ═══ Fig 2: Heatmap emociones x categorias ═══
print("\n[Fig 2] Heatmap emociones x categorias...")
cats = sorted(df_by_cat['categoria'].unique())
heatmap_data = []
for cat in cats:
    row = {}
    for emo in EMOTION_VARS_ES:
        if emo in df_by_cat.columns:
            row[emo] = df_by_cat[df_by_cat['categoria'] == cat][emo].mean()
    row['Category'] = tr_cat(cat)
    heatmap_data.append(row)

df_heat = pd.DataFrame(heatmap_data).set_index('Category')
df_heat.columns = [tr_emo(c) for c in df_heat.columns]
fig, ax = plt.subplots(figsize=(7.16, 3.5))
sns.heatmap(df_heat, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=ax,
            linewidths=0.5, cbar_kws={'label': 'Mean probability'})
ax.set_title('Emotion profile by news category (2017-2023)')
plt.tight_layout()
save_fig(fig, 'fig_02_heatmap_categories')

# ═══ Fig 3: ARIMAX Global observada vs predicha ═══
print("\n[Fig 3] ARIMAX Global (se genera post-ejecucion del script 03)...")
arimax_json = BASE_DIR / 'article_outputs' / 'arimax' / 'all_results.json'
if arimax_json.exists():
    with open(arimax_json, 'r', encoding='utf-8') as f:
        arimax_results = json.load(f)

    # Find global model figure
    global_fig_path = None
    for r in arimax_results:
        if 'Global' in r.get('variable', ''):
            global_fig_path = r.get('figure', '')
            break

    if global_fig_path and os.path.exists(global_fig_path):
        print(f"   ARIMAX global figure already exists: {global_fig_path}")
    else:
        print("   ARIMAX figure not found, will be generated by script 03")
else:
    print("   ARIMAX results not yet available. Run script 03 first.")

# ═══ Fig 4: IRF ortogonal (paneles seleccionados) ═══
print("\n[Fig 4] IRF ortogonal...")
irf_json = VAR_DIR / 'irf_data.json'
if irf_json.exists():
    with open(irf_json, 'r', encoding='utf-8') as f:
        irf_data = json.load(f)

    # Select key IRFs: emotion -> IPC and IPC -> emotion
    ipc_col_name = 'IPC_ECOICOP Índice General'
    emotion_names = ['Ira', 'Miedo', 'Tristeza', 'Alegria']

    fig, axes = plt.subplots(2, 4, figsize=(7.16, 4))

    for j, emo in enumerate(emotion_names):
        key_to = f"{emo} -> {ipc_col_name}"
        key_from = f"{ipc_col_name} -> {emo}"

        if key_to in irf_data:
            vals = irf_data[key_to]
            axes[0, j].plot(range(len(vals)), vals, color=COLORS.get(emo, 'gray'), lw=1.5)
            axes[0, j].axhline(0, color='black', ls='-', lw=0.5)
            axes[0, j].set_title(f'{tr_emo(emo)} -> CPI', fontsize=8)
            if j == 0:
                axes[0, j].set_ylabel('Response')

        if key_from in irf_data:
            vals = irf_data[key_from]
            axes[1, j].plot(range(len(vals)), vals, color=COLORS.get(emo, 'gray'), lw=1.5)
            axes[1, j].axhline(0, color='black', ls='-', lw=0.5)
            axes[1, j].set_title(f'CPI -> {tr_emo(emo)}', fontsize=8)
            axes[1, j].set_xlabel('Months')
            if j == 0:
                axes[1, j].set_ylabel('Response')

    plt.suptitle('Orthogonalized Impulse Response Functions', fontsize=10, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'fig_04_irf_selected')
else:
    print("   IRF data not yet available. Run script 04 first.")

# ═══ Fig 5: FEVD del IPC ═══
print("\n[Fig 5] FEVD del IPC...")
fevd_file = VAR_DIR / 'fevd_ipc.csv'
if fevd_file.exists():
    fevd_ipc = pd.read_csv(fevd_file, index_col=0)

    fig, ax = plt.subplots(figsize=(3.5, 3))  # single column
    bottom = np.zeros(len(fevd_ipc))

    # Tristeza tenia el mismo azul que el IPC: dos series indistinguibles.
    colors_fevd = {
        'IPC_ECOICOP Índice General': '#1f77b4',
        'Ira': '#d62728', 'Miedo': '#9467bd',
        'Tristeza': '#ff7f0e', 'Alegria': '#2ca02c'
    }

    for col in fevd_ipc.columns:
        label = 'CPI' if 'IPC' in col else tr_emo(col)
        color = colors_fevd.get(col, '#7f7f7f')
        ax.bar(fevd_ipc.index, fevd_ipc[col], bottom=bottom, label=label,
               color=color, alpha=0.8, width=0.8)
        bottom += fevd_ipc[col].values

    ax.set_xlabel('Forecast horizon (months)')
    ax.set_ylabel('Proportion of variance')
    ax.set_title('FEVD of CPI')
    ax.set_xticks(list(fevd_ipc.index))
    ax.legend(fontsize=6, loc='upper center', ncol=5, columnspacing=0.8,
              handlelength=1.0, frameon=False)
    ax.set_ylim(0, 1.20)
    plt.tight_layout()
    save_fig(fig, 'fig_05_fevd_ipc')
else:
    print("   FEVD data not yet available. Run script 04 first.")

# ═══ Fig 6: Series temporales con linea de quiebre COVID ═══
print("\n[Fig 6] Series temporales con quiebre COVID...")
fig, axes = plt.subplots(2, 1, figsize=(7.16, 5), sharex=True)

if 'IPC_ECOICOP Índice General' in df_global.columns:
    axes[0].plot(df_global['Fecha'], df_global['IPC_ECOICOP Índice General'],
                color='#1f77b4', lw=1.5)
    axes[0].axvline(BREAK_COVID, color='red', ls='--', lw=1.5, label='COVID-19')
    axes[0].axvline(pd.Timestamp('2022-02-01'), color='orange', ls='--', lw=1.5, label='Ukraine')
    axes[0].set_ylabel('CPI (General Index)')
    axes[0].set_title('(a) Consumer Price Index')
    axes[0].legend(fontsize=7)

if 'Negatividad' in df_global.columns:
    axes[1].plot(df_global['Fecha'], df_global['Negatividad'],
                color='darkred', lw=1.5)
    axes[1].axvline(BREAK_COVID, color='red', ls='--', lw=1.5)
    axes[1].axvline(pd.Timestamp('2022-02-01'), color='orange', ls='--', lw=1.5)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Negativity index')
    axes[1].set_title('(b) Composite negativity (Anger + Fear + Sadness)')

for ax in axes:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
save_fig(fig, 'fig_06_series_covid_break')

# ═══ Fig 7: Perfil emocional pre/COVID/post-COVID ═══
print("\n[Fig 7] Perfil emocional por periodo...")
periods = {
    'Pre-COVID\n(2017-01 to\n2020-02)': (df_global['Fecha'] < '2020-03-01'),
    'COVID\n(2020-03 to\n2021-12)': (df_global['Fecha'] >= '2020-03-01') & (df_global['Fecha'] < '2022-01-01'),
    'Post-COVID\n(2022-01 to\n2023-12)': (df_global['Fecha'] >= '2022-01-01')
}

fig, ax = plt.subplots(figsize=(3.5, 3))
x_pos = np.arange(4)  # 4 main emotions
width = 0.25
emo_labels = ['Ira', 'Miedo', 'Alegria', 'Tristeza']

for i, (period_name, mask) in enumerate(periods.items()):
    means = [df_global.loc[mask, emo].mean() if emo in df_global.columns else 0
             for emo in emo_labels]
    ax.bar(x_pos + i * width, means, width, label=period_name.replace('\n', ' '), alpha=0.8)

ax.set_xticks(x_pos + width)
ax.set_xticklabels(['Anger', 'Fear', 'Joy', 'Sadness'], fontsize=7)
ax.set_ylabel('Mean probability')
ax.set_title('Emotional profile by period')
ax.legend(fontsize=5, loc='upper right')
plt.tight_layout()
save_fig(fig, 'fig_07_emotional_profile_periods')

# ═══ Fig 8: Tendencias por categoria ═══
print("\n[Fig 8] Tendencias por categoria...")
cats = sorted(df_by_cat['categoria'].unique())
n_cats = min(len(cats), 6)
fig, axes = plt.subplots(2, 3, figsize=(7.16, 5), sharex=True)
axes_flat = axes.flatten()

for i, cat in enumerate(cats[:n_cats]):
    ax = axes_flat[i]
    cat_data = df_by_cat[df_by_cat['categoria'] == cat].sort_values('Fecha')
    for emo in ['Ira', 'Miedo', 'Alegria', 'Tristeza']:
        if emo in cat_data.columns:
            ax.plot(cat_data['Fecha'], cat_data[emo],
                    color=COLORS.get(emo, 'gray'), lw=0.8, alpha=0.8)
    ax.set_title(tr_cat(cat), fontsize=8, fontweight='bold')
    ax.axvline(BREAK_COVID, color='red', ls='--', lw=0.5, alpha=0.5)
    ax.tick_params(axis='x', rotation=45, labelsize=6)
    if i == 0:
        ax.legend(['Anger', 'Fear', 'Joy', 'Sadness'], fontsize=5, loc='upper left')

# Hide extra axes if < 6 categories
for i in range(n_cats, 6):
    axes_flat[i].set_visible(False)

plt.suptitle('Emotion trends by news category', fontsize=10, fontweight='bold')
plt.tight_layout()
save_fig(fig, 'fig_08_trends_by_category')

# Summary
print("\n" + "=" * 80)
figures = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
print(f"{len(figures)} FIGURAS GENERADAS EN {OUTPUT_DIR}/")
for f in sorted(figures):
    size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
    print(f"   - {f} ({size_kb:.0f} KB)")
print("=" * 80)
