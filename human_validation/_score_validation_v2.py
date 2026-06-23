# -*- coding: utf-8 -*-
"""Score validation v2 (run after A1/A2 fill the intensity templates).

Metrics (no sklearn/scipy dependency):
  PRIMARY  (P2d): Spearman rho between human intensity (0-4) and RoBERTa probability,
                  per core emotion and pooled.
  SECONDARY(dominant emotion): accuracy, Cohen's kappa, Gwet's AC1, PABAK
                  (AC1/PABAK are robust to the high-neutral prevalence).
  ROBUSTNESS (P2b): dominant-emotion agreement restricted to high-confidence
                  headlines (model max prob > THRESHOLD).
  INTER-RATER: A1 vs A2 on intensity (rho) and dominant (kappa/AC1).
Outputs: resultados_validacion_v2.json
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = Path(__file__).resolve().parent
KEY = HERE / '_answer_key_v2.csv'
FILES = {'A1': HERE / 'Anotacion_Humana_v2_A1_Millenial.xlsx',
         'A2': HERE / 'Anotacion_Humana_v2_A2_Gen Z.xlsx'}
CORE4 = ['anger', 'fear', 'joy', 'sadness']
CATS = ['anger', 'fear', 'joy', 'sadness', 'surprise', 'disgust', 'neutral']
INT_COLS = {'anger': 'Ira (0-4)', 'fear': 'Miedo (0-4)',
            'joy': 'Alegría (0-4)', 'sadness': 'Tristeza (0-4)'}
THRESHOLD = 0.40

def rankavg(x):
    x = np.asarray(x, float); order = x.argsort()
    r = np.empty(len(x)); r[order] = np.arange(len(x))
    # average ties
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); starts = csum - cnt
    avg = (starts + csum - 1) / 2.0
    return avg[inv] + 1

def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None, int(len(a))
    ra, rb = rankavg(a), rankavg(b)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    return rho, int(len(a))

def cohen_kappa(y1, y2, labels):
    idx = {l: i for i, l in enumerate(labels)}; q = len(labels); n = len(y1)
    cm = np.zeros((q, q))
    for a, b in zip(y1, y2): cm[idx[a], idx[b]] += 1
    po = np.trace(cm) / n
    pe = float((cm.sum(1) / n * (cm.sum(0) / n)).sum())
    return (po - pe) / (1 - pe) if (1 - pe) > 1e-12 else float('nan'), po

def gwet_ac1(y1, y2, labels):
    idx = {l: i for i, l in enumerate(labels)}; q = len(labels); n = len(y1)
    cm = np.zeros((q, q))
    for a, b in zip(y1, y2): cm[idx[a], idx[b]] += 1
    po = np.trace(cm) / n
    pk = (cm.sum(1) + cm.sum(0)) / (2 * n)            # avg marginal proportion
    pe = float((pk * (1 - pk)).sum()) / (q - 1)
    return (po - pe) / (1 - pe) if (1 - pe) > 1e-12 else float('nan')

def pabak(po, q):
    return (po - 1.0 / q) / (1 - 1.0 / q)

key = pd.read_csv(KEY)
results = {'threshold': THRESHOLD, 'annotators': {}, 'inter_rater': None}
ann = {}
for tag, f in FILES.items():
    if not f.exists():
        print(f'[skip] {f.name} missing'); continue
    a = pd.read_excel(f, sheet_name='Titulares')
    int_done = a[list(INT_COLS.values())].notna().all(axis=1)
    dom_done = a['Emoción dominante'].notna()
    if int_done.sum() == 0 and dom_done.sum() == 0:
        print(f'[skip] {tag}: not annotated yet'); continue
    ann[tag] = a
    m = a.merge(key, on='ID', how='inner')

    # PRIMARY: intensity vs probability
    rhos = {}
    pooled_h, pooled_p = [], []
    for emo, col in INT_COLS.items():
        h = pd.to_numeric(m[col], errors='coerce').values
        p = m[emo].values  # roberta prob for that emotion
        rho, npair = spearman(h, p)
        rhos[emo] = {'rho': rho, 'n': npair}
        ok = ~np.isnan(h)
        pooled_h += list(h[ok]); pooled_p += list(p[ok])
    rho_pool, n_pool = spearman(pooled_h, pooled_p)

    # SECONDARY: dominant emotion
    d = m[m['Emoción dominante'].astype(str).str.lower().isin(CATS)].copy()
    sec = None
    if len(d) > 5:
        h = d['Emoción dominante'].astype(str).str.lower().tolist()
        mod = d['dominant'].astype(str).str.lower().tolist()
        k, po = cohen_kappa(h, mod, CATS)
        sec = {'n': len(d), 'accuracy': float(po), 'cohen_kappa': float(k),
               'gwet_ac1': float(gwet_ac1(h, mod, CATS)), 'pabak': float(pabak(po, len(CATS)))}
        # ROBUSTNESS: high-confidence subset
        maxp = d[['anger', 'fear', 'joy', 'sadness', 'surprise', 'disgust', 'others']].max(axis=1)
        hc = d[maxp > THRESHOLD]
        if len(hc) > 5:
            hh = hc['Emoción dominante'].astype(str).str.lower().tolist()
            mm = hc['dominant'].astype(str).str.lower().tolist()
            kk, poo = cohen_kappa(hh, mm, CATS)
            sec['high_confidence'] = {'n': len(hc), 'coverage': round(len(hc) / len(d), 3),
                                      'accuracy': float(poo), 'cohen_kappa': float(kk),
                                      'gwet_ac1': float(gwet_ac1(hh, mm, CATS))}
    results['annotators'][tag] = {'intensity_spearman': rhos,
                                   'intensity_spearman_pooled': {'rho': rho_pool, 'n': n_pool},
                                   'dominant': sec}
    print(f'[{tag}] pooled rho={rho_pool}  dominant={sec}')

# INTER-RATER
if 'A1' in ann and 'A2' in ann:
    a1 = ann['A1'].merge(ann['A2'], on='ID', suffixes=('_1', '_2'))
    ir = {'intensity': {}, 'dominant': None}
    p1, p2 = [], []
    for emo, col in INT_COLS.items():
        x = pd.to_numeric(a1[f'{col}_1'], errors='coerce').values
        y = pd.to_numeric(a1[f'{col}_2'], errors='coerce').values
        rho, npair = spearman(x, y); ir['intensity'][emo] = {'rho': rho, 'n': npair}
        ok = ~(np.isnan(x) | np.isnan(y)); p1 += list(x[ok]); p2 += list(y[ok])
    ir['intensity_pooled'] = dict(zip(['rho', 'n'], spearman(p1, p2)))
    d = a1[a1['Emoción dominante_1'].astype(str).str.lower().isin(CATS)
           & a1['Emoción dominante_2'].astype(str).str.lower().isin(CATS)]
    if len(d) > 5:
        h1 = d['Emoción dominante_1'].astype(str).str.lower().tolist()
        h2 = d['Emoción dominante_2'].astype(str).str.lower().tolist()
        k, po = cohen_kappa(h1, h2, CATS)
        ir['dominant'] = {'n': len(d), 'accuracy': float(po), 'cohen_kappa': float(k),
                          'gwet_ac1': float(gwet_ac1(h1, h2, CATS))}
    results['inter_rater'] = ir
    print(f'[inter-rater] {ir}')

if not results['annotators']:
    print('\n[info] No annotations found yet — fill the A1/A2 templates and re-run.')
else:
    with open(HERE / 'resultados_validacion_v2.json', 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False, default=str)
    print('\n[saved] resultados_validacion_v2.json')
