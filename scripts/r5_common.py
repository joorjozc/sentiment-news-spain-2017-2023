"""Utilidades compartidas por los scripts de R5 (r5_var.py, r5_negativity.py)."""

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

B = 1999
SEED = 42


def gfevd(res, horizon=12):
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
