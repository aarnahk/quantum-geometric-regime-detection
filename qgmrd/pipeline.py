"""End-to-end embedding: feature frame -> observable time series.

v0 uses a single global fit of the scaler and PCA (offline). The paper's
causal per-crisis preprocessing (fit only on pre-crisis rows) is implemented
in scripts/causal_eval.py (Task 2); see README. Labelling this clearly avoids
overclaiming the current results as causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

from .embedding import ground_state
from .observables import reduced_purity, spectral_entropy
from .operators import random_hermitian_operators


def embed_series(
    features: pd.DataFrame,
    n: int = 8,
    p: int = 8,
    dim_a: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute spectral entropy and reduced purity at every timestep.

    Parameters
    ----------
    features : output of build_features.
    n : Hilbert dimension (must be divisible by dim_a).
    p : number of PCA components / operators.
    dim_a : subsystem-A dimension for the purity bipartition.
    seed : RNG seed for operator construction.

    Returns
    -------
    DataFrame indexed like ``features`` with columns
    ['spectral_entropy', 'reduced_purity'].
    """
    X = StandardScaler().fit_transform(features.values)
    p_eff = min(p, X.shape[1])
    Xp = PCA(n_components=p_eff, random_state=seed).fit_transform(X)
    Xp = normalize(Xp)                    # L2-normalize each row (paper Algo 3)

    ops = random_hermitian_operators(p_eff, n, seed=seed)

    T = Xp.shape[0]
    se = np.empty(T)
    rp = np.empty(T)
    for t in range(T):
        evals, psi = ground_state(Xp[t], ops)
        se[t] = spectral_entropy(evals)
        rp[t] = reduced_purity(psi, dim_a=dim_a)

    return pd.DataFrame(
        {"spectral_entropy": se, "reduced_purity": rp}, index=features.index
    )


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d between samples a and b (NaNs dropped)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(
        ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    )
    if sp == 0:
        return 0.0
    return float((a.mean() - b.mean()) / sp)
