"""Causal expanding-window z-score (paper Algorithm 1).

Smooth the raw observable with a trailing window of length w, then z-score
each point against the mean and std of *strictly past* smoothed values. No
future data enters the normalization, so the score is deployable in real time.
"""

from __future__ import annotations

import numpy as np


def causal_zscore(raw: np.ndarray, w: int = 20, m: int = 60) -> np.ndarray:
    """Return the causal z-scored series (NaN for t < m).

    Parameters
    ----------
    raw : raw observable series, length T.
    w   : trailing smoothing window.
    m   : minimum history before scoring begins.
    """
    raw = np.asarray(raw, dtype=float)
    T = raw.shape[0]

    # trailing-mean smoothing
    s = np.empty(T)
    for t in range(T):
        lo = max(0, t - w + 1)
        window = raw[lo : t + 1]
        s[t] = np.nanmean(window) if np.isfinite(window).any() else np.nan

    z = np.full(T, np.nan)
    for t in range(m, T):
        past = s[:t]                      # past only, excludes current point
        mu = np.nanmean(past)
        sd = np.nanstd(past)
        if sd > 0:
            z[t] = (s[t] - mu) / sd
    return z
