"""Classical baselines: the make-or-break test for SLD.

Two statistics, same rolling-window / day-over-day-change construction as
``qgmrd.sld.sld_qfi_time_series``, but on the raw feature vectors directly,
no quantum embedding. If SLD does not beat these with non-overlapping
bootstrap CIs on Cohen's d, the "quantum" framing adds nothing measurable
over a classical distributional-shift detector on the same inputs. Two
baselines, not one, for the same reason the harness runs two null models
everywhere else: a negative result on a single arbitrary baseline is weak
evidence; failing both a parametric and a nonparametric one is not.

``classical_bures_time_series``: Bures-Wasserstein distance between N(mu1,
cov1) and N(mu2, cov2),

    d^2 = |mu1 - mu2|^2 + Tr(cov1 + cov2 - 2 sqrtm(sqrtm(cov1) cov2 sqrtm(cov1)))

the classical analog of the quantum Bures metric between density matrices:
same functional form, no quantum machinery. Covariance is Ledoit-Wolf
shrunk (data-driven intensity, no hand-picked constant) since the window is
only 20 rows over up to 8 dims and the raw empirical covariance is
near-singular.

``mmd_time_series``: kernel two-sample MMD^2 (RBF, median-heuristic
bandwidth) between the same two windows, fully nonparametric, no Gaussian
assumption at all.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm
from sklearn.covariance import LedoitWolf


def _shrunk_cov(window: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf-shrunk covariance, keeping a 20-samples/<=8-dims window
    comfortably invertible without a hand-picked shrinkage constant."""
    return LedoitWolf().fit(window).covariance_


def bures_wasserstein_distance(
    mu1: np.ndarray, cov1: np.ndarray, mu2: np.ndarray, cov2: np.ndarray
) -> float:
    """Bures-Wasserstein distance between N(mu1, cov1) and N(mu2, cov2)."""
    mean_term = float(np.sum((mu1 - mu2) ** 2))
    c1_sqrt = sqrtm(cov1).real
    cross = sqrtm(c1_sqrt @ cov2 @ c1_sqrt).real
    cov_term = np.trace(cov1) + np.trace(cov2) - 2 * np.trace(cross)
    return float(np.sqrt(max(mean_term + cov_term, 0.0)))


def classical_bures_time_series(X: np.ndarray, window: int = 20) -> np.ndarray:
    """Bures-Wasserstein distance between adjacent rolling windows of X.

    At each t (t >= window): today's trailing `window`-row Gaussian fit vs.
    yesterday's, the classical parallel to SLD's rho_t vs. rho_{t-1}. NaN
    before `window` history exists, matching ``sld_qfi_time_series``.
    """
    T = X.shape[0]
    out = np.full(T, np.nan)
    prev_mu, prev_cov = None, None
    for t in range(window - 1, T):
        w = X[t - window + 1 : t + 1]
        mu = w.mean(axis=0)
        cov = _shrunk_cov(w)
        if prev_mu is not None:
            out[t] = bures_wasserstein_distance(mu, cov, prev_mu, prev_cov)
        prev_mu, prev_cov = mu, cov
    return out


def _global_gamma(X: np.ndarray, causal_mask: np.ndarray = None,
                  sample_size: int = 400, seed: int = 0) -> float:
    """Median-heuristic RBF bandwidth from a random subsample of ALL pairwise
    distances in X, fixed once for the whole series.

    Recomputing the bandwidth fresh from each pair of adjacent (95%-
    overlapping) windows was tried first and is wrong: right when a real
    shift is happening, the local pooled sample already spans both regimes
    internally, which inflates the local median distance and suppresses the
    statistic's own sensitivity exactly when it matters. A bandwidth that
    characterizes the data's overall scale, not the local window pair,
    avoids that self-sabotage.

    ``causal_mask``, if given, restricts the subsample to those rows (e.g.
    the pre-cutoff rows a crisis's preprocessing was fit on). Without it,
    this samples from the WHOLE series, including rows after the crisis
    being scored, a real look-ahead leak in this one auxiliary parameter.
    Callers inside the causal pipeline must pass it.
    """
    pool = X if causal_mask is None else X[causal_mask]
    rng = np.random.default_rng(seed)
    n = len(pool)
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    sample = pool[idx]
    d2 = np.sum((sample[:, None, :] - sample[None, :, :]) ** 2, axis=-1)
    med = np.median(d2[np.triu_indices_from(d2, k=1)])
    return 1.0 / (2.0 * med) if med > 0 else 1.0


def _mmd2(a: np.ndarray, b: np.ndarray, gamma: float) -> float:
    """Biased (V-statistic) MMD^2 with an RBF kernel at a fixed bandwidth."""

    def kern(u, v):
        diff = u[:, None, :] - v[None, :, :]
        return np.exp(-gamma * np.sum(diff ** 2, axis=-1))

    kaa, kbb, kab = kern(a, a), kern(b, b), kern(a, b)
    return float(kaa.mean() + kbb.mean() - 2 * kab.mean())


def mmd_time_series(X: np.ndarray, window: int = 20,
                    causal_mask: np.ndarray = None) -> np.ndarray:
    """MMD^2 between adjacent rolling windows of X, the nonparametric
    counterpart to ``classical_bures_time_series`` (no Gaussian assumption).
    Same construction: today's trailing window vs. yesterday's, at a
    bandwidth fixed once from the whole series (see ``_global_gamma``).

    ``causal_mask`` restricts bandwidth calibration to pre-cutoff rows; see
    ``_global_gamma``. Pass the same pre-cutoff mask used to fit the rest of
    the causal pipeline (e.g. ``raw_channels``'s ``hmm_fit``).
    """
    gamma = _global_gamma(X, causal_mask=causal_mask)
    T = X.shape[0]
    out = np.full(T, np.nan)
    prev_w = None
    for t in range(window - 1, T):
        w = X[t - window + 1 : t + 1]
        if prev_w is not None:
            out[t] = max(_mmd2(w, prev_w, gamma), 0.0)
        prev_w = w
    return out
