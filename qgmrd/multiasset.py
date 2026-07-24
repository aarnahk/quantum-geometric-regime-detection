"""Cross-asset aggregation for the multi-asset panel (HANDOFF Task 3).

With k assets there are k(k-1)/2 pairwise correlations; feeding all of them as
features is the scaling problem the aggregate exists to avoid. Two candidate
scalar summaries of "how coupled is the system right now" are provided, and the
choice between them is settled by the pre-registered degeneracy diagnostic in
``scripts/`` -- NOT by which one flatters the channels (HANDOFF Task 3):

``absorption_ratio_series``      : top-eigenvalue share lambda_1 / trace of the
                                   rolling correlation matrix (Absorption Ratio,
                                   Kritzman-Li-Page-Rigobon 2011). Measures how
                                   much cross-asset variance loads on a single
                                   factor -- the contagion signature. SIGN-
                                   INVARIANT (flipping a variable leaves the
                                   correlation matrix's eigenvalues unchanged).

``risk_aligned_mean_corr_series``: mean pairwise correlation AFTER applying the
                                   a-priori risk-direction sign map (flip the
                                   dollar). This is a RISK-ALIGNMENT measure, not
                                   a neutral correlation -- label it that way
                                   everywhere. The sign map is a modeling choice
                                   fixed a priori by asset class; gold and bonds
                                   do not stably sit on the risk-on side (2008
                                   Treasuries rallied as a haven; 2022 both fell
                                   with equities), so the fixed sign is wrong in
                                   some regimes by construction -- an accepted
                                   cost of an a-priori map, never tuned per crisis.

``top_eigvec_loading_series``    : squared loading of the leading eigenvector on
                                   one asset (SPY), the corroborating check for
                                   whether AR has collapsed into "the SPY factor".

All three read a LOG-RETURN frame (columns = assets) and share one rolling
correlation-matrix computation, so AR and MC cannot diverge on implementation.

WINDOW is fixed a priori at 60 trading days: a 4x4 correlation estimated from 20
daily observations is too noisy, and the Absorption-Ratio literature uses
multi-month estimation windows. Pinned, not tuned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Risk-direction sign map, fixed a priori by asset class (HANDOFF Task 3).
# Risk-on assets are the reference (+1); the dollar is the sole risk-off leg and
# is flipped (-1) so co-movement in a stress episode reads as positive
# correlation. AR ignores this map (sign-invariant); it affects only MC.
SIGN_MAP: dict[str, int] = {"SPY": 1, "GLD": 1, "TLT": 1, "UUP": -1}

CORR_WINDOW = 60  # a-priori, see module docstring


def rolling_corr_matrices(returns: pd.DataFrame, window: int = CORR_WINDOW) -> np.ndarray:
    """Rolling correlation matrices, shape ``(T, k, k)`` aligned to ``returns``.

    Days without a full, NaN-free lookback of ``window`` rows are left as NaN
    matrices. One computation feeds every aggregate below.
    """
    R = np.asarray(returns, dtype=float)
    T, k = R.shape
    C = np.full((T, k, k), np.nan)
    for t in range(window - 1, T):
        w = R[t - window + 1 : t + 1]
        if np.isnan(w).any():
            continue
        C[t] = np.corrcoef(w, rowvar=False)
    return C


def absorption_ratio_series(returns: pd.DataFrame, window: int = CORR_WINDOW) -> pd.Series:
    """Top-eigenvalue share lambda_1 / trace of the rolling correlation matrix.

    In [1/k, 1]: ~1/k when risk is spread across k independent factors, ->1 when
    one factor dominates (contagion). Sign-invariant.
    """
    C = rolling_corr_matrices(returns, window)
    ar = np.full(len(returns), np.nan)
    for t in range(len(C)):
        if np.isnan(C[t]).any():
            continue
        evals = np.linalg.eigvalsh(C[t])
        ar[t] = evals[-1] / evals.sum()
    return pd.Series(ar, index=returns.index, name="absorption_ratio")


def risk_aligned_mean_corr_series(
    returns: pd.DataFrame, window: int = CORR_WINDOW, sign_map: dict[str, int] = SIGN_MAP
) -> pd.Series:
    """Sign-aligned mean pairwise correlation -- a RISK-ALIGNMENT measure.

    Applies the a-priori sign map, then averages the (signed) off-diagonal
    correlations: mean_{i<j} s_i s_j C_ij. Flipping a variable's sign flips its
    correlations with the others, so this equals "flip the dollar, then take the
    mean pairwise correlation" -- signed, not absolute.
    """
    cols = list(returns.columns)
    missing = [c for c in cols if c not in sign_map]
    if missing:
        raise KeyError(f"sign_map lacks {missing}; have {list(sign_map)}")
    signs = np.array([sign_map[c] for c in cols], dtype=float)
    S = np.outer(signs, signs)
    k = len(cols)
    iu = np.triu_indices(k, k=1)
    C = rolling_corr_matrices(returns, window)
    mc = np.full(len(returns), np.nan)
    for t in range(len(C)):
        if np.isnan(C[t]).any():
            continue
        mc[t] = (S * C[t])[iu].mean()
    return pd.Series(mc, index=returns.index, name="risk_aligned_mean_corr")


def top_eigvec_loading_series(
    returns: pd.DataFrame, asset: str = "SPY", window: int = CORR_WINDOW
) -> pd.Series:
    """Squared loading of the leading eigenvector on ``asset``.

    Corroborating diagnostic for AR degeneracy: if the top factor is essentially
    "the SPY factor", its squared SPY loading sits near 1 and AR is an equity
    detector in disguise.
    """
    cols = list(returns.columns)
    j = cols.index(asset)
    C = rolling_corr_matrices(returns, window)
    load = np.full(len(returns), np.nan)
    for t in range(len(C)):
        if np.isnan(C[t]).any():
            continue
        _, evecs = np.linalg.eigh(C[t])
        load[t] = evecs[j, -1] ** 2
    return pd.Series(load, index=returns.index, name=f"top_eigvec_{asset}_load2")
