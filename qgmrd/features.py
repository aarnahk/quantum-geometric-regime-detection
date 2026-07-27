"""Feature construction from a two-asset price frame.

Mirrors the paper's raw feature set at a v0 level: log returns, 5/20-day
rolling volatility, 5/20-day momentum, and a rolling cross-correlation
between the two assets. The paper enriches these further with rolling
statistics; that enrichment is a straightforward v1 extension.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qgmrd.multiasset import (
    CORR_WINDOW,
    absorption_ratio_series,
    risk_aligned_mean_corr_series,
)


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a feature DataFrame from a (T, 2) price frame.

    ``prices`` columns are treated as the two assets (e.g. SPY, DIA).
    Rows with NaNs from the rolling windows are dropped.
    """
    if prices.shape[1] != 2:
        raise ValueError("build_features expects exactly two price columns")

    logret = np.log(prices).diff()
    cols = list(prices.columns)
    feats: dict[str, pd.Series] = {}
    for c in cols:
        r = logret[c]
        feats[f"{c}_ret"] = r
        feats[f"{c}_vol5"] = r.rolling(5).std()
        feats[f"{c}_vol20"] = r.rolling(20).std()
        feats[f"{c}_mom5"] = r.rolling(5).mean()
        feats[f"{c}_mom20"] = r.rolling(20).mean()
    feats["xcorr20"] = logret[cols[0]].rolling(20).corr(logret[cols[1]])

    return pd.DataFrame(feats).dropna()


def build_features_multiasset(
    prices: pd.DataFrame, aggregate: str = "AR", corr_window: int = CORR_WINDOW
) -> pd.DataFrame:
    """k-asset feature frame for the multi-asset panel (Task 3).

    A SEPARATE builder from ``build_features`` so the validated 2-asset pipeline
    (and every SPY/DIA result) is untouched -- the versioned-break discipline.

    Curated per-asset features are **log return and 20-day volatility only** (the
    a-priori "option 1" curation), plus ONE cross-asset aggregate. For k assets
    that is ``k*2 + 1`` features -- 9 at k=4, near the validated pipeline's 11 and
    well under the p=8 PCA cap, so PCA keeps nearly everything. The curation is
    theoretical (drop the 5-day and momentum features as least regime-informative),
    fixed a priori, never tuned against |d|.

    ``aggregate`` is the cross-asset scalar chosen by the pre-registered
    degeneracy diagnostic (``scripts/multiasset_aggregate_diagnostic.py``):
    ``"AR"`` (absorption ratio, the selected default) or ``"MC"`` (sign-aligned
    mean pairwise correlation). The 5/20-day momentum and 5-day vol features of
    the 2-asset builder are intentionally omitted.
    """
    if prices.shape[1] < 2:
        raise ValueError("build_features_multiasset expects at least two assets")

    logret = np.log(prices).diff()
    feats: dict[str, pd.Series] = {}
    for c in prices.columns:
        r = logret[c]
        feats[f"{c}_ret"] = r
        feats[f"{c}_vol20"] = r.rolling(20).std()

    if aggregate == "AR":
        feats["absorption_ratio"] = absorption_ratio_series(logret, corr_window)
    elif aggregate == "MC":
        feats["risk_aligned_mean_corr"] = risk_aligned_mean_corr_series(logret, corr_window)
    else:
        raise ValueError(f"unknown aggregate {aggregate!r}; use 'AR' or 'MC'")

    return pd.DataFrame(feats).dropna()
