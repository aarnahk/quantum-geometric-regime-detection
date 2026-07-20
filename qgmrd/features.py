"""Feature construction from a two-asset price frame.

Mirrors the paper's raw feature set at a v0 level: log returns, 5/20-day
rolling volatility, 5/20-day momentum, and a rolling cross-correlation
between the two assets. The paper enriches these further with rolling
statistics; that enrichment is a straightforward v1 extension.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
