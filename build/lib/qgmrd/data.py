"""Data sources.

synthetic_prices : a self-contained two-asset series with an injected
                   high-volatility, high-correlation crisis window, so the
                   pipeline can be smoke-tested with no network access.

load_yfinance   : one-line swap-in for real SPY/DIA data when you run locally.
                   Requires ``pip install yfinance`` and internet access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_prices(
    T: int = 2000,
    crisis: tuple[int, int] = (1000, 1120),
    seed: int = 1,
) -> pd.DataFrame:
    """Two correlated assets with a crisis window of elevated vol and corr."""
    rng = np.random.default_rng(seed)
    base_vol, base_corr, base_drift = 0.008, 0.30, 0.0003
    crisis_vol, crisis_corr, crisis_drift = 0.030, 0.85, -0.0020

    rets = np.empty((T, 2))
    for t in range(T):
        in_crisis = crisis[0] <= t < crisis[1]
        vol = crisis_vol if in_crisis else base_vol
        corr = crisis_corr if in_crisis else base_corr
        drift = crisis_drift if in_crisis else base_drift
        cov = np.array(
            [[vol**2, corr * vol * vol], [corr * vol * vol, vol**2]]
        )
        rets[t] = rng.multivariate_normal([drift, drift], cov)

    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.date_range("2015-01-01", periods=T, freq="B")
    return pd.DataFrame(prices, columns=["SPY", "DIA"], index=idx)


def crisis_mask(index: pd.DatetimeIndex, prices: pd.DataFrame,
                crisis: tuple[int, int] = (1000, 1120)) -> np.ndarray:
    """Boolean mask over ``index`` marking the synthetic crisis dates."""
    crisis_dates = prices.index[crisis[0] : crisis[1]]
    return index.isin(crisis_dates)


def load_yfinance(tickers=("SPY", "DIA"), start="2005-01-01", end=None):
    """Swap-in for real data (local use). Returns a close-price frame."""
    import yfinance as yf  # imported lazily; not needed for the synthetic demo

    data = yf.download(list(tickers), start=start, end=end, progress=False)
    close = data["Close"][list(tickers)].dropna()
    return close
