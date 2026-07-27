"""Data sources.

synthetic_prices : a self-contained two-asset series with an injected
                   high-volatility, high-correlation crisis window, so the
                   pipeline can be smoke-tested with no network access.

load_prices     : THE loader for real data. Reads a pinned CSV snapshot so
                   every README number is reproducible; fetches once if the
                   snapshot is missing.

load_yfinance   : raw network fetch. Do not call directly from analysis
                   scripts -- results would drift between runs (see
                   ``load_prices``). Requires ``pip install yfinance``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Data is PINNED for reproducibility. Every real-data figure in the README was
# computed from this window and this snapshot. `end` is exclusive.
DATA_START = "2005-01-01"
DATA_END = "2026-07-01"
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "spy_dia_close.csv"

# Multi-asset panel (Task 3): a SECOND pinned snapshot, added alongside
# the SPY/DIA one (never overwriting it, so existing results still reproduce).
# UUP inception is 2007-02, so an all-columns frame starts ~2007 and the panel
# drops the 2007 Quant Meltdown -- pre-registered as a 14-crisis panel.
MULTI_ASSET_TICKERS = ("SPY", "TLT", "UUP", "GLD")
MULTI_ASSET_CACHE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "multi_asset_close.csv"
)


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


def load_yfinance(tickers=("SPY", "DIA"), start=DATA_START, end=DATA_END):
    """Raw network fetch. Returns a close-price frame.

    ``auto_adjust=True`` is passed EXPLICITLY rather than left to the yfinance
    default, which has changed across versions -- an implicit default would
    silently alter every number in this repo on a dependency bump.
    """
    import yfinance as yf  # imported lazily; not needed for the synthetic demo

    data = yf.download(list(tickers), start=start, end=end,
                       auto_adjust=True, progress=False)
    close = data["Close"][list(tickers)].dropna()
    close.index.name = "Date"
    return close


def load_prices(tickers=("SPY", "DIA"), start=DATA_START, end=DATA_END,
                refresh=False):
    """Pinned, reproducible price loader -- use this, not ``load_yfinance``.

    Reads the committed CSV snapshot at ``CACHE_PATH`` if present; otherwise
    fetches once and writes it. Every real-data number in the README comes from
    this snapshot.

    Why a snapshot and not just a pinned ``end`` date: yfinance returns
    dividend/split-adjusted closes, so each new distribution retroactively
    rescales the ENTIRE price history. A pinned end date freezes the last row
    but not the earlier ones, and Cohen's d compares a crisis window against
    every other day -- so unpinned data quietly shifts every d, correlation,
    and null floor in the repo between runs. Only a snapshot fixes that.

    Pass ``refresh=True`` to deliberately re-pull and overwrite the snapshot
    (expect numbers to move; re-run every script and update the README).
    """
    tickers = list(tickers)
    if CACHE_PATH.exists() and not refresh:
        close = pd.read_csv(CACHE_PATH, index_col="Date", parse_dates=True)
        missing = [t for t in tickers if t not in close.columns]
        if missing:
            raise KeyError(f"{CACHE_PATH.name} lacks {missing}; "
                           f"re-run with refresh=True")
        close = close[tickers]
    else:
        close = load_yfinance(tickers, start=start, end=end)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        close.to_csv(CACHE_PATH)

    return close.loc[(close.index >= pd.Timestamp(start))
                     & (close.index < pd.Timestamp(end))]


def load_multi_asset_prices(tickers=MULTI_ASSET_TICKERS, start=DATA_START,
                            end=DATA_END, refresh=False):
    """Pinned loader for the multi-asset panel snapshot (Task 3).

    Same reproducibility contract as ``load_prices`` -- reads the committed
    ``multi_asset_close.csv`` if present, else fetches once and writes it. Kept
    separate from ``load_prices`` so the SPY/DIA snapshot is never disturbed.

    The fetch drops any date lacking all ``tickers`` (via ``load_yfinance``), so
    the frame begins when every asset exists (UUP inception, ~2007-02).
    """
    tickers = list(tickers)
    if MULTI_ASSET_CACHE_PATH.exists() and not refresh:
        close = pd.read_csv(MULTI_ASSET_CACHE_PATH, index_col="Date", parse_dates=True)
        missing = [t for t in tickers if t not in close.columns]
        if missing:
            raise KeyError(f"{MULTI_ASSET_CACHE_PATH.name} lacks {missing}; "
                           f"re-run with refresh=True")
        close = close[tickers]
    else:
        close = load_yfinance(tickers, start=start, end=end)
        MULTI_ASSET_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        close.to_csv(MULTI_ASSET_CACHE_PATH)

    return close.loc[(close.index >= pd.Timestamp(start))
                     & (close.index < pd.Timestamp(end))]
