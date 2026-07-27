"""Multi-asset cross-asset aggregates (Task 3)."""

import numpy as np
import pandas as pd

from qgmrd.features import build_features_multiasset
from qgmrd.multiasset import (
    absorption_ratio_series,
    risk_aligned_mean_corr_series,
    top_eigvec_loading_series,
)

COLS = ["SPY", "TLT", "UUP", "GLD"]


def _prices(T: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = 0.01 * rng.normal(size=(T, 4))
    px = 100.0 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.date_range("2012-01-02", periods=T, freq="B")
    return pd.DataFrame(px, columns=COLS, index=idx)


def _frame(R: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2012-01-02", periods=len(R), freq="B")
    return pd.DataFrame(R, columns=COLS, index=idx)


def test_absorption_ratio_bounds():
    rng = np.random.default_rng(0)
    ar = absorption_ratio_series(_frame(rng.normal(size=(200, 4))), window=60).dropna()
    assert len(ar) > 0
    # lambda_1 / trace of a k x k correlation matrix lies in [1/k, 1].
    assert (ar >= 1 / 4 - 1e-9).all()
    assert (ar <= 1 + 1e-9).all()


def test_absorption_ratio_saturates_when_one_factor():
    # Four near-identical assets -> one factor -> AR -> 1.
    rng = np.random.default_rng(1)
    base = rng.normal(size=(200, 1))
    R = base + 1e-6 * rng.normal(size=(200, 4))
    ar = absorption_ratio_series(_frame(R), window=60).dropna()
    assert (ar > 0.99).all()


def test_absorption_ratio_sign_invariant():
    # Flipping any asset's sign leaves the correlation-matrix eigenvalues, hence
    # AR, unchanged -- the property that makes the sign map irrelevant to AR.
    rng = np.random.default_rng(2)
    R = rng.normal(size=(200, 4))
    ar = absorption_ratio_series(_frame(R), window=60)
    Rf = R.copy()
    Rf[:, 2] *= -1  # flip UUP
    ar_flipped = absorption_ratio_series(_frame(Rf), window=60)
    pd.testing.assert_series_equal(ar, ar_flipped, check_names=False)


def test_risk_aligned_mean_corr_realigns_the_dollar():
    # Construct a risk basket (SPY/TLT/GLD co-moving) with the dollar (UUP) moving
    # OPPOSITE it. The naive mean pairwise correlation (all +1 signs) is dragged
    # toward zero by the three negative dollar pairs; the risk-aligned aggregate
    # (flip UUP) turns those positive, so all six pairs align and MC is high.
    rng = np.random.default_rng(3)
    base = rng.normal(size=200)
    noise = 0.15 * rng.normal(size=(200, 4))
    R = np.column_stack([base, base, -base, base]) + noise  # UUP = -base
    frame = _frame(R)

    naive = risk_aligned_mean_corr_series(
        frame, window=60, sign_map={c: 1 for c in COLS}
    ).dropna()
    aligned = risk_aligned_mean_corr_series(frame, window=60).dropna()  # default map

    assert naive.median() < 0.2          # dollar pairs cancel the basket pairs
    assert aligned.median() > 0.7        # sign map realigns every pair
    assert aligned.median() > naive.median() + 0.5


def test_top_eigvec_loading_flags_spy_domination():
    # When SPY drives the common factor, its squared top-eigenvector loading is
    # large; an idiosyncratic SPY (independent of the rest) makes it small.
    rng = np.random.default_rng(4)
    common = rng.normal(size=200)
    # SPY loads on the common factor with the others -> high SPY loading.
    R_spy = np.column_stack([common, common, common, common]) + 0.1 * rng.normal(size=(200, 4))
    high = top_eigvec_loading_series(_frame(R_spy), asset="SPY", window=60).dropna()
    assert high.median() > 0.2  # ~1/4 even-split floor; shared factor lifts it

    # SPY independent, the other three share a factor -> low SPY loading.
    R_iso = np.column_stack([
        rng.normal(size=200), common, common, common,
    ]) + 0.1 * rng.normal(size=(200, 4))
    low = top_eigvec_loading_series(_frame(R_iso), asset="SPY", window=60).dropna()
    assert low.median() < high.median()


def test_build_features_multiasset_curation():
    # Option-1 curation: ret + vol20 per asset, plus one aggregate = k*2 + 1.
    f = build_features_multiasset(_prices())
    expected = [f"{c}_{s}" for c in COLS for s in ("ret", "vol20")] + ["absorption_ratio"]
    assert list(f.columns) == expected
    assert f.shape[1] == 4 * 2 + 1 == 9
    assert int(f.isna().sum().sum()) == 0  # dropna leaves a clean frame


def test_build_features_multiasset_mc_variant():
    f = build_features_multiasset(_prices(), aggregate="MC")
    assert "risk_aligned_mean_corr" in f.columns
    assert "absorption_ratio" not in f.columns
