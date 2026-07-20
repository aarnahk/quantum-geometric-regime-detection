"""Smoke tests: the pipeline runs and observables are well-formed."""

import numpy as np

from qgmrd import (
    build_features,
    causal_zscore,
    embed_series,
    ground_state,
    random_hermitian_operators,
    reduced_purity,
    spectral_entropy,
    synthetic_prices,
)


def test_operators_hermitian():
    ops = random_hermitian_operators(p=5, n=8, seed=42)
    assert ops.shape == (5, 8, 8)
    for A in ops:
        assert np.allclose(A, A.conj().T)


def test_ground_state_normalized():
    ops = random_hermitian_operators(p=8, n=8, seed=42)
    x = np.linspace(-1, 1, 8)
    evals, psi = ground_state(x, ops)
    assert np.isclose(np.linalg.norm(psi), 1.0)
    assert np.all(np.diff(evals) >= -1e-9)  # ascending


def test_observables_ranges():
    ops = random_hermitian_operators(p=8, n=8, seed=42)
    x = np.linspace(-1, 1, 8)
    evals, psi = ground_state(x, ops)
    s = spectral_entropy(evals)
    pur = reduced_purity(psi, dim_a=2)
    assert s >= 0.0
    assert 0.0 < pur <= 1.0 + 1e-9


def test_pipeline_runs():
    prices = synthetic_prices(T=400, crisis=(200, 260))
    feats = build_features(prices)
    obs = embed_series(feats, n=8, p=8)
    assert len(obs) == len(feats)
    assert np.isfinite(obs.values).all()
    z = causal_zscore(obs["spectral_entropy"].values)
    assert np.isnan(z[:60]).all()          # no scores before min history
    assert np.isfinite(z[60:]).any()
