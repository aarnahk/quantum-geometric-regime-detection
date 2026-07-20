"""v3 tests for the SLD mixed-state QFI.

test_pure_state_limit : SLD-QFI of the pure embedded state w.r.t. a feature
                        direction equals 4 g_aa from the metric - the JCP
                        Eq. 8 machinery reproducing Proposition 2 of the
                        regime paper. The closure between the two papers.
test_sld_defining_eq  : L satisfies drho = (L rho + rho L)/2 on the support.
test_qfi_nonnegative  : F_Q = Tr(L^2 rho) >= 0 for mixed window states.
test_time_series_runs : the rolling-window time channel produces finite,
                        nonnegative scores after warmup.
"""

import numpy as np

from qgmrd.geometry import metric_pt
from qgmrd.operators import random_hermitian_operators
from qgmrd.sld import (
    sld_operator,
    sld_qfi,
    sld_qfi_feature,
    sld_qfi_time_series,
    window_density_matrix,
)

P, N, SEED = 6, 8, 42


def _setup():
    ops = random_hermitian_operators(P, N, seed=SEED)
    rng = np.random.default_rng(0)
    return ops, rng


def test_pure_state_limit():
    """SLD-QFI (JCP Eq. 8) reduces to 4 g_aa (regime-paper Prop. 2)."""
    ops, rng = _setup()
    for _ in range(4):
        x = rng.uniform(-1, 1, size=P)
        g = metric_pt(x, ops)
        for a in (0, 1):
            F_sld = sld_qfi_feature(x, ops, a=a, eps=1e-5)
            F_metric = 4.0 * g[a, a]
            denom = max(abs(F_metric), 1e-10)
            assert abs(F_sld - F_metric) / denom < 1e-3


def test_sld_defining_equation():
    """drho = (L rho + rho L)/2 must hold on the support of rho."""
    ops, rng = _setup()
    x = rng.uniform(-1, 1, size=(5, P))
    states = np.stack(
        [np.linalg.eigh(_ham(xi, ops))[1][:, 0] for xi in x]
    )
    rho = window_density_matrix(states[:4])
    rho2 = window_density_matrix(states[1:5])
    drho = rho2 - rho
    L = sld_operator(rho, drho)
    recon = 0.5 * (L @ rho + rho @ L)
    # project both sides onto the support of rho before comparing
    evals, U = np.linalg.eigh(rho)
    keep = evals > 1e-10
    Ps = U[:, keep] @ U[:, keep].conj().T
    lhs = Ps @ drho @ Ps
    rhs = Ps @ recon @ Ps
    assert np.abs(lhs - rhs).max() < 1e-8


def _ham(x, ops):
    from qgmrd.embedding import error_hamiltonian

    return error_hamiltonian(x, ops)


def test_qfi_nonnegative():
    ops, rng = _setup()
    x = rng.uniform(-1, 1, size=(6, P))
    states = np.stack(
        [np.linalg.eigh(_ham(xi, ops))[1][:, 0] for xi in x]
    )
    rho = window_density_matrix(states[:5])
    rho2 = window_density_matrix(states[1:6])
    F = sld_qfi(rho, rho2 - rho)
    assert F >= -1e-10


def test_time_series_runs():
    ops, rng = _setup()
    X = rng.uniform(-1, 1, size=(60, P))
    scores = sld_qfi_time_series(X, ops, window=10)
    assert np.isnan(scores[:9]).all()
    tail = scores[10:]
    assert np.isfinite(tail).all()
    assert (tail >= -1e-10).all()
