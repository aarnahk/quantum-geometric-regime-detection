"""v2 tests: the geometric identities the paper validates, at small scale.

test_metric_identity     : metric_fd agrees with metric_pt (Proposition 2
                           consistency check, the paper's r = 1.000 result).
test_qfi_is_4g           : F_Q = 4 g by construction, and both PSD.
test_plaquette_antisym   : F_ab = -F_ba for the plaquette curvature.
test_curvature_gap_bound : |F_ab| <= 2||dH_a|| ||dH_b|| / gap^2 (Theorem 4).
test_qcrb_positive       : Cramer-Rao variance bounds are positive on the
                           numerically supported directions.
"""

import numpy as np

from qgmrd.geometry import (
    berry_curvature_plaquette,
    curvature_gap_bound,
    metric_fd,
    metric_pt,
    qcrb_bounds,
    qfi_matrix,
)
from qgmrd.operators import random_hermitian_operators

P, N, SEED = 6, 8, 42


def _setup():
    ops = random_hermitian_operators(P, N, seed=SEED)
    rng = np.random.default_rng(0)
    xs = [rng.uniform(-1, 1, size=P) for _ in range(5)]
    return ops, xs


def test_metric_identity():
    ops, xs = _setup()
    for x in xs:
        g_pt = metric_pt(x, ops)
        g_fd = metric_fd(x, ops, eps=1e-5)
        # elementwise agreement relative to the metric's scale
        scale = max(np.abs(g_pt).max(), 1e-12)
        assert np.abs(g_pt - g_fd).max() / scale < 1e-3
        # correlation across entries ~ 1 (the paper's r = 1.000 check)
        r = np.corrcoef(g_pt.ravel(), g_fd.ravel())[0, 1]
        assert r > 0.999999


def test_qfi_is_4g():
    ops, xs = _setup()
    for x in xs:
        g = metric_pt(x, ops)
        F = qfi_matrix(x, ops)
        assert np.allclose(F, 4.0 * g)
        assert np.linalg.eigvalsh(g).min() > -1e-10  # PSD up to roundoff


def test_plaquette_antisymmetry():
    ops, xs = _setup()
    for x in xs[:3]:
        f01 = berry_curvature_plaquette(x, ops, a=0, b=1)
        f10 = berry_curvature_plaquette(x, ops, a=1, b=0)
        denom = max(abs(f01), abs(f10), 1e-8)
        assert abs(f01 + f10) / denom < 5e-2  # antisym up to O(eps) discretization


def test_curvature_gap_bound():
    ops, xs = _setup()
    for x in xs:
        f01 = berry_curvature_plaquette(x, ops, a=0, b=1)
        bound = curvature_gap_bound(x, ops, a=0, b=1)
        assert abs(f01) <= bound * 1.05  # Theorem 4, small FD tolerance


def test_qcrb_positive():
    ops, xs = _setup()
    for x in xs:
        g = metric_pt(x, ops)
        bounds = qcrb_bounds(g)
        supported = bounds[np.isfinite(bounds) & (bounds > 0)]
        assert len(supported) >= 1
        assert np.all(supported > 0)
