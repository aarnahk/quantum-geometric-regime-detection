"""v2 demo: all geometric channels vs the HMM baseline, plus the QFI report.

    python scripts/run_demo_v2.py

Prints (1) the Proposition 2 identity check (finite-difference metric vs
perturbation-theory metric) and a sample Cramer-Rao bound, then (2) the
Cohen's d table across all five channels and the HMM baseline on synthetic
regime-switch data. Swap in load_yfinance for real SPY/DIA.
"""

from __future__ import annotations

import numpy as np

from qgmrd.baseline import hmm_high_variance_prob
from qgmrd.channels import (
    berry_phase_rate_series,
    ground_energy_series,
    qfi_logdet_series,
)
from qgmrd.data import crisis_mask, load_yfinance, synthetic_prices
from qgmrd.features import build_features
from qgmrd.geometry import metric_fd, metric_pt, qcrb_bounds
from qgmrd.operators import random_hermitian_operators
from qgmrd.pipeline import cohens_d, embed_series
from qgmrd.sld import sld_qfi_time_series
from qgmrd.zscore import causal_zscore

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize


def main() -> None:
    prices = load_yfinance(("SPY","DIA"), start="2005-01-01")
    features = build_features(prices)

    # shared preprocessing (v0-style global fit; v1 makes this causal)
    X = StandardScaler().fit_transform(features.values)
    p = min(8, X.shape[1])
    Xp = normalize(PCA(n_components=p, random_state=42).fit_transform(X))
    ops = random_hermitian_operators(p, n=8, seed=42)

    # ---- Proposition 2 identity check at a sample point ----
    x0 = Xp[len(Xp) // 2]
    g_pt = metric_pt(x0, ops)
    g_fd = metric_fd(x0, ops, eps=1e-5)
    r = np.corrcoef(g_pt.ravel(), g_fd.ravel())[0, 1]
    rel = np.abs(g_pt - g_fd).max() / np.abs(g_pt).max()
    crb = qcrb_bounds(g_pt)
    print("\nProposition 2 check (4g = F_Q via two independent computations)")
    print(f"  corr(g_FD, g_PT) = {r:.9f}   max rel. error = {rel:.2e}")
    finite = crb[np.isfinite(crb) & (crb > 0)]
    print(f"  QCRB: min variance bound across supported directions = {finite.min():.3e}")

    # ---- channels ----
    obs = embed_series(features, n=8, p=p, dim_a=2, seed=42)
    se_z = causal_zscore(obs["spectral_entropy"].values)
    rp_z = causal_zscore(obs["reduced_purity"].values)
    e0_z = causal_zscore(ground_energy_series(Xp, ops))
    bpr_z = causal_zscore(berry_phase_rate_series(Xp, ops))
    qfi_z = causal_zscore(qfi_logdet_series(Xp, ops))
    sld_z = causal_zscore(sld_qfi_time_series(Xp, ops, window=20))

    spy_ret = np.log(prices["SPY"]).diff().reindex(features.index).values
    hmm_z = causal_zscore(hmm_high_variance_prob(np.nan_to_num(spy_ret)))

    import pandas as pd
    idx = features.index
    covid = (idx >= pd.Timestamp("2020-02-19")) & (idx <= pd.Timestamp("2020-04-30"))
    mask = covid

    def d(z):
        return abs(cohens_d(z[mask], z[~mask]))

    rows = [
        ("Ground energy E0", "geometric v2", d(e0_z)),
        ("Berry phase rate", "geometric v2", d(bpr_z)),
        ("QFI log-det", "geometric v2", d(qfi_z)),
        ("SLD mixed-state QFI (w=20)", "geometric v3", d(sld_z)),
        ("Spectral entropy", "geometric v0", d(se_z)),
        ("Reduced purity", "geometric v0", d(rp_z)),
        ("Gaussian HMM (high-var prob)", "baseline", d(hmm_z)),
    ]

    print("\nCohen's |d|, COVID window vs. rest (SPY/DIA, offline)\n")
    print(f"{'method':<32}{'type':<16}{'|d|':>8}")
    print("-" * 56)
    for name, kind, val in sorted(rows, key=lambda r: -r[2]):
        print(f"{name:<32}{kind:<16}{val:>8.2f}")
    print()


if __name__ == "__main__":
    main()
