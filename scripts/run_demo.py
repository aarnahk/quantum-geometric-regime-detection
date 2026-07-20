"""Run the v0 pipeline end-to-end on synthetic data and print a results table.

    python -m scripts.run_demo          (from the repo root, with the package installed)
    python scripts/run_demo.py          (if running the file directly)

Swap ``synthetic_prices`` for ``load_yfinance`` to run on real SPY/DIA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qgmrd.baseline import hmm_high_variance_prob
from qgmrd.data import crisis_mask, synthetic_prices
from qgmrd.features import build_features
from qgmrd.pipeline import cohens_d, embed_series
from qgmrd.zscore import causal_zscore


def main() -> None:
    prices = synthetic_prices()
    features = build_features(prices)

    # geometric observables -> causal z-scores
    obs = embed_series(features, n=8, p=8, dim_a=2, seed=42)
    se_z = causal_zscore(obs["spectral_entropy"].values)
    rp_z = causal_zscore(obs["reduced_purity"].values)

    # HMM baseline on SPY returns, aligned to the feature index
    spy_ret = np.log(prices["SPY"]).diff().reindex(features.index).values
    spy_ret = np.nan_to_num(spy_ret)
    hmm_z = causal_zscore(hmm_high_variance_prob(spy_ret))

    mask = crisis_mask(features.index, prices)

    def score(z):
        return cohens_d(z[mask], z[~mask])

    rows = [
        ("Spectral entropy", "geometric", score(se_z)),
        ("Reduced purity", "geometric", abs(score(rp_z))),  # sign depends on basis
        ("Gaussian HMM (high-var prob)", "baseline", score(hmm_z)),
    ]

    print("\nCohen's d, crisis window vs. rest  (synthetic data, v0 offline)\n")
    print(f"{'method':<32}{'type':<12}{'|d|':>8}")
    print("-" * 52)
    for name, kind, d in sorted(rows, key=lambda r: -abs(r[2])):
        print(f"{name:<32}{kind:<12}{abs(d):>8.2f}")
    print()


if __name__ == "__main__":
    main()
