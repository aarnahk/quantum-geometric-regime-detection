"""Task 1: pairwise correlation of the eleven causal z-scored detectors on SPY/DIA.

Answers Open Question 1 -- is the SLD mixed-state QFI channel redundant with the
pure-state channels, or genuinely decorrelated? Reports the SLD row explicitly.
Result and full matrix: see README.

    python scripts/channel_correlations.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

from qgmrd.baseline import hmm_high_variance_prob
from qgmrd.channels import (
    berry_phase_rate_series,
    ground_energy_series,
    qfi_logdet_series,
)
from qgmrd.data import load_prices
from qgmrd.features import build_features
from qgmrd.operators import random_hermitian_operators
from qgmrd.pipeline import embed_series
from qgmrd.sld import sld_qfi_time_series
from qgmrd.zscore import causal_zscore

DIVERGE_THRESHOLD = 0.10  # |Pearson - Spearman| above this is flagged


def main() -> None:
    prices = load_prices(("SPY", "DIA"))
    features = build_features(prices)

    # shared preprocessing (v0-style global fit; Task 2 makes this causal)
    X = StandardScaler().fit_transform(features.values)
    p = min(8, X.shape[1])
    Xp = normalize(PCA(n_components=p, random_state=42).fit_transform(X))
    ops = random_hermitian_operators(p, n=8, seed=42)

    obs = embed_series(features, n=8, p=p, dim_a=2, seed=42)

    spy_ret = np.log(prices["SPY"]).diff().reindex(features.index).values

    channels = {
        "spectral_entropy": causal_zscore(obs["spectral_entropy"].values),
        "reduced_purity": causal_zscore(obs["reduced_purity"].values),
        "ground_energy_E0": causal_zscore(ground_energy_series(Xp, ops)),
        "berry_phase_rate": causal_zscore(berry_phase_rate_series(Xp, ops)),
        "qfi_logdet": causal_zscore(qfi_logdet_series(Xp, ops)),
        "sld_qfi_w20": causal_zscore(sld_qfi_time_series(Xp, ops, window=20)),
        "hmm_high_var_prob": causal_zscore(hmm_high_variance_prob(np.nan_to_num(spy_ret))),
    }
    df = pd.DataFrame(channels, index=features.index)

    pearson = df.corr(method="pearson")
    spearman = df.corr(method="spearman")

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda v: f"{v:6.3f}")

    print("\nPearson correlation, causal z-scored channels (SPY/DIA, full series)\n")
    print(pearson)

    print("\nSpearman correlation (rank-based, robust to fat-tailed crisis days)\n")
    print(spearman)

    print("\nSLD row (Pearson | Spearman | divergence)\n")
    for col in df.columns:
        if col == "sld_qfi_w20":
            continue
        pr = pearson.loc["sld_qfi_w20", col]
        sr = spearman.loc["sld_qfi_w20", col]
        flag = " <-- Pearson/Spearman diverge" if abs(pr - sr) > DIVERGE_THRESHOLD else ""
        print(f"  sld_qfi_w20 vs {col:<20} pearson={pr:6.3f}  spearman={sr:6.3f}{flag}")

    print("\nAll pairs where Pearson and Spearman diverge by more than "
          f"{DIVERGE_THRESHOLD}\n")
    cols = df.columns
    any_diverge = False
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pr = pearson.iloc[i, j]
            sr = spearman.iloc[i, j]
            if abs(pr - sr) > DIVERGE_THRESHOLD:
                any_diverge = True
                print(f"  {cols[i]:<20} vs {cols[j]:<20} pearson={pr:6.3f}  spearman={sr:6.3f}")
    if not any_diverge:
        print("  (none)")
    print()


if __name__ == "__main__":
    main()
