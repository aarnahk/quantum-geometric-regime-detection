"""Task 2: causal (past-fit) preprocessing evaluation.

Every prior number in this repo fits the StandardScaler and PCA on the *whole*
price history -- including the crisis being detected. That is an offline event
study, not causal detection: the preprocessing already "knows" where the crisis
is. This script removes exactly that look-ahead and measures how much signal
survives.

Crisis windows come from the shared registry in ``qgmrd/crises.py``: Hammond's
Table G.10 windows extended by +/-10 trading days (his Sec. 4.1 convention),
applied uniformly across this repo. An earlier ad-hoc COVID window was replaced
by that convention; every number this script prints is under the new one.

Protocol, per crisis:
  1. cutoff = extended_window_start - 10 business days.
  2. Fit StandardScaler + PCA on rows with index < cutoff ONLY. The 10-day
     buffer exists because features include 20-day rolling volatility; a strict
     same-day cutoff would let crisis vol leak backward through the rolling
     window. Ten business days exceeds the longest rolling window in play.
  3. .transform() the FULL timeline with those past-fit objects, so crisis days
     are projected into a frame defined entirely by pre-crisis normalcy.
  4. Recompute all seven channels + causal z-scores on that embedding.
  5. Score Cohen's |d|, crisis window vs. rest, and print offline vs. causal
     side by side. The GAP between the two columns is the result.

Operators are data-independent (seeded), so they need no causal handling -- the
same operators are used both ways (see README). The HMM baseline IS refit
causally here (on pre-cutoff returns only), so the baseline column is honest
end-to-end rather than secretly still global.

Scope note -- this closes Gap 1 (leaky preprocessing) only. Gap 2 remains open:
Cohen's d still compares the crisis window against ALL other days, including
future ones, so even the causal column is offline separability, not real-time
detection (see README). Do not read the causal |d| as a deployment number.

    python scripts/causal_eval.py
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

from qgmrd.baseline import hmm_high_variance_prob
from qgmrd.channels import (
    berry_phase_rate_series,
    ground_energy_series,
    qfi_logdet_series,
)
from qgmrd.crises import LEGACY_THREE, MIN_PRECUTOFF_ROWS
from qgmrd.crises import context as crisis_context
from qgmrd.crises import get as crisis_get
from qgmrd.data import load_prices
from qgmrd.embedding import ground_state
from qgmrd.features import build_features
from qgmrd.observables import reduced_purity, spectral_entropy
from qgmrd.operators import random_hermitian_operators
from qgmrd.sld import sld_qfi_time_series
from qgmrd.zscore import causal_zscore

# (name, start_month, end_month) from the shared registry. Windows are Hammond
# Table G.10 extended by +/-10 trading days -- see qgmrd/crises.py. These three
# are the crises Task 2 and the null models were built on; the full 15-crisis
# panel lives in scripts/multi_crisis_panel.py and uses the same registry.
CRISES = [crisis_get(name) for name in LEGACY_THREE]

N = 8
DIM_A = 2
SLD_W = 20
SEED = 42


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d between samples a and b (NaNs dropped)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return 0.0
    return float((a.mean() - b.mean()) / sp)


def raw_channels(Xp: np.ndarray, ops: np.ndarray, returns: np.ndarray,
                 hmm_fit: np.ndarray) -> dict[str, np.ndarray]:
    """All seven raw (pre-z-score) channel series from an embedding Xp.

    ``hmm_fit`` is a boolean mask selecting the rows the HMM may fit on. For
    the offline column it is all-True; for a causal column it is the pre-cutoff
    rows only. The HMM is fit on that slice, then predicts over the full series.
    """
    T = Xp.shape[0]
    se = np.empty(T)
    rp = np.empty(T)
    for t in range(T):
        evals, psi = ground_state(Xp[t], ops)
        se[t] = spectral_entropy(evals)
        rp[t] = reduced_purity(psi, dim_a=DIM_A)

    return {
        "spectral_entropy": se,
        "reduced_purity": rp,
        "ground_energy_E0": ground_energy_series(Xp, ops),
        "berry_phase_rate": berry_phase_rate_series(Xp, ops),
        "qfi_logdet": qfi_logdet_series(Xp, ops),
        "sld_qfi_w20": sld_qfi_time_series(Xp, ops, window=SLD_W),
        "hmm_high_var_prob": hmm_high_variance_prob_causal(returns, hmm_fit),
    }


def hmm_high_variance_prob_causal(returns: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    """HMM high-variance posterior, fit ONLY on returns[fit_mask].

    Reuses the module baseline when the mask is all rows (offline); otherwise
    fits on the pre-cutoff slice and predicts over the full series, matching
    the scaler/PCA causal treatment so the baseline column is not secretly
    global.
    """
    r = np.nan_to_num(np.asarray(returns, dtype=float))
    if fit_mask.all():
        return hmm_high_variance_prob(r)

    from hmmlearn.hmm import GaussianHMM

    model = GaussianHMM(n_components=2, covariance_type="diag",
                        n_iter=200, random_state=SEED)
    model.fit(r[fit_mask].reshape(-1, 1))
    high = int(np.argmax(model.covars_.reshape(2)))
    return model.predict_proba(r.reshape(-1, 1))[:, high]


def zscored(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: causal_zscore(v) for k, v in raw.items()}


def d_table(channels: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    return {k: abs(cohens_d(z[mask], z[~mask])) for k, z in channels.items()}


def fit_divergence(global_scaler: StandardScaler, global_pca: PCA,
                   causal_scaler: StandardScaler, causal_pca: PCA) -> dict[str, float]:
    """How far the past-fit scaler/PCA sits from the global fit.

    Turns the "with thousands of pre-cutoff rows the past-fit nearly coincides
    with the global fit" claim into measured numbers. NOTE these are the
    *preprocessing-parameter* divergences; whether they survive into the
    embedding is a separate question (see ``embedding_row_cosine`` -- the
    L2 normalize after PCA can absorb scale differences).

    Returns per-feature arrays (mean_rel, scale_rel) so the distribution and
    which features diverge is visible, not just the max, plus per-component
    |cos| between causal and global PCA axes (abs handles sign flips).
    """
    gm, cm = global_scaler.mean_, causal_scaler.mean_
    gs, cs = global_scaler.scale_, causal_scaler.scale_
    mean_rel = np.abs(cm - gm) / np.maximum(np.abs(gm), 1e-12)
    scale_rel = np.abs(cs - gs) / np.maximum(np.abs(gs), 1e-12)
    cos = np.abs(np.sum(global_pca.components_ * causal_pca.components_, axis=1)) / (
        np.linalg.norm(global_pca.components_, axis=1)
        * np.linalg.norm(causal_pca.components_, axis=1)
    )
    return {
        "mean_rel": mean_rel,
        "scale_rel": scale_rel,
        "max_mean_reldiff": float(mean_rel.max()),
        "max_scale_reldiff": float(scale_rel.max()),
        "min_component_cos": float(cos.min()),
        "component_cos": cos,
    }


def main() -> None:
    prices = load_prices(("SPY", "DIA"))
    features = build_features(prices)
    idx = features.index
    feat_names = list(features.columns)
    ops = random_hermitian_operators(min(8, features.shape[1]), n=N, seed=SEED)
    returns = np.log(prices["SPY"]).diff().reindex(idx).values

    # ---- offline embedding (global fit, all rows) ----
    scaler = StandardScaler().fit(features.values)
    p = min(8, features.shape[1])
    pca = PCA(n_components=p, random_state=SEED).fit(scaler.transform(features.values))
    Xp_offline = normalize(pca.transform(scaler.transform(features.values)))
    offline = zscored(raw_channels(Xp_offline, ops, returns,
                                   hmm_fit=np.ones(len(idx), dtype=bool)))

    print("\n" + "=" * 68)
    print("Task 2 -- causal (past-fit) preprocessing vs. offline event study")
    print("SPY/DIA, 2005-present. Gap between columns = size of the look-ahead.")
    print("Caveat: Gap 2 is still open -- Cohen's d compares the crisis window")
    print("against future days too, so even 'causal' here is offline")
    print("separability (leaky preprocessing removed), not real-time detection.")
    print("=" * 68)

    for name, start_month, end_month in CRISES:
        ctx = crisis_context(idx, start_month, end_month)
        mask, n_pre = ctx["mask"], ctx["n_pre"]
        window = f"{start_month}..{end_month} +/-10td"
        off_d = d_table(offline, mask)

        if n_pre < MIN_PRECUTOFF_ROWS:
            print(f"\n### {name}  ({window})")
            print(f"  SKIPPED: only {n_pre} pre-cutoff rows (< {MIN_PRECUTOFF_ROWS}).")
            continue

        pre, cutoff = ctx["pre"], ctx["cutoff"]

        # ---- causal embedding: fit on pre-cutoff rows, transform full ----
        sc = StandardScaler().fit(features.values[pre])
        pc = PCA(n_components=p, random_state=SEED).fit(sc.transform(features.values[pre]))
        Xp_causal = normalize(pc.transform(sc.transform(features.values)))
        causal = zscored(raw_channels(Xp_causal, ops, returns, hmm_fit=pre))
        cau_d = d_table(causal, mask)

        div = fit_divergence(scaler, pca, sc, pc)

        # Does the preprocessing divergence survive into the embedding? The L2
        # normalize() after PCA projects each row onto the unit sphere, which
        # can absorb scale differences. Signed row-wise cosine (NOT abs: a PCA
        # sign flip changes H(x) physically, so it is a real difference) is the
        # direct test of whether the two embeddings agree where channels read.
        row_cos = np.sum(Xp_offline * Xp_causal, axis=1)  # both rows unit-norm

        print(f"\n### {name}  ({window}: {ctx['start'].date()} -> "
              f"{ctx['end'].date()}, {ctx['n_window']} days)   "
              f"cutoff {cutoff.date()}, {n_pre} pre-cutoff rows")
        print(f"  preprocessing params: max |Δmean|/|mean| = {div['max_mean_reldiff']:.3f}, "
              f"max |Δscale|/|scale| = {div['max_scale_reldiff']:.3f}, "
              f"min |cos| PCA axes = {div['min_component_cos']:.3f}")
        print("  per-feature |Δscale|/|scale| (sorted): "
              + ", ".join(f"{feat_names[i]}={div['scale_rel'][i]:.2f}"
                          for i in np.argsort(-div["scale_rel"])))
        print(f"  scale-reldiff distribution: median={np.median(div['scale_rel']):.3f}, "
              f"mean={div['scale_rel'].mean():.3f}, "
              f"n>0.10={int((div['scale_rel'] > 0.10).sum())}/{len(div['scale_rel'])}")
        print(f"  EMBEDDING agreement (post-normalize, row-wise signed cosine): "
              f"mean={row_cos.mean():.4f}, min={row_cos.min():.4f}, "
              f"frac<0.99={float((row_cos < 0.99).mean()):.3f}")

        # Does the embedding perturbation reach the z-scored channel SERIES, or
        # only the daily embedding? Two very different conclusions if |d| barely
        # moves: high series-corr => channels are robust to the perturbation;
        # low series-corr with matching |d| => Cohen's d is too coarse to see
        # series-level change, a caveat on the metric itself (every repo number).
        series_corr = {}
        for ch in off_d:
            a, b = offline[ch], causal[ch]
            ok = ~np.isnan(a) & ~np.isnan(b)
            series_corr[ch] = (float(np.corrcoef(a[ok], b[ok])[0, 1])
                               if ok.sum() > 2 else float("nan"))

        print(f"{'channel':<22}{'offline |d|':>13}{'causal |d|':>13}{'delta':>10}"
              f"{'corr(z_off,z_caus)':>20}")
        print("-" * 78)
        for ch in sorted(off_d, key=lambda c: -off_d[c]):
            o, c = round(off_d[ch], 2), round(cau_d[ch], 2)  # delta from shown values
            print(f"{ch:<22}{o:>13.2f}{c:>13.2f}{c - o:>+10.2f}"
                  f"{series_corr[ch]:>20.3f}")
    print()


if __name__ == "__main__":
    main()
