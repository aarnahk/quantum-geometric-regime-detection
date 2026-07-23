"""Null-model tests (HANDOFF Sec. 8, item 1b): does each channel clear its own
noise floor?

Every Cohen's |d| in this repo has so far been reported against a *missing*
floor. In an autocorrelated, heavy-tailed score series, any contiguous window
separates from the rest by some amount for free -- persistence and fat tails
cluster extreme values, so a window "this long looks this separated" by chance
alone. The null distribution IS that free-lunch floor. A channel's real |d|
counts as signal only if it sits in the far right tail of ITS OWN floor.

Two nulls (Hammond Sec. 5.1), attacking the same question from opposite sides:

  (a) Random matched-length windows. Keep the SERIES fixed, move the WINDOW:
      draw a window of the same (non-NaN) length as the real crisis, placed
      elsewhere and non-overlapping the true crisis, |d| of window vs. rest.
      Tests whether the actual crisis DATES are special, or whether an
      arbitrary same-length window separates just as well. Preserves the
      series' real autocorrelation and marginals (genuine contiguous chunks).

  (b) Circular shift. Keep the WINDOW fixed, slide the SERIES underneath it by
      a random offset (wrap-around), |d| of the fixed crisis mask vs. rest.
      Tests whether the score's ALIGNMENT to the crisis is real or accidental.
      A circular shift is a bijection -- every value kept, only the phase
      changes -- so it preserves the autocorrelation exactly (up to one seam)
      while destroying correspondence to the crisis dates.

The floor is computed PER CHANNEL from that channel's own series: Hammond's
~0.53 came from his pipeline, and a floor depends on each channel's own
distributional shape (heavy-tailed series inflate it). We never import 0.53.

Reporting is pre-registered before any result exists (HANDOFF Sec. 2d): all
seven channels are reported on every window regardless of outcome, INCLUDING
the flagship SLD channel. A channel failing to clear its null -- "mathematically
correct, novel, decorrelated, and not demonstrated to detect above chance" --
is a legitimate finding and is written up as such, not softened or buried.

Four decisions fixed up front (see also the caveats printed by main()):

  1. MULTIPLE COMPARISONS. 7 channels x 3 crises x 2 nulls = 42 tests; ~2 clear
     at alpha=0.05 by chance. Primary family = COVID + 2022 (28 tests); China
     already carries an uninterpretability caveat (blind HMM control) and is
     reported SEPARATELY as exploratory, excluded from the FDR correction.
     Across the primary family we report raw p AND Benjamini-Hochberg q-values,
     and state the expected false-positive count (28 * 0.05 ~ 1.4).

  2. CONSERVATIVE (NOT CLEAN) FLOOR. Random null windows sometimes land on other
     genuine crises (2008, 2011, 2018). Those are not null periods, so the floor
     is inflated -- but that is the SAFE direction (harder to clear). We keep
     those periods in; the floor is "harder to clear than a true null," not a
     clean null. Stated, not corrected.

  3. SHIFT-P RESOLUTION IS ILLUSORY. Adjacent offsets on an autocorrelated
     series are near-duplicate draws, so M shifts is far fewer than M
     independent samples. We estimate N_eff = M / tau_int (integrated
     autocorrelation time, Geyer initial-positive-sequence) and FLOOR the shift
     p-value at 1/(N_eff+1) rather than 1/(N_draws+1).

  4. FOLDING CAVEAT. |d| discards direction, so a channel moving the WRONG way
     in a crisis still registers as detecting. We keep |d| for consistency with
     the rest of the repo, print the real effect's SIGN ('dir') for
     transparency, and flag it. The null test itself is on |d| vs. |d|.

Runs on the CAUSAL (past-fit) z-scored series -- the same per-crisis
preprocessing as scripts/causal_eval.py (HANDOFF Sec. 7). The nulls are pure
score-series transforms: no re-embedding, no re-fitting of the pipeline.

    python scripts/null_model.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from causal_eval import (  # noqa: E402  (shared pipeline pieces)
    CRISES,
    CUTOFF_BUFFER_DAYS,
    MIN_PRECUTOFF_ROWS,
    N,
    SEED,
    cohens_d,
    raw_channels,
    zscored,
)
from qgmrd.data import load_prices  # noqa: E402
from qgmrd.features import build_features  # noqa: E402
from qgmrd.operators import random_hermitian_operators  # noqa: E402

N_WINDOW_DRAWS = 5000        # random-window null (a) placements
PRIMARY_CRISES = {"COVID 2020", "Rate Hikes 2022"}   # FDR family; China is exploratory
FDR_ALPHA = 0.05


def causal_channels_for_crisis(features, ops, returns, idx, start):
    """Causal (past-fit) z-scored channels for one crisis.

    Fits scaler + PCA on rows strictly before ``crisis_start - buffer`` only,
    transforms the full timeline through them (same protocol as causal_eval),
    and returns the seven causal z-scored channel series plus the pre-cutoff
    row count.
    """
    crisis_start = pd.Timestamp(start)
    cutoff = crisis_start - pd.tseries.offsets.BDay(CUTOFF_BUFFER_DAYS)
    pre = (idx < cutoff).values if hasattr(idx < cutoff, "values") else (idx < cutoff)
    n_pre = int(pre.sum())

    p = min(8, features.shape[1])
    sc = StandardScaler().fit(features.values[pre])
    pc = PCA(n_components=p, random_state=SEED).fit(sc.transform(features.values[pre]))
    Xp = normalize(pc.transform(sc.transform(features.values)))
    causal = zscored(raw_channels(Xp, ops, returns, hmm_fit=pre))
    return causal, n_pre


def integrated_autocorr_time(x: np.ndarray) -> tuple[float, float]:
    """(tau_int, N_eff) via Geyer's initial-positive-sequence estimator.

    tau_int = 1 + 2 * sum_{l>=1} rho(l), truncated at the first non-positive
    autocorrelation (the standard guard against summing noise in the tail).
    N_eff = M / tau_int is how many effectively-independent samples the M
    circular shifts are worth -- adjacent shifts of an autocorrelated series
    are near-duplicates, so the raw shift count overstates resolution.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if n < 3 or denom == 0.0:
        return 1.0, float(n)
    maxlag = min(n - 1, 2000)
    tau = 1.0
    for lag in range(1, maxlag + 1):
        rho = float(np.dot(x[: n - lag], x[lag:])) / denom
        if rho <= 0.0:
            break
        tau += 2.0 * rho
    tau = max(tau, 1.0)
    return tau, n / tau


def run_nulls(z: np.ndarray, crisis_mask: np.ndarray, n_draws: int, seed: int) -> dict:
    """Both nulls for one channel on one crisis window.

    Operates on the finite sub-series (leading causal-warmup NaNs removed) so
    the circular shift preserves the exact marginal. The crisis window is a
    contiguous, fully-finite block there, so its non-NaN length L is the
    matched length for the random-window null.
    """
    z = np.asarray(z, dtype=float)
    finite = ~np.isnan(z)
    zf = z[finite]
    mf = np.asarray(crisis_mask, dtype=bool)[finite]
    M = len(zf)

    pos = np.where(mf)[0]
    lo, hi = int(pos.min()), int(pos.max())
    L = int(mf.sum())

    real_signed = cohens_d(zf[mf], zf[~mf])
    real = abs(real_signed)

    # ---- null (a): random matched-length windows, non-overlapping the crisis
    starts = np.array([s for s in range(0, M - L + 1)
                       if (s + L - 1 < lo) or (s > hi)], dtype=int)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(starts, size=n_draws, replace=True)
    da = np.empty(n_draws)
    for i, s in enumerate(chosen):
        m = np.zeros(M, dtype=bool)
        m[s : s + L] = True
        da[i] = abs(cohens_d(zf[m], zf[~m]))

    # ---- null (b): all M-1 circular shifts (deterministic), mask fixed
    db = np.empty(M - 1)
    for k in range(1, M):
        zr = np.roll(zf, k)
        db[k - 1] = abs(cohens_d(zr[mf], zr[~mf]))

    tau, n_eff = integrated_autocorr_time(zf)

    p_a = (1 + int(np.sum(da >= real))) / (1 + len(da))
    p_b_raw = (1 + int(np.sum(db >= real))) / (1 + len(db))
    p_b = max(p_b_raw, 1.0 / (n_eff + 1.0))   # floor at the effective resolution

    return {
        "real_signed": real_signed,
        "real": real,
        "L": L,
        "M": M,
        "a_med": float(np.median(da)),
        "a_lo": float(np.percentile(da, 2.5)),
        "a_hi": float(np.percentile(da, 97.5)),
        "a_pct": float((da < real).mean() * 100.0),
        "a_p": p_a,
        "b_med": float(np.median(db)),
        "b_lo": float(np.percentile(db, 2.5)),
        "b_hi": float(np.percentile(db, 97.5)),
        "b_pct": float((db < real).mean() * 100.0),
        "b_p_raw": p_b_raw,
        "b_p": p_b,
        "tau": tau,
        "n_eff": n_eff,
    }


def bh_fdr(pvals: list[float]) -> np.ndarray:
    """Benjamini-Hochberg q-values (monotone-enforced)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]   # enforce monotonicity from the top
    out = np.empty(n)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def main() -> None:
    prices = load_prices(("SPY", "DIA"))
    features = build_features(prices)
    idx = features.index
    ops = random_hermitian_operators(min(8, features.shape[1]), n=N, seed=SEED)
    returns = np.log(prices["SPY"]).diff().reindex(idx).values

    print("\n" + "=" * 78)
    print("Null-model tests (HANDOFF Sec. 8.1b) -- per-channel noise floor")
    print("Causal (past-fit) z-scored series, SPY/DIA 2005-present.")
    print("Null (a): random matched-length windows | Null (b): circular shift.")
    print("Floor is PER CHANNEL (never Hammond's 0.53). |d| is FOLDED: it")
    print("discards direction, so a wrong-way channel still scores -- 'dir'")
    print("shows the real effect's sign. Random windows may overlap 2008/2011/")
    print("2018, so the floor is 'harder to clear than a true null,' not clean.")
    print("LEGEND: 'pct' = percentile of the real |d| WITHIN its own null")
    print("(fraction of null draws BELOW it; high = strong). 'p' = fraction of")
    print("null draws AT OR ABOVE it (low = strong). They are complements,")
    print("pct ~ 100*(1-p) -- do not read a p-value as a percentile.")
    print("=" * 78)

    primary_p: list[float] = []
    primary_key: list[str] = []

    for name, start, end in CRISES:
        causal, n_pre = causal_channels_for_crisis(features, ops, returns, idx, start)
        if n_pre < MIN_PRECUTOFF_ROWS:
            print(f"\n### {name}  ({start} -> {end})")
            print(f"  SKIPPED: only {n_pre} pre-cutoff rows (< {MIN_PRECUTOFF_ROWS}).")
            continue

        crisis_mask = ((idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end)))
        crisis_mask = crisis_mask.values if hasattr(crisis_mask, "values") else crisis_mask

        tag = "PRIMARY" if name in PRIMARY_CRISES else "EXPLORATORY (excluded from FDR)"
        print(f"\n### {name}  ({start} -> {end})   [{tag}]")
        print(f"{'channel':<20}{'dir':>4}{'|d|':>7}"
              f"{'(a)med':>8}{'(a)95%CI':>16}{'(a)pct':>8}{'(a)p':>9}"
              f"{'(b)med':>8}{'(b)pct':>8}{'(b)p':>9}{'tau':>7}{'Neff':>7}")
        print("-" * 108)

        results = {ch: run_nulls(causal[ch], crisis_mask, N_WINDOW_DRAWS, SEED)
                   for ch in causal}
        for ch in sorted(results, key=lambda c: -results[c]["real"]):
            r = results[ch]
            direction = "+" if r["real_signed"] >= 0 else "-"
            ci = f"[{r['a_lo']:.2f},{r['a_hi']:.2f}]"
            print(f"{ch:<20}{direction:>4}{r['real']:>7.2f}"
                  f"{r['a_med']:>8.2f}{ci:>16}{r['a_pct']:>8.1f}{r['a_p']:>9.4f}"
                  f"{r['b_med']:>8.2f}{r['b_pct']:>8.1f}{r['b_p']:>9.4f}"
                  f"{r['tau']:>7.1f}{r['n_eff']:>7.0f}")

            if name in PRIMARY_CRISES:
                primary_p.extend([r["a_p"], r["b_p"]])
                primary_key.extend([f"{name}/{ch}/(a)", f"{name}/{ch}/(b)"])

    # ---- Benjamini-Hochberg across the primary family only ----
    print("\n" + "=" * 78)
    print(f"Benjamini-Hochberg FDR -- PRIMARY family (COVID + 2022), "
          f"{len(primary_p)} tests")
    print(f"Expected false positives at alpha={FDR_ALPHA}: "
          f"{len(primary_p) * FDR_ALPHA:.1f}. China 2015 is exploratory and "
          f"NOT in this family.")
    print("=" * 78)
    q = bh_fdr(primary_p)
    order = np.argsort(q)
    print(f"{'test':<34}{'raw p':>10}{'BH q':>10}{'q<0.05':>9}")
    print("-" * 63)
    for i in order:
        passes = "yes" if q[i] < FDR_ALPHA else "no"
        print(f"{primary_key[i]:<34}{primary_p[i]:>10.4f}{q[i]:>10.4f}{passes:>9}")
    n_pass = int((q < FDR_ALPHA).sum())
    print(f"\n{n_pass}/{len(primary_p)} primary tests survive BH-FDR at q<{FDR_ALPHA}.")
    print("China 2015 results above are exploratory only (blind HMM control; "
          "see README/causal_eval).")
    print()


if __name__ == "__main__":
    main()
