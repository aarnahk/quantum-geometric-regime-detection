"""Null-model tests: does each channel clear its own noise floor?

In an autocorrelated, heavy-tailed series any contiguous window separates from
the rest by some amount for free; these two nulls measure that free lunch per
channel. (a) random matched-length windows -- are the crisis DATES special?
(b) circular shift -- is the ALIGNMENT to the crisis real? Runs on the causal
z-scored series, per-channel floors never imported, BH-FDR over a pre-specified
family, shift p-values floored at 1/(N_eff+1). Full results and caveats: README.

    python scripts/null_model.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from causal_eval import (  # noqa: E402  (shared pipeline pieces)
    CRISES,
    MIN_PRECUTOFF_ROWS,
    N,
    SEED,
    cohens_d,
    raw_channels,
    zscored,
)
from qgmrd.baseline import realized_vol_series  # noqa: E402
from qgmrd.crises import context as crisis_context  # noqa: E402
from qgmrd.data import load_prices  # noqa: E402
from qgmrd.features import build_features  # noqa: E402
from qgmrd.operators import random_hermitian_operators  # noqa: E402

N_WINDOW_DRAWS = 5000        # random-window null (a) placements
PRIMARY_CRISES = {"2020 COVID", "2022 Rate Hikes"}   # FDR family; China is exploratory
FDR_ALPHA = 0.05
CONTROL = "realized_vol_20d"  # positive control; reported, NOT in the FDR family


def causal_channels_for_crisis(features, ops, returns, ctx, rv=None):
    """Causal (past-fit) z-scored channels for one crisis, plus the control.

    Fits scaler + PCA on rows strictly before the crisis context's cutoff only,
    transforms the full timeline through them (same protocol as causal_eval),
    and returns the seven causal z-scored channel series.

    ``rv`` is the raw realized-vol control series. It is appended as an eighth
    channel and z-scored identically, but it is NOT part of the FDR family --
    it exists to answer whether this test can detect anything at all.
    """
    pre = ctx["pre"]
    p = min(8, features.shape[1])
    sc = StandardScaler().fit(features.values[pre])
    pc = PCA(n_components=p, random_state=SEED).fit(sc.transform(features.values[pre]))
    Xp = normalize(pc.transform(sc.transform(features.values)))
    raw = raw_channels(Xp, ops, returns, hmm_fit=pre)
    if rv is not None:
        raw[CONTROL] = rv          # needs no fitting -- same series every crisis
    return zscored(raw)


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
    rv = realized_vol_series(prices["SPY"]).reindex(idx).values

    print("\n" + "=" * 78)
    print("Null-model tests -- per-channel noise floor")
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
    control_by_crisis: dict[str, dict] = {}

    for name, start_month, end_month in CRISES:
        ctx = crisis_context(idx, start_month, end_month)
        if ctx["n_pre"] < MIN_PRECUTOFF_ROWS:
            print(f"\n### {name}  ({start_month}..{end_month} +/-10td)")
            print(f"  SKIPPED: only {ctx['n_pre']} pre-cutoff rows "
                  f"(< {MIN_PRECUTOFF_ROWS}).")
            continue

        causal = causal_channels_for_crisis(features, ops, returns, ctx, rv=rv)
        crisis_mask = ctx["mask"]

        tag = "PRIMARY" if name in PRIMARY_CRISES else "EXPLORATORY (excluded from FDR)"
        print(f"\n### {name}  ({start_month}..{end_month} +/-10td: "
              f"{ctx['start'].date()} -> {ctx['end'].date()}, "
              f"{ctx['n_window']} days)   [{tag}]")
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

            # The control is reported but NEVER enters the FDR family -- it is
            # not a hypothesis under test, it is the instrument check.
            if name in PRIMARY_CRISES and ch != CONTROL:
                primary_p.extend([r["a_p"], r["b_p"]])
                primary_key.extend([f"{name}/{ch}/(a)", f"{name}/{ch}/(b)"])

        control_by_crisis[name] = results

    # ---- positive control: can this test detect anything at all? ----
    print("\n" + "=" * 78)
    print("POSITIVE CONTROL -- does the SINGLE-CRISIS test have a working")
    print("instrument? 20-day realized volatility through the identical")
    print("downstream. It is reported here and EXCLUDED from the FDR family.")
    print("")
    print("Reading a control FAILURE requires separating two cases that license")
    print("OPPOSITE conclusions:")
    print("  (a) STATISTIC INVALID -- the test is broken; every channel's result")
    print("      on that window is uninterpretable.")
    print("  (b) CONTROL POORLY SUITED TO THIS WINDOW -- realized vol's own")
    print("      persistence (vol clustering) gives it a structurally high")
    print("      floor, and/or the crisis was a slow grind rather than a vol")
    print("      spike. The test is fine; the channels stand.")
    print("The tau/floor columns below are what separate them: a high floor")
    print("EXPLAINED BY a high tau is a channel property, not a test property.")
    print("=" * 78)
    print(f"\n{'crisis':<16}{'channel':<20}{'|d|':>7}{'(a)flr':>8}{'(b)flr':>8}"
          f"{'(a)pct':>8}{'(a)p':>9}{'(b)p':>9}{'tau':>8}{'Neff':>7}  clears?")
    print("-" * 100)
    for name in [c[0] for c in CRISES if c[0] in PRIMARY_CRISES]:
        res = control_by_crisis.get(name)
        if not res:
            continue
        for ch in sorted(res, key=lambda c: (c != CONTROL, -res[c]["real"])):
            r = res[ch]
            clears = "YES" if (r["a_p"] < 0.05 and r["b_p"] < 0.05) else "no"
            tag = " <-- CONTROL" if ch == CONTROL else ""
            print(f"{name.split()[0]:<16}{ch:<20}{r['real']:>7.2f}"
                  f"{r['a_med']:>8.2f}{r['b_med']:>8.2f}{r['a_pct']:>8.1f}"
                  f"{r['a_p']:>9.4f}{r['b_p']:>9.4f}{r['tau']:>8.1f}"
                  f"{r['n_eff']:>7.0f}  {clears}{tag}")
        print()

    ctrl_clears = {}
    for name, res in control_by_crisis.items():
        if name in PRIMARY_CRISES and CONTROL in res:
            r = res[CONTROL]
            ctrl_clears[name] = (r["a_p"] < 0.05 and r["b_p"] < 0.05)
    n_ctrl = sum(ctrl_clears.values())
    print(f"CONTROL CLEARS ON {n_ctrl} OF {len(ctrl_clears)} PRIMARY CRISES: "
          + ", ".join(f"{k}={'YES' if v else 'no'}" for k, v in ctrl_clears.items()))
    print("Compare the control's tau and floor against the other channels' on")
    print("any window where it fails, then state which case holds. Do NOT read a")
    print("control failure as case (a) without checking the persistence columns.")

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
