"""Harness diagnostics: POSITIVE CONTROLS + specific bug checks.

WHY THIS EXISTS. Every result in this repo is a non-detection. The panel says
no channel clears its noise floor; the null models said the same on single
crises; even the Gaussian HMM control fails. That pattern has two completely
different explanations which, so far, predict IDENTICAL output:

    (i)  the channels genuinely carry no detectable signal at this power, or
    (ii) the evaluation harness cannot detect anything at all.

Nothing built so far distinguishes them, because nothing built so far has
asked the harness to find something that is unambiguously present. Until a
positive control exists, "nothing cleared" is not a finding -- it is an
uninterpretable measurement, and the write-up cannot lean on it.

This script is DIAGNOSTIC ONLY. It recomputes nothing that the README reports
and changes no existing result. It answers: does the harness work?

------------------------------------------------------------------------------
POSITIVE CONTROLS (the important part)

  1. REALIZED VOLATILITY AS A CHANNEL. SPY's 20-day trailing realized vol,
     pushed through the IDENTICAL downstream -- same causal z-score (w=20,
     m=60), same 15 crisis masks, same Cohen's |d|, same two nulls, same panel
     median, same BH-FDR. Realized vol is definitionally elevated in a
     volatility crisis; it needs no preprocessing, so it also isolates the
     EVALUATION from the EMBEDDING.
       clears  -> the harness works, and the geometric negative gets STRONGER
                  (a channel that must separate does; the others do not).
       fails   -> every negative in this repo is uninterpretable. Not "the
                  channels have nothing" but "we cannot measure whether
                  anything has anything."
     This test can only validate the HARNESS. It can never validate a channel.

  2. END-TO-END SYNTHETIC. A different failure mode from (1): test 1 checks the
     METRIC with a trusted channel on real data; test 2 checks the CODE --
     masking, indexing, null construction, FDR -- against data whose ground
     truth we manufactured. Synthetic returns are generated ON THE REAL TRADING
     CALENDAR with vol/corr spikes injected EXACTLY on the 15 real crisis
     windows, so every line of the panel runs unchanged and only the prices are
     fake. A failure is then attributable to code, not to window definitions or
     calendar handling.

BUG CANDIDATES

  3. HMM CONVERGENCE. Dismissed as "known harmless" since the start, and the
     HMM now has the LOWEST panel median of all seven -- which is what a broken
     control looks like. Reports monitor_.converged_, iterations and final
     log-likelihood delta for every per-crisis fit. Also reports POSTERIOR
     SATURATION, a second failure convergence alone cannot reveal: the causal
     HMM is fit only on pre-cutoff returns, so an early crisis calibrates it on
     a calm world and its high-variance posterior can pin at 1.0 across every
     later crisis. A near-binary posterior, causally z-scored, would produce
     exactly the tiny |d| seen on 2007 (0.17) and Volmageddon (0.06).

  4. MASK ALIGNMENT. Shift every crisis mask by +/-60 trading days and recompute
     the panel median. Flat under a +/-30 shift => either the masks are not where
     we think they are, or the signal is not localized to the crisis. Peaked at
     zero => alignment confirmed. Realized vol supplies the reference curve, so
     we can compare SHAPES rather than guess what "localized" looks like.

  5. SMOOTHING vs SHORT WINDOWS. Algorithm 1 takes a 20-day trailing mean before
     z-scoring; a trailing mean spreads a step into a ramp and drags pre-crisis
     days upward. Ablate w in {1, 5, 10, 20} on the panel median.

  6. EXPANDING-NORMALIZATION DRIFT. causal_zscore normalizes against the mean
     and std of ALL prior smoothed values, so by 2020 the denominator includes
     2008. Cohen's |d| compares a window to the rest of the SAME series, so a
     later crisis divided by a bigger denominator scores lower -- the market did
     not move less, the yardstick grew. Regress per-crisis |d| on crisis date.

  7. NaN / WARM-UP AUDIT. Per crisis and channel, how many days inside the
     window have finite z-scores. A partly-NaN window silently shrinks its own
     sample and inflates the variance of its |d|.

OUTCOME. Test 1 showed the panel MEDIAN cannot detect realized volatility, and
test 1b showed the COUNT can (p = 0.039 vs 0.93). The count was adopted as the
panel's headline statistic on that basis -- selected on the control alone, with
the geometric channels untouched. THIS FILE IS THE PROVENANCE RECORD for that
choice; see scripts/multi_crisis_panel.py for the statistic itself and for why
its integer-coarse resolution means no confirmatory claim is reachable there.

FORKING-PATH DISCIPLINE. Tests 4, 5 and 6 sweep a parameter. Every value is
reported; none is adopted. Moving the windows to wherever |d| peaks, or picking
the smoothing width that maximises the panel median, is precisely the tuning
the honesty architecture forbids (HANDOFF Sec. 6). These sweeps diagnose the
harness; they do not choose a protocol.

    python scripts/diagnostics.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from causal_eval import (  # noqa: E402
    DIM_A,
    N,
    SEED,
    SLD_W,
    cohens_d,
    raw_channels,
)
from multi_crisis_panel import (  # noqa: E402
    FDR_ALPHA,
    align_panel,
    panel_null_tests,
)
from null_model import bh_fdr  # noqa: E402
from qgmrd.crises import CRISIS_WINDOWS, MIN_PRECUTOFF_ROWS  # noqa: E402
from qgmrd.crises import context as crisis_context  # noqa: E402
from qgmrd.data import load_prices, synthetic_prices  # noqa: E402
from qgmrd.features import build_features  # noqa: E402
from qgmrd.operators import random_hermitian_operators  # noqa: E402
from qgmrd.zscore import causal_zscore  # noqa: E402

ZS_W, ZS_M = 20, 60          # causal_zscore defaults (paper Algorithm 1)
RV_WINDOW = 20               # realized-vol lookback, matches the vol20 feature
W_ABLATION = (1, 5, 10, 20)
SHIFT_GRID = tuple(range(-60, 61, 10))
SATURATION_EPS = 0.01        # posterior within this of 0 or 1 counts as pinned

RULE = "=" * 100


def head(n, title: str) -> None:
    print("\n" + RULE)
    print(f"TEST {n} -- {title}")
    print(RULE)


# --------------------------------------------------------------------------
# A vectorised causal_zscore, verified identical to the production one.
#
# The w-ablation needs the z-score recomputed 4 x 8 x 15 times; the reference
# implementation is an O(T^2) Python loop. This reproduces it exactly, and the
# equality assertion below is itself one of the bug checks.
# --------------------------------------------------------------------------

def fast_causal_zscore(raw: np.ndarray, w: int = ZS_W, m: int = ZS_M) -> np.ndarray:
    s = pd.Series(np.asarray(raw, dtype=float))
    sm = s.rolling(w, min_periods=1).mean()          # trailing mean, NaN-skipping
    mu = sm.expanding(min_periods=1).mean().shift(1)  # strictly past
    sd = sm.expanding(min_periods=1).std(ddof=0).shift(1)
    z = (sm - mu) / sd
    out = np.array(z.to_numpy(dtype=float), copy=True)
    out[:m] = np.nan
    out[~(sd.to_numpy() > 0)] = np.nan
    return out


# --------------------------------------------------------------------------

def instrumented_hmm(returns: np.ndarray, fit_mask: np.ndarray) -> dict:
    """Refit the causal HMM with the SAME seed/params, exposing convergence.

    causal_eval.hmm_high_variance_prob_causal does not return the model, so the
    diagnostic refits deterministically and asserts the posterior matches. That
    assertion is a check in its own right: it proves this replica is the same
    computation the panel actually ran.
    """
    from hmmlearn.hmm import GaussianHMM

    r = np.nan_to_num(np.asarray(returns, dtype=float))
    model = GaussianHMM(n_components=2, covariance_type="diag",
                        n_iter=200, random_state=SEED)
    model.fit(r[fit_mask].reshape(-1, 1))
    high = int(np.argmax(model.covars_.reshape(2)))
    post = model.predict_proba(r.reshape(-1, 1))[:, high]
    hist = list(model.monitor_.history)
    # hmmlearn exposes this as `converged` here, NOT `converged_` as HANDOFF
    # Sec. 6 records -- following that note verbatim raises AttributeError.
    mon = model.monitor_
    converged = getattr(mon, "converged", getattr(mon, "converged_", None))
    return {
        "post": post,
        "converged": bool(converged),
        "n_iter": len(hist),
        "delta": (hist[-1] - hist[-2]) if len(hist) >= 2 else float("nan"),
        "pinned_hi": float((post > 1 - SATURATION_EPS).mean()),
        "pinned_lo": float((post < SATURATION_EPS).mean()),
    }


def build_panel(features, prices, ops, idx, label="real"):
    """Per-crisis RAW channels (pre-z-score), cached so w can be re-ablated.

    Same protocol as scripts/multi_crisis_panel.py: scaler + PCA + HMM fit only
    on rows before the crisis cutoff, full timeline transformed through them.
    Adds realized vol as an extra channel -- it needs no fitting, so it is the
    same series for every crisis and isolates evaluation from embedding.
    """
    returns = np.log(prices["SPY"]).diff().reindex(idx).values
    rv = (np.log(prices["SPY"]).diff().rolling(RV_WINDOW).std()
          .reindex(idx).values)
    p = min(8, features.shape[1])
    out = []
    for name, sm, em in CRISIS_WINDOWS:
        ctx = crisis_context(idx, sm, em)
        if ctx["n_window"] == 0 or ctx["n_pre"] < MIN_PRECUTOFF_ROWS:
            continue
        pre = ctx["pre"]
        sc = StandardScaler().fit(features.values[pre])
        pc = PCA(n_components=p, random_state=SEED).fit(sc.transform(features.values[pre]))
        Xp = normalize(pc.transform(sc.transform(features.values)))
        raw = raw_channels(Xp, ops, returns, hmm_fit=pre)
        hmm = instrumented_hmm(returns, pre)
        assert np.allclose(hmm["post"], raw["hmm_high_var_prob"]), (
            f"{label}/{name}: instrumented HMM replica diverges from the "
            "production path -- the diagnostic would not be measuring the "
            "same computation")
        raw["realized_vol_20d"] = rv
        out.append({"name": name, "ctx": ctx, "raw": raw, "hmm": hmm})
        print(f"  built {label}/{name:<22} ({ctx['n_window']:>3}d window, "
              f"{ctx['n_pre']:>4} pre-cutoff rows)")
    return out


def zscore_panel(panel_raw, w=ZS_W):
    return [{"name": c["name"], "ctx": c["ctx"],
             "z": {k: fast_causal_zscore(v, w=w) for k, v in c["raw"].items()}}
            for c in panel_raw]


def report_null_table(results, channels, title, highlight=()):
    print(f"\n{title}")
    print(f"{'channel':<20}{'median|d|':>11}{'null med':>10}{'(a)pct':>8}"
          f"{'(a)p':>9}{'(b)p':>9}{'BH q':>9}  verdict")
    print("-" * 92)
    keys, pv = [], []
    for ch in channels:
        keys += [f"{ch}/(a)", f"{ch}/(b)"]
        pv += [results[ch]["a_p"], results[ch]["b_p"]]
    q = bh_fdr(pv)
    qmap = {keys[i]: q[i] for i in range(len(keys))}
    n_pass = 0
    for ch in sorted(channels, key=lambda c: -results[c]["real_med"]):
        r = results[ch]
        qa, qb = qmap[f"{ch}/(a)"], qmap[f"{ch}/(b)"]
        both = qa < FDR_ALPHA and qb < FDR_ALPHA
        n_pass += int(qa < FDR_ALPHA) + int(qb < FDR_ALPHA)
        verdict = ("CLEARS BOTH" if both else
                   "clears (a) only" if qa < FDR_ALPHA else
                   "clears (b) only" if qb < FDR_ALPHA else "does not clear")
        star = " <<<" if ch in highlight else ""
        print(f"{ch:<20}{r['real_med']:>11.2f}{r['a_med']:>10.2f}"
              f"{r['a_pct']:>8.1f}{r['a_p']:>9.4f}{r['b_p']:>9.4f}"
              f"{min(qa, qb):>9.4f}  {verdict}{star}")
    print(f"\n{n_pass}/{len(pv)} tests survive BH-FDR at q < {FDR_ALPHA} "
          f"({len(pv) * FDR_ALPHA:.1f} expected by chance).")
    return qmap


# --------------------------------------------------------------------------

def synthetic_on_calendar(idx_prices, masks_union, seed=11):
    """Synthetic SPY/DIA on the REAL calendar, spikes on the REAL crisis days.

    Same generative model as qgmrd.data.synthetic_prices (elevated vol,
    elevated cross-correlation, negative drift inside the crisis) but placed on
    the actual trading dates so that crises.py, the past-fit preprocessing and
    the null construction all run byte-for-byte unchanged. If the panel cannot
    find a crisis THIS pipeline manufactured on ITS OWN dates, the defect is in
    the evaluation code.
    """
    rng = np.random.default_rng(seed)
    T = len(idx_prices)
    base_vol, base_corr, base_drift = 0.008, 0.30, 0.0003
    cri_vol, cri_corr, cri_drift = 0.030, 0.85, -0.0020

    vol = np.where(masks_union, cri_vol, base_vol)
    corr = np.where(masks_union, cri_corr, base_corr)
    drift = np.where(masks_union, cri_drift, base_drift)

    z = rng.standard_normal((T, 2))
    r0 = z[:, 0]
    r1 = corr * z[:, 0] + np.sqrt(np.maximum(1 - corr ** 2, 0)) * z[:, 1]
    rets = np.column_stack([drift + vol * r0, drift + vol * r1])
    px = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(px, columns=["SPY", "DIA"], index=idx_prices)


# --------------------------------------------------------------------------

def main() -> None:
    prices = load_prices(("SPY", "DIA"))
    features = build_features(prices)
    idx = features.index
    ops = random_hermitian_operators(min(8, features.shape[1]), n=N, seed=SEED)

    print(RULE)
    print("HARNESS DIAGNOSTICS -- positive controls and bug checks")
    print("Diagnostic only: no existing result is recomputed or changed.")
    print(RULE)

    # --- verify the fast z-score is the production z-score ----------------
    probe = np.log(prices["SPY"]).diff().rolling(RV_WINDOW).std().reindex(idx).values
    ref, fast = causal_zscore(probe, w=ZS_W, m=ZS_M), fast_causal_zscore(probe)
    both = ~np.isnan(ref) & ~np.isnan(fast)
    assert (np.isnan(ref) == np.isnan(fast)).all(), "z-score NaN pattern differs"
    assert np.allclose(ref[both], fast[both], atol=1e-10), "z-score values differ"
    print(f"\nvectorised causal_zscore verified identical to qgmrd.zscore "
          f"({both.sum()} finite points, max|diff| = "
          f"{np.abs(ref[both] - fast[both]).max():.2e})")

    print("\nbuilding real-data panel (raw channels cached for the ablations)...")
    panel_raw = build_panel(features, prices, ops, idx)
    K = len(panel_raw)
    panel = zscore_panel(panel_raw, w=ZS_W)
    channels = list(panel[0]["z"].keys())
    geo = [c for c in channels if c != "realized_vol_20d"]

    # ======================================================================
    head(1, "POSITIVE CONTROL: realized volatility through the same harness")
    print("SPY 20-day trailing realized vol -> same causal z-score, same masks,")
    print("same |d|, same two nulls, same panel median, same FDR. No fitting, so")
    print("this isolates the EVALUATION from the EMBEDDING.")
    print("\nIf realized vol does NOT clear, the harness cannot detect a signal")
    print("that must be present, and every negative in this repo is void.")

    Zs, Ms = align_panel(panel, channels)
    results = {ch: panel_null_tests(Zs[ch], Ms[ch]) for ch in channels}
    report_null_table(results, channels,
                      "Panel medians, 7 channels + realized-vol control:",
                      highlight=("realized_vol_20d",))

    rv = results["realized_vol_20d"]
    print(f"\nrealized vol per-crisis |d|: "
          + ", ".join(f"{panel[k]['name'].split()[0]}={rv['real_per_crisis'][k]:.2f}"
                      for k in range(K)))
    for key, nm in (("2020 COVID", "COVID"), ("2008 GFC", "GFC")):
        k = [i for i, c in enumerate(panel) if c["name"] == key][0]
        print(f"  {nm:<6} realized-vol |d| = {rv['real_per_crisis'][k]:.2f}")

    # ======================================================================
    head("1b", "What panel statistic WOULD have found the control?")
    print("Run on the POSITIVE CONTROL ONLY, deliberately. If we scanned")
    print("statistics across the geometric channels and adopted whichever one")
    print("flattered them, that is a forking path. Choosing the statistic on a")
    print("channel whose answer we already know -- realized vol MUST separate --")
    print("keeps the choice uncontaminated. The geometric channels are NOT")
    print("evaluated here; that is a separate, pre-registered run once a")
    print("statistic is chosen.")
    print("\nSame per-crisis null draws, four different panel summaries:")
    ctrl = results["realized_vol_20d"]
    da = ctrl["draws_a"]
    real_pc = ctrl["real_per_crisis"]
    q95 = np.percentile(da, 95, axis=1)

    stats = {
        "median (current)": (np.median(real_pc), np.median(da, axis=0)),
        "mean": (np.mean(real_pc), np.mean(da, axis=0)),
        "max": (np.max(real_pc), np.max(da, axis=0)),
        "count of crises > own 95th pct":
            (float(np.sum(real_pc > q95)), (da > q95[:, None]).sum(axis=0).astype(float)),
    }
    print(f"{'statistic':<32}{'real':>9}{'null med':>10}{'null 95%':>10}{'p':>9}")
    print("-" * 70)
    for nm, (real_v, null_v) in stats.items():
        p = (1 + int(np.sum(null_v >= real_v))) / (1 + len(null_v))
        print(f"{nm:<32}{real_v:>9.2f}{np.median(null_v):>10.2f}"
              f"{np.percentile(null_v, 95):>10.2f}{p:>9.4f}")
    print(f"\nPer-crisis: {int(np.sum(real_pc > q95))} of {K} crises have a "
          f"realized-vol |d| above their OWN 95th-percentile floor.")
    for k in np.argsort(-real_pc):
        mark = "CLEARS" if real_pc[k] > q95[k] else "      "
        print(f"  {mark}  {panel[k]['name']:<22} |d| = {real_pc[k]:>5.2f}   "
              f"own 95th pct = {q95[k]:.2f}")

    # ======================================================================
    head(2, "POSITIVE CONTROL: end-to-end synthetic (ground truth known)")
    print("Synthetic SPY/DIA on the REAL trading calendar, vol/corr spikes")
    print("injected on the REAL 15 crisis windows. Every line of the panel runs")
    print("unchanged; only the prices are manufactured. Failure here means the")
    print("defect is in the evaluation CODE, not the data.")

    masks_union_prices = np.zeros(len(prices.index), dtype=bool)
    for name, sm, em in CRISIS_WINDOWS:
        ctx = crisis_context(idx, sm, em)
        if ctx["n_window"]:
            masks_union_prices |= np.isin(prices.index, idx[ctx["mask"]])
    syn_prices = synthetic_on_calendar(prices.index, masks_union_prices)
    syn_features = build_features(syn_prices)
    syn_idx = syn_features.index
    print(f"\ninjected crisis days: {int(masks_union_prices.sum())} of "
          f"{len(prices.index)} ({masks_union_prices.mean() * 100:.1f}%)")
    syn_raw = build_panel(syn_features, syn_prices, ops, syn_idx, label="synth")
    syn_panel = zscore_panel(syn_raw, w=ZS_W)
    sZ, sM = align_panel(syn_panel, channels)
    syn_results = {ch: panel_null_tests(sZ[ch], sM[ch]) for ch in channels}
    report_null_table(syn_results, channels,
                      "SYNTHETIC panel medians (ground truth: crisis IS there):",
                      highlight=("realized_vol_20d",))

    print("\nSingle-crisis synthetic (qgmrd.data.synthetic_prices, injected "
          "rows 1000-1120):")
    sp = synthetic_prices()
    sf = build_features(sp)
    s_mask = np.isin(sf.index, sp.index[1000:1120])
    s_ret = np.log(sp["SPY"]).diff().reindex(sf.index).values
    s_scaler = StandardScaler().fit(sf.values)
    s_pca = PCA(n_components=min(8, sf.shape[1]), random_state=SEED).fit(
        s_scaler.transform(sf.values))
    s_Xp = normalize(s_pca.transform(s_scaler.transform(sf.values)))
    s_raw = raw_channels(s_Xp, ops, s_ret, hmm_fit=np.ones(len(sf), dtype=bool))
    s_raw["realized_vol_20d"] = (np.log(sp["SPY"]).diff()
                                 .rolling(RV_WINDOW).std().reindex(sf.index).values)
    print(f"{'channel':<20}{'|d|':>8}")
    print("-" * 28)
    for ch in sorted(s_raw, key=lambda c: -abs(cohens_d(
            fast_causal_zscore(s_raw[c])[s_mask],
            fast_causal_zscore(s_raw[c])[~s_mask]))):
        z = fast_causal_zscore(s_raw[ch])
        print(f"{ch:<20}{abs(cohens_d(z[s_mask], z[~s_mask])):>8.2f}")

    # ======================================================================
    head(3, "HMM convergence and posterior saturation")
    print("The HMM control has the LOWEST panel median of all seven. If it is")
    print("not converging, or its posterior is pinned at 0/1, the writeup's")
    print("'even the control fails' framing is unsupported and must change.")
    print(f"\n{'crisis':<22}{'converged':>11}{'iters':>7}{'delta':>12}"
          f"{'pinned@1':>10}{'pinned@0':>10}{'|d|':>7}")
    print("-" * 79)
    hmm_d = results["hmm_high_var_prob"]["real_per_crisis"]
    n_bad = 0
    for k, c in enumerate(panel_raw):
        h = c["hmm"]
        n_bad += int(not h["converged"])
        print(f"{c['name']:<22}{str(h['converged']):>11}{h['n_iter']:>7}"
              f"{h['delta']:>12.2e}{h['pinned_hi'] * 100:>9.1f}%"
              f"{h['pinned_lo'] * 100:>9.1f}%{hmm_d[k]:>7.2f}")
    print(f"\n{n_bad} of {K} per-crisis HMM fits FAILED to converge.")

    # ======================================================================
    head(4, "Mask alignment sensitivity (+/- 60 trading days)")
    print("Circularly shift every crisis mask together and recompute the panel")
    print("median. A localized, correctly-aligned signal must PEAK at shift 0.")
    print("Flat => masks misaligned, or signal not localized where we think.")
    print("Realized vol is the reference curve for what 'localized' looks like.")
    print("REPORTED, NOT ADOPTED: windows are not moved to wherever |d| peaks.")
    print(f"\n{'channel':<20}" + "".join(f"{s:>7}" for s in SHIFT_GRID) + "   peak")
    print("-" * (20 + 7 * len(SHIFT_GRID) + 8))
    for ch in channels:
        Z, Msk = Zs[ch], Ms[ch]
        M_len = Z.shape[1]
        curve = []
        for sh in SHIFT_GRID:
            ds = []
            for k in range(K):
                pos = np.flatnonzero(Msk[k])
                mm = np.zeros(M_len, dtype=bool)
                mm[(pos + sh) % M_len] = True
                ds.append(abs(cohens_d(Z[k][mm], Z[k][~mm])))
            curve.append(float(np.median(ds)))
        peak = SHIFT_GRID[int(np.argmax(curve))]
        print(f"{ch:<20}" + "".join(f"{v:>7.2f}" for v in curve) + f"{peak:>7d}")

    # ======================================================================
    head("4b", "Alignment, PER CRISIS, on crises the control actually sees")
    print("Test 4 shifts all 15 masks and takes the median, so if the median is")
    print("dominated by crises the control cannot see, a flat curve says nothing")
    print("about alignment. This isolates the question: on the crises where")
    print("realized vol has a large |d|, does its |d| PEAK at shift 0?")
    print("Peaks at 0 => masks are correctly aligned and the flatness in test 4")
    print("is a property of the MEDIAN, not of the masks.")
    Zc, Mc_ = Zs["realized_vol_20d"], Ms["realized_vol_20d"]
    M_len = Zc.shape[1]
    strong = np.argsort(-results["realized_vol_20d"]["real_per_crisis"])[:4]
    print(f"\n{'crisis':<22}" + "".join(f"{s:>7}" for s in SHIFT_GRID) + "   peak")
    print("-" * (22 + 7 * len(SHIFT_GRID) + 8))
    for k in strong:
        pos = np.flatnonzero(Mc_[k])
        curve = []
        for sh in SHIFT_GRID:
            mm = np.zeros(M_len, dtype=bool)
            mm[(pos + sh) % M_len] = True
            curve.append(abs(cohens_d(Zc[k][mm], Zc[k][~mm])))
        print(f"{panel[k]['name']:<22}" + "".join(f"{v:>7.2f}" for v in curve)
              + f"{SHIFT_GRID[int(np.argmax(curve))]:>7d}")

    # ======================================================================
    head(5, "Smoothing ablation: causal_zscore w in {1, 5, 10, 20}")
    print("Algorithm 1 smooths with a 20-day trailing mean before z-scoring,")
    print("which spreads a step into a ramp and lifts pre-crisis days.")
    print("NOTE ON THE PREMISE: under G.10 +/-10 the SHORTEST windows are 62-63")
    print("trading days (Flash Crash 62, Brexit 62, Repo 63) -- the +/-10")
    print("extension roughly doubled the bare G.10 months, so the dilution is")
    print("milder than '20-day smoother on a 30-day event', though a 20-day")
    print("smoother against 62 days still blurs about a third of it.")
    print("ALL values reported; none adopted. Picking the winning w is tuning.")
    short = ["2010 Flash Crash", "2016 Brexit", "2019 Repo Crisis"]
    short_i = [i for i, c in enumerate(panel_raw) if c["name"] in short]
    for w in W_ABLATION:
        pw = zscore_panel(panel_raw, w=w)
        zw, mw = align_panel(pw, channels)
        rw = {ch: panel_null_tests(zw[ch], mw[ch]) for ch in channels}
        line = ", ".join(f"{ch[:9]}={rw[ch]['real_med']:.2f}" for ch in channels)
        n_clear = sum(1 for ch in channels
                      if rw[ch]["a_p"] < 0.05 and rw[ch]["b_p"] < 0.05)
        sm_med = {ch: float(np.median([rw[ch]["real_per_crisis"][i]
                                       for i in short_i])) for ch in channels}
        print(f"\n  w = {w:>2}  panel medians: {line}")
        print(f"          short-crisis medians (Flash/Brexit/Repo): "
              + ", ".join(f"{ch[:9]}={sm_med[ch]:.2f}" for ch in channels))
        print(f"          channels with raw p < 0.05 on BOTH nulls: {n_clear}"
              f"/{len(channels)}")

    # ======================================================================
    head(6, "Expanding-normalization drift: does |d| decay with crisis date?")
    print("causal_zscore normalizes against ALL prior smoothed values, so by")
    print("2020 the denominator includes 2008. |d| compares a window to the")
    print("rest of the SAME series, so a later crisis divided by a bigger")
    print("denominator scores lower -- the yardstick grew, not the market.")
    print("CAVEAT: 15 points, and crises genuinely differ in severity. A null")
    print("result here does NOT rule drift out. This is a screen, not a proof.")
    years = np.array([c["ctx"]["start"].year + c["ctx"]["start"].dayofyear / 365.0
                      for c in panel])
    print(f"\n{'channel':<20}{'spearman':>10}{'p':>9}{'OLS slope/yr':>15}")
    print("-" * 54)
    from scipy import stats as sps
    for ch in channels:
        d = results[ch]["real_per_crisis"]
        rho, pv = sps.spearmanr(years, d)
        slope = sps.linregress(years, d).slope
        print(f"{ch:<20}{rho:>10.2f}{pv:>9.3f}{slope:>15.4f}")

    print("\nThe mechanism itself (expanding SD of the smoothed series at each")
    print("crisis midpoint) -- if drift is real this must RISE with date:")
    print(f"{'crisis':<22}{'realized_vol sd':>17}{'spectral_ent sd':>17}")
    print("-" * 56)
    for k, c in enumerate(panel_raw):
        mid = int(np.flatnonzero(c["ctx"]["mask"]).mean())
        row = f"{c['name']:<22}"
        for ch in ("realized_vol_20d", "spectral_entropy"):
            s = pd.Series(c["raw"][ch]).rolling(ZS_W, min_periods=1).mean()
            row += f"{float(s[:mid].std(ddof=0)):>17.5f}"
        print(row)

    # ======================================================================
    head(7, "NaN / warm-up audit: finite z-scores inside each crisis window")
    print("A partly-NaN window silently shrinks its own sample and inflates the")
    print("variance of its |d|. Expected clean (earliest window is at index")
    print("~610, warm-up ends ~80), but 'expected clean' is what audits check.")
    print(f"\n{'crisis':<22}{'window':>8}" +
          "".join(f"{ch[:9]:>10}" for ch in channels))
    print("-" * (30 + 10 * len(channels)))
    total_missing = 0
    for c in panel:
        L = int(c["ctx"]["mask"].sum())
        row = f"{c['name']:<22}{L:>8}"
        for ch in channels:
            n_fin = int(np.sum(~np.isnan(c["z"][ch][c["ctx"]["mask"]])))
            total_missing += L - n_fin
            row += f"{n_fin:>10}"
        print(row)
    print(f"\nTotal missing crisis-window days across all channels: "
          f"{total_missing}")

    # ======================================================================
    print("\n" + RULE)
    print("VERDICT")
    print(RULE)
    rvq = min(bh_fdr([results[ch]["a_p"] for ch in channels]
                     + [results[ch]["b_p"] for ch in channels]))
    print(f"realized-vol panel median |d| = {rv['real_med']:.2f} vs. null median "
          f"{rv['a_med']:.2f}  (p_a = {rv['a_p']:.4f}, p_b = {rv['b_p']:.4f})")
    print(f"synthetic realized-vol median |d| = "
          f"{syn_results['realized_vol_20d']['real_med']:.2f} vs. null "
          f"{syn_results['realized_vol_20d']['a_med']:.2f}  "
          f"(p_a = {syn_results['realized_vol_20d']['a_p']:.4f})")
    print(f"HMM fits failing to converge: {n_bad}/{K}")
    print(f"(smallest BH q anywhere in the real panel incl. control: {rvq:.4f})")
    print(RULE + "\n")


if __name__ == "__main__":
    main()
