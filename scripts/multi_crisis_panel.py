"""Multi-crisis panel (HANDOFF Sec. 8, item 1) -- the critical-path task.

WHY THIS EXISTS. The null-model tests (scripts/null_model.py, HANDOFF Open
Question 3) found that essentially nothing clears its own noise floor on a
single crisis -- including the Gaussian HMM control. That result cannot
distinguish "no signal" from "signal too weak to see with one short window,"
and only the weaker claim is supported. A panel is the only lever that raises
power, so this is the gating task for any per-channel detection claim.

THE HEADLINE STATISTIC IS THE MEDIAN |d| ACROSS CRISES, not per-crisis values.
Three reasons, all decided before any result:

  1. POWER. A single crisis's |d| sits inside a huge noise floor -- the existing
     per-crisis null 95%% intervals run to [0.03, 1.46] and wider, because an
     autocorrelated, fat-tailed series lets ANY contiguous window separate by
     luck. The median of k draws concentrates around the true median at rate
     1/sqrt(k): with k = 15 the null's spread narrows roughly 4x. Luck does not
     repeat in the same direction eight times out of fifteen, so the same real
     effect moves far further into the tail of a median-null than of a
     single-window null. The real value does not change; the FLOOR gets sharp.

  2. MULTIPLE COMPARISONS COLLAPSE. Per-crisis testing is 15 x 7 x 2 = 210
     tests, expecting ~10.5 chance survivors at alpha = 0.05 -- an uncountable
     result. Testing the median is 7 channels x 2 nulls = 14 tests, expecting
     0.7. A survivor becomes countable as evidence instead of drowning.

  3. IT MATCHES HAMMOND. His Table 3 is a 17-crisis median. Every comparison
     this repo has made to his figures so far has been a point estimate against
     a median -- a type error the panel fixes.

Median, not mean: |d| has a heavy right tail, so a mean would let one lucky
crisis carry the panel. A median cannot be moved by one crisis, and it stays
robust to control-blind crises WITHOUT dropping them (post-hoc dropping is a
forking path -- see the control-sanity flags, which are reported, not acted on).

PRE-REGISTERED HYPOTHESES, recorded here before any result existed. Both are
reported regardless of outcome, in the committed language (HANDOFF Sec. 2d):

  H1: ground energy E0 clears its null on the panel median. Motivation: it was
      the ONLY single-crisis survivor (2022, q = 0.006, 100th percentile of its
      own floor -- not one of 5000 random windows reached it).

  H2: the SLD mixed-state QFI channel clears its null on the panel median.
      Motivation: it cleared NOTHING on single crises despite having the
      easiest bar (lowest floor, null median 0.23-0.47) and the most power
      (least autocorrelated series, N_eff ~ 204 vs ~ 13 for spectral entropy).

ANTI-FORKING-PATHS CONSTRAINT. We do NOT search for which channel wins which
crisis type. Hammond tested for per-crisis specialization and found none
(p = 0.31), so any pattern visible across 15 crises is almost certainly noise.
The median is the statistic; the per-crisis table is DESCRIPTIVE ONLY and
carries that caveat wherever it is printed.

------------------------------------------------------------------------------
PROTOCOL

Windows: the 15 post-2005 entries of Hammond's Table G.10, extended by +/-10
trading days (his Sec. 4.1), from the shared registry qgmrd/crises.py. Not
tuned, not added to.

Per crisis: causal (past-fit) preprocessing exactly as scripts/causal_eval.py
does it -- scaler, PCA and the HMM baseline fit only on rows before
(extended window start - 10 business days), then the full timeline transformed
through those past-fit objects. Crises with < 200 pre-cutoff rows are skipped
and the skip is reported. All 7 channels including the HMM control.

Null tests on the MEDIAN, both from null_model.py, restructured for a panel:

  (a) Random matched-length windows. Per crisis INDEPENDENTLY, draw a window of
      that crisis's own length elsewhere in that crisis's own causal series,
      non-overlapping it; take the median of the 15 resulting |d|. Repeat.
      CAVEAT, stated because it cuts against us: drawing independently across
      crises destroys the cross-crisis dependence that the real 15 values have
      (they come from one market), so this null median is TIGHTER than it
      should be -- an easier bar. Null (a) is anti-conservative.

  (b) Circular shift, ONE COMMON shift applied to all 15 series at once, every
      crisis mask held fixed. This preserves the cross-crisis dependence
      exactly -- it slides the whole world under a fixed crisis calendar -- so
      it is the conservative one. Its p-value is floored at 1/(N_eff+1) as in
      null_model.py, because a common shift is still ONE autocorrelated draw
      dimension; N_eff uses the most autocorrelated of the 15 series (again the
      conservative choice).

The two nulls BRACKET the truth. A channel clearing only (a) is a weaker
result than one clearing both, and is reported as such.

Block-bootstrap CIs (HANDOFF Sec. 8 item 1; the Task 2 writeup defers to these
in several places). The null and the CI answer DIFFERENT questions: the null
asks whether the median exceeds chance (location), the CI asks how precisely
the median is measured (width). A channel can clear its null with a CI too wide
to be useful, or have a tight CI around a median sitting squarely in noise.

  Construction: ONE circular block resample of the whole timeline per replicate,
  with the crisis labels riding along on the blocks, then all 15 |d| recomputed
  on that single resampled series and the median taken. This matters. The 15
  real |d| are NOT independent -- each is scored against a "rest" group holding
  almost the entire series, including the other 14 crisis windows, so they share
  nearly all their data. Resampling each crisis separately would understate the
  variance of their median and produce a CI that is too narrow -- the same
  failure mode as using an iid bootstrap on an autocorrelated series. One
  resample, fifteen dependent statistics, correct propagation.

  Block length by the Politis-White (2004) automatic rule, not by hand: the
  block length is the one free knob here, and a knob tuned by eye on a reported
  statistic is exactly the fitting the honesty architecture forbids.

  Scope limit: this propagates WITHIN-series sampling error only. It does not
  propagate "which 15 crises" uncertainty -- the crisis set is fixed by Table
  G.10, so the interval is conditional on that set.

Per-crisis control sanity: any crisis whose HMM control scores below 0.2 is
FLAGGED (China 2015 was blind at 0.05 in the three-crisis work). Flags are
reported and the crisis is NOT dropped -- post-hoc dropping is a forking path.

    python scripts/multi_crisis_panel.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from causal_eval import (  # noqa: E402  (shared pipeline pieces)
    N,
    SEED,
    cohens_d,
    raw_channels,
    zscored,
)
from null_model import bh_fdr, integrated_autocorr_time  # noqa: E402
from qgmrd.crises import CRISIS_WINDOWS, MIN_PRECUTOFF_ROWS  # noqa: E402
from qgmrd.crises import context as crisis_context  # noqa: E402
from qgmrd.data import load_prices  # noqa: E402
from qgmrd.features import build_features  # noqa: E402
from qgmrd.operators import random_hermitian_operators  # noqa: E402

N_WINDOW_DRAWS = 5000     # null (a) panel medians
N_BOOT = 2000             # block-bootstrap replicates
FDR_ALPHA = 0.05
HMM_BLIND_THRESHOLD = 0.2  # control-sanity flag; flagged, never dropped
MIN_BOOT_CRISIS_DAYS = 10  # a crisis contributes to a replicate only above this

CI_LO, CI_HI = 2.5, 97.5


# --------------------------------------------------------------------------
# Fast Cohen's |d| for a contiguous (possibly wrapped) window.
#
# Every null here is "same series, different window placement," so all the
# per-draw work reduces to two prefix sums. Building them over the DOUBLED
# series makes a wrapped window contiguous, which is what lets the circular
# shift use the identical code path as the random-window null.
# --------------------------------------------------------------------------

def _prefix_sums(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    zz = np.concatenate([z, z])
    c1 = np.concatenate([[0.0], np.cumsum(zz)])
    c2 = np.concatenate([[0.0], np.cumsum(zz * zz)])
    return c1, c2, float(z.sum()), float((z * z).sum())


def _window_abs_d(c1, c2, t1, t2, m_len, starts, L):
    """|Cohen's d| of window [s, s+L) vs. the rest, vectorised over ``starts``.

    Identical arithmetic to causal_eval.cohens_d: pooled SD with
    ((na-1)var_a + (nb-1)var_b)/(na+nb-2), which equals (SS_a + SS_b)/(M-2).
    """
    starts = np.asarray(starts, dtype=np.int64)
    s1 = c1[starts + L] - c1[starts]
    s2 = c2[starts + L] - c2[starts]
    r1, r2 = t1 - s1, t2 - s2
    na, nb = float(L), float(m_len - L)
    ss_a = np.maximum(s2 - s1 * s1 / na, 0.0)
    ss_b = np.maximum(r2 - r1 * r1 / nb, 0.0)
    sp = np.sqrt((ss_a + ss_b) / (m_len - 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (s1 / na - r1 / nb) / sp
    return np.abs(np.where(sp > 0, d, 0.0))


# --------------------------------------------------------------------------
# Politis-White (2004) automatic block length
# --------------------------------------------------------------------------

def politis_white_block_length(x: np.ndarray) -> tuple[int, bool]:
    """(block length, search_saturated) for the circular block bootstrap.

    Follows Politis & White (2004): pick the flat-top lag-window bandwidth M
    from where the autocorrelation becomes negligible, then

        b_opt = (2 G^2 / D_CB)^(1/3) n^(1/3),
        G  = sum_k lambda(k/M) |k| R(k),
        D_CB = (4/3) (sum_k lambda(k/M) R(k))^2

    with lambda the flat-top window. Autocovariances R(k) enter G and D as a
    ratio, so the result is scale-free. Returns ``search_saturated=True`` when
    the autocorrelation never drops below the threshold inside the rule's own
    search range -- that is not an error, but it means the block length is a
    lower bound for that channel's persistence and we say so.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 16:
        return 1, False
    xc = x - x.mean()

    k_n = max(5, int(np.ceil(2 * np.sqrt(np.log10(n)))))
    m_max = int(np.ceil(np.sqrt(n))) + k_n

    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(xc, nfft)
    acov = np.fft.irfft(f * np.conj(f), nfft)[: m_max + 1].real / n
    if acov[0] <= 0:
        return 1, False
    rho = acov / acov[0]

    thresh = 2.0 * np.sqrt(np.log10(n) / n)
    m_hat, saturated = None, False
    for m in range(1, m_max - k_n + 1):
        if np.all(np.abs(rho[m + 1 : m + 1 + k_n]) < thresh):
            m_hat = m
            break
    if m_hat is None:
        m_hat, saturated = m_max, True

    M = int(min(2 * m_hat, m_max))
    kk = np.arange(-M, M + 1)
    u = np.abs(kk) / M
    lam = np.where(u <= 0.5, 1.0, np.maximum(2.0 * (1.0 - u), 0.0))
    R = acov[np.abs(kk)]
    G = float(np.sum(lam * np.abs(kk) * R))
    g0 = float(np.sum(lam * R))
    D = (4.0 / 3.0) * g0 * g0

    b_max = int(np.ceil(min(3.0 * np.sqrt(n), n / 3.0)))
    if G <= 0 or D <= 0:
        return 1, saturated
    b = ((2.0 * G * G) / D) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return int(np.clip(round(b), 1, b_max)), saturated


# --------------------------------------------------------------------------
# Block bootstrap: ONE resample of the whole timeline, 15 dependent statistics
# --------------------------------------------------------------------------

def block_bootstrap_medians(Z: np.ndarray, masks: np.ndarray, b: int,
                            n_boot: int, rng) -> tuple[np.ndarray, dict]:
    """Bootstrap distribution of the panel median |d| for one channel.

    ``Z`` is (K, M): crisis k's own causal z-scored series (they differ because
    each crisis has its own past-fit preprocessing). ``masks`` is (K, M) with
    crisis k's window.

    Per replicate we draw ONE set of circular block start positions and apply
    the SAME selection to all K series, with each crisis's labels carried along
    by ``masks[k][sel]``. That keeps the K statistics dependent through their
    shared data, which is how the real 15 are related -- resampling each crisis
    on its own would understate the median's variance.

    Because blocks are long relative to a short crisis window, a crisis can be
    under-represented (or absent) in a replicate. Such a crisis is dropped from
    that replicate's median and the frequency is reported. This widens the
    interval, i.e. errs conservative, which is why it is kept rather than
    patched around.
    """
    K, M = Z.shape
    n_blocks = int(np.ceil(M / b))
    lens = np.full(n_blocks, b, dtype=np.int64)
    lens[-1] = M - (n_blocks - 1) * b

    starts = rng.integers(0, M, size=(n_boot, n_blocks)).astype(np.int64)
    ends = starts + lens[None, :]

    out = np.full((n_boot, K), np.nan)
    thin = np.zeros(K, dtype=int)

    for k in range(K):
        z = Z[k]
        m = masks[k].astype(float)
        acc = {}
        for key, arr in (("z", z), ("z2", z * z), ("zm", z * m),
                         ("z2m", z * z * m), ("m", m)):
            dd = np.concatenate([arr, arr])
            c = np.concatenate([[0.0], np.cumsum(dd)])
            acc[key] = (c[ends] - c[starts]).sum(axis=1)

        na = acc["m"]
        nb = M - na
        s1a, s2a = acc["zm"], acc["z2m"]
        s1b, s2b = acc["z"] - acc["zm"], acc["z2"] - acc["z2m"]

        ok = (na >= MIN_BOOT_CRISIS_DAYS) & (nb >= 2)
        thin[k] = int((~ok).sum())
        if not ok.any():
            continue

        naf, nbf = na[ok], nb[ok]
        ss_a = np.maximum(s2a[ok] - s1a[ok] ** 2 / naf, 0.0)
        ss_b = np.maximum(s2b[ok] - s1b[ok] ** 2 / nbf, 0.0)
        sp = np.sqrt((ss_a + ss_b) / (M - 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            d = (s1a[ok] / naf - s1b[ok] / nbf) / sp
        vals = np.abs(np.where(sp > 0, d, 0.0))
        idx = np.flatnonzero(ok)
        out[idx, k] = vals

    with np.errstate(invalid="ignore"):
        med = np.nanmedian(out, axis=1)
    kept = (~np.isnan(out)).sum(axis=1)
    diag = {
        "per_crisis_drop": thin / float(n_boot),
        "kept_mean": float(kept.mean()),
        "kept_p5": float(np.percentile(kept, 5)),
        "kept_min": int(kept.min()),
        "any_dropped_frac": float(np.mean(kept < K)),
    }
    return med, diag


# --------------------------------------------------------------------------

def main() -> None:
    rng_master = np.random.default_rng(SEED)

    prices = load_prices(("SPY", "DIA"))
    features = build_features(prices)
    idx = features.index
    ops = random_hermitian_operators(min(8, features.shape[1]), n=N, seed=SEED)
    returns = np.log(prices["SPY"]).diff().reindex(idx).values
    p = min(8, features.shape[1])

    print("\n" + "=" * 100)
    print("MULTI-CRISIS PANEL (HANDOFF Sec. 8 item 1) -- 15 crises, 7 channels")
    print("Windows: Hammond Table G.10 post-2005, extended +/-10 trading days")
    print("(his Sec. 4.1), from qgmrd/crises.py. Causal past-fit preprocessing")
    print("per crisis, exactly as scripts/causal_eval.py. HEADLINE = the MEDIAN")
    print("|d| across crises; the per-crisis table is DESCRIPTIVE ONLY.")
    print("=" * 100)

    # ---- per-crisis causal channels -------------------------------------
    panel: list[dict] = []
    skipped: list[tuple[str, int]] = []

    for name, sm, em in CRISIS_WINDOWS:
        ctx = crisis_context(idx, sm, em)
        if ctx["n_window"] == 0 or ctx["n_pre"] < MIN_PRECUTOFF_ROWS:
            skipped.append((name, ctx.get("n_pre", 0)))
            continue
        pre = ctx["pre"]
        sc = StandardScaler().fit(features.values[pre])
        pc = PCA(n_components=p, random_state=SEED).fit(sc.transform(features.values[pre]))
        Xp = normalize(pc.transform(sc.transform(features.values)))
        causal = zscored(raw_channels(Xp, ops, returns, hmm_fit=pre))
        panel.append({"name": name, "ctx": ctx, "z": causal})
        print(f"  fitted {name:<22} window {ctx['start'].date()} -> "
              f"{ctx['end'].date()} ({ctx['n_window']:>3}d), "
              f"cutoff {ctx['cutoff'].date()}, {ctx['n_pre']} pre-cutoff rows")

    if skipped:
        print("\nSKIPPED crises (< %d pre-cutoff rows):" % MIN_PRECUTOFF_ROWS)
        for name, n_pre in skipped:
            print(f"  {name:<22} {n_pre} pre-cutoff rows")
    else:
        print(f"\nNo crises skipped: all {len(panel)} have >= "
              f"{MIN_PRECUTOFF_ROWS} pre-cutoff rows.")

    channels = list(panel[0]["z"].keys())
    K = len(panel)

    # ---- per-crisis |d| table (descriptive only) ------------------------
    real_d = {ch: np.array([abs(cohens_d(c["z"][ch][c["ctx"]["mask"]],
                                         c["z"][ch][~c["ctx"]["mask"]]))
                            for c in panel]) for ch in channels}

    print("\n" + "=" * 100)
    print("PER-CRISIS causal |d| -- DESCRIPTIVE ONLY, NOT a specialization claim.")
    print("Hammond tested for per-crisis specialization and found none (p = 0.31),")
    print("so any 'channel X owns crisis type Y' pattern below is almost certainly")
    print("noise. Do not mine it. The median row is the statistic.")
    print("=" * 100)
    hdr = f"{'crisis':<22}" + "".join(f"{ch[:11]:>12}" for ch in channels) + "  flag"
    print(hdr)
    print("-" * len(hdr))
    flags = []
    for i, c in enumerate(panel):
        row = f"{c['name']:<22}" + "".join(f"{real_d[ch][i]:>12.2f}" for ch in channels)
        blind = real_d["hmm_high_var_prob"][i] < HMM_BLIND_THRESHOLD
        if blind:
            flags.append((c["name"], real_d["hmm_high_var_prob"][i]))
        print(row + ("  <-- CONTROL BLIND" if blind else ""))
    print("-" * len(hdr))
    med = {ch: float(np.median(real_d[ch])) for ch in channels}
    print(f"{'MEDIAN (headline)':<22}" + "".join(f"{med[ch]:>12.2f}" for ch in channels))

    print(f"\nControl-sanity flags (HMM control |d| < {HMM_BLIND_THRESHOLD}): "
          f"{len(flags)} of {K} crises.")
    for name, v in flags:
        print(f"  {name:<22} HMM |d| = {v:.2f}")
    print("These are REPORTED, NOT DROPPED. Removing crises after seeing the")
    print("control is a forking path; the median is robust to a few bad crises")
    print("by construction, which is part of why it is the headline.")

    # ---- assemble the finite-aligned panel matrices ---------------------
    Zs, Ms = {}, {}
    for ch in channels:
        finite = np.ones(len(idx), dtype=bool)
        for c in panel:
            finite &= ~np.isnan(c["z"][ch])
        Zs[ch] = np.array([c["z"][ch][finite] for c in panel])
        Ms[ch] = np.array([c["ctx"]["mask"][finite] for c in panel])
        # every crisis window must survive the warm-up trim intact
        for k in range(K):
            assert Ms[ch][k].sum() > 0, f"{ch}/{panel[k]['name']}: window trimmed away"

    # cross-check the vectorised |d| against the reference implementation
    for ch in channels:
        M_len = Zs[ch].shape[1]
        for k in range(K):
            pos = np.flatnonzero(Ms[ch][k])
            c1, c2, t1, t2 = _prefix_sums(Zs[ch][k])
            fast = _window_abs_d(c1, c2, t1, t2, M_len, [pos[0]], len(pos))[0]
            ref = abs(cohens_d(Zs[ch][k][Ms[ch][k]], Zs[ch][k][~Ms[ch][k]]))
            assert abs(fast - ref) < 1e-8, f"{ch}/{k}: {fast} vs {ref}"

    # ---- null tests on the MEDIAN ---------------------------------------
    print("\n" + "=" * 100)
    print("NULL TESTS ON THE PANEL MEDIAN")
    print(f"(a) random matched-length windows, drawn per crisis INDEPENDENTLY, "
          f"{N_WINDOW_DRAWS} panel draws.")
    print("    Independence across crises makes this null TIGHTER than the real")
    print("    (dependent) 15 -> an easier bar. Anti-conservative; stated, not hidden.")
    print("(b) circular shift, ONE COMMON shift for all 15 series, masks fixed.")
    print("    Preserves cross-crisis dependence exactly. Conservative.")
    print("    p floored at 1/(N_eff+1); N_eff from the MOST autocorrelated series.")
    print("The two nulls BRACKET the truth: clearing only (a) is the weaker result.")
    print("=" * 100)

    results = {}
    for ch in channels:
        Z, Msk = Zs[ch], Ms[ch]
        M_len = Z.shape[1]
        pre = [_prefix_sums(Z[k]) for k in range(K)]
        spans = [(int(np.flatnonzero(Msk[k])[0]), int(np.flatnonzero(Msk[k])[-1]),
                  int(Msk[k].sum())) for k in range(K)]
        real_med = float(np.median([_window_abs_d(*pre[k], M_len, [spans[k][0]],
                                                  spans[k][2])[0] for k in range(K)]))

        # (a) independent random matched-length windows per crisis
        rng = np.random.default_rng(SEED)
        da = np.empty((K, N_WINDOW_DRAWS))
        for k in range(K):
            lo, hi, L = spans[k]
            cand = np.array([s for s in range(0, M_len - L + 1)
                             if (s + L - 1 < lo) or (s > hi)], dtype=np.int64)
            da[k] = _window_abs_d(*pre[k], M_len, rng.choice(cand, N_WINDOW_DRAWS), L)
        null_a = np.median(da, axis=0)

        # (b) one common circular shift, all crises at once
        shifts = np.arange(1, M_len, dtype=np.int64)
        db = np.empty((K, len(shifts)))
        for k in range(K):
            lo, _, L = spans[k]
            db[k] = _window_abs_d(*pre[k], M_len, (lo - shifts) % M_len, L)
        null_b = np.median(db, axis=0)

        taus = [integrated_autocorr_time(Z[k])[0] for k in range(K)]
        n_eff = M_len / max(taus)

        p_a = (1 + int(np.sum(null_a >= real_med))) / (1 + len(null_a))
        p_b_raw = (1 + int(np.sum(null_b >= real_med))) / (1 + len(null_b))
        p_b = max(p_b_raw, 1.0 / (n_eff + 1.0))

        results[ch] = {
            "real_med": real_med,
            "a_med": float(np.median(null_a)), "a_lo": float(np.percentile(null_a, 2.5)),
            "a_hi": float(np.percentile(null_a, 97.5)),
            "a_pct": float((null_a < real_med).mean() * 100.0), "a_p": p_a,
            "b_med": float(np.median(null_b)),
            "b_pct": float((null_b < real_med).mean() * 100.0),
            "b_p_raw": p_b_raw, "b_p": p_b,
            "tau_max": float(max(taus)), "n_eff": float(n_eff),
        }

    print(f"{'channel':<20}{'median|d|':>11}{'(a)null':>9}{'(a)95%':>16}"
          f"{'(a)pct':>8}{'(a)p':>9}{'(b)null':>9}{'(b)pct':>8}{'(b)p':>9}"
          f"{'tau':>7}{'Neff':>7}")
    print("-" * 113)
    for ch in sorted(results, key=lambda c: -results[c]["real_med"]):
        r = results[ch]
        ci = f"[{r['a_lo']:.2f},{r['a_hi']:.2f}]"
        print(f"{ch:<20}{r['real_med']:>11.2f}{r['a_med']:>9.2f}{ci:>16}"
              f"{r['a_pct']:>8.1f}{r['a_p']:>9.4f}{r['b_med']:>9.2f}"
              f"{r['b_pct']:>8.1f}{r['b_p']:>9.4f}"
              f"{r['tau_max']:>7.0f}{r['n_eff']:>7.0f}")

    # ---- BH-FDR over the 14 tests ---------------------------------------
    keys, pvals = [], []
    for ch in channels:
        keys += [f"{ch}/(a)", f"{ch}/(b)"]
        pvals += [results[ch]["a_p"], results[ch]["b_p"]]
    q = bh_fdr(pvals)

    print("\n" + "=" * 100)
    print(f"Benjamini-Hochberg FDR over {len(pvals)} tests "
          f"(7 channels x 2 nulls). Expected chance survivors at "
          f"alpha={FDR_ALPHA}: {len(pvals) * FDR_ALPHA:.1f}.")
    print("Per-crisis testing would have been 15 x 7 x 2 = 210 tests, expecting")
    print("~10.5 -- an uncountable result. Nulling the median is what makes a")
    print("survivor countable as evidence.")
    print("=" * 100)
    print(f"{'test':<28}{'raw p':>10}{'BH q':>10}{'q<0.05':>9}")
    print("-" * 57)
    for i in np.argsort(q):
        print(f"{keys[i]:<28}{pvals[i]:>10.4f}{q[i]:>10.4f}"
              f"{('yes' if q[i] < FDR_ALPHA else 'no'):>9}")
    n_pass = int((q < FDR_ALPHA).sum())
    print(f"\n{n_pass}/{len(pvals)} tests survive BH-FDR at q < {FDR_ALPHA} "
          f"(vs. {len(pvals) * FDR_ALPHA:.1f} expected by chance).")
    qmap = {keys[i]: float(q[i]) for i in range(len(keys))}

    # ---- block-bootstrap CIs on the median ------------------------------
    print("\n" + "=" * 100)
    print(f"BLOCK-BOOTSTRAP CIs ON THE MEDIAN ({N_BOOT} replicates, "
          f"{CI_LO}-{CI_HI} percentile)")
    print("ONE circular block resample of the whole timeline per replicate,")
    print("crisis labels riding along; all 15 |d| recomputed on that single")
    print("resample, then the median. The 15 real |d| share nearly all their")
    print("data (each 'rest' group holds the other 14 crisis windows), so")
    print("resampling crises independently would give a CI that is too narrow.")
    print("Block length: Politis-White (2004) automatic rule, not hand-picked.")
    print("DIFFERENT QUESTION FROM THE NULL: the null asks whether the median")
    print("beats chance; the CI asks how precisely it is measured.")
    print("Conditional on this fixed 15-crisis set (G.10) -- 'which crises'")
    print("uncertainty is NOT propagated.")
    print("=" * 100)
    print(f"{'channel':<20}{'median|d|':>11}{'CI low':>9}{'CI high':>9}"
          f"{'width':>8}{'block b':>9}{'PW sat':>8}{'kept/15':>9}{'kept p5':>9}")
    print("-" * 92)
    boot = {}
    for ch in channels:
        Z, Msk = Zs[ch], Ms[ch]
        bs, sats = zip(*[politis_white_block_length(Z[k]) for k in range(K)])
        b = int(max(bs))            # most conservative of the 15 (widest CI)
        saturated = any(sats)
        rng = np.random.default_rng(SEED + 1)
        med_b, diag = block_bootstrap_medians(Z, Msk, b, N_BOOT, rng)
        lo = float(np.nanpercentile(med_b, CI_LO))
        hi = float(np.nanpercentile(med_b, CI_HI))
        boot[ch] = {"lo": lo, "hi": hi, "b": b, "sat": saturated, "diag": diag}
        print(f"{ch:<20}{results[ch]['real_med']:>11.2f}{lo:>9.2f}{hi:>9.2f}"
              f"{hi - lo:>8.2f}{b:>9d}{('yes' if saturated else 'no'):>8}"
              f"{diag['kept_mean']:>9.1f}{diag['kept_p5']:>9.0f}")
    print("\n'PW sat' = the Politis-White autocorrelation search hit the end of")
    print("its own range, so that block length is a LOWER bound on the channel's")
    print("persistence and its CI is, if anything, too narrow.")
    print(f"'kept/15' = mean number of crises contributing to a replicate's")
    print(f"median; a crisis is dropped from a replicate if fewer than "
          f"{MIN_BOOT_CRISIS_DAYS} of its")
    print("days survive that resample. This is REAL and worth reading: the")
    print("Politis-White block length (~150-180) is larger than the shortest")
    print("crisis windows (62 days), so short crises are drawn all-or-nothing.")
    print("It widens the interval -- the safe direction -- but the CI is a")
    print("median over roughly 'kept/15' crises per replicate, not always 15.")

    # ---- pre-registered hypotheses --------------------------------------
    print("\n" + "=" * 100)
    print("PRE-REGISTERED HYPOTHESES (recorded in this file's docstring before")
    print("any result existed). Both reported regardless of outcome.")
    print("=" * 100)
    for tag, ch, why in (
        ("H1", "ground_energy_E0",
         "only single-crisis survivor (2022, q = 0.006, 100th pct of its floor)"),
        ("H2", "sld_qfi_w20",
         "cleared nothing on single crises despite easiest floor + highest N_eff"),
    ):
        r, bt = results[ch], boot[ch]
        qa, qb = qmap[f"{ch}/(a)"], qmap[f"{ch}/(b)"]
        cleared = (qa < FDR_ALPHA, qb < FDR_ALPHA)
        verdict = ("CLEARS BOTH NULLS" if all(cleared)
                   else "CLEARS ONLY THE ANTI-CONSERVATIVE NULL (a)" if cleared[0]
                   else "CLEARS ONLY THE CONSERVATIVE NULL (b)" if cleared[1]
                   else "DOES NOT CLEAR EITHER NULL")
        print(f"\n{tag}: {ch} clears its null on the panel median.")
        print(f"    prior:   {why}")
        print(f"    median |d| = {r['real_med']:.2f}   "
              f"95% CI [{bt['lo']:.2f}, {bt['hi']:.2f}]   (block b = {bt['b']})")
        print(f"    null (a) median {r['a_med']:.2f}, pct {r['a_pct']:.1f}, "
              f"p = {r['a_p']:.4f}, q = {qa:.4f}")
        print(f"    null (b) median {r['b_med']:.2f}, pct {r['b_pct']:.1f}, "
              f"p = {r['b_p']:.4f} (raw {r['b_p_raw']:.4f}), q = {qb:.4f}")
        print(f"    VERDICT: {verdict}")

    print("\n" + "=" * 100)
    print("STANDING FRAMING DISCIPLINE (HANDOFF Sec. 2d, Open Question 3): a")
    print("channel that does not clear is NOT thereby shown to be signal-free.")
    print("This design can distinguish 'clears the floor' from 'does not clear")
    print("at this power' -- it cannot prove absence. The stronger negative is")
    print("as much an overclaim as the optimistic direction.")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
