"""Multi-crisis panel (HANDOFF Sec. 8, item 1) -- the critical-path task.

WHY THIS EXISTS. The null-model tests (scripts/null_model.py, HANDOFF Open
Question 3) found that essentially nothing clears its own noise floor on a
single crisis -- including the Gaussian HMM control. That result cannot
distinguish "no signal" from "signal too weak to see with one short window,"
and only the weaker claim is supported. A panel is the only lever that raises
power, so this is the gating task for any per-channel detection claim.

THE HEADLINE STATISTIC IS THE COUNT: the number of crises in which a channel
exceeds ITS OWN per-crisis 95th-percentile null.

THE MEDIAN WAS THE ORIGINAL HEADLINE AND IT WAS WRONG. It is retained below,
labelled superseded, so the change is visible rather than hidden. The argument
for it was that the median of k draws concentrates the NULL at rate 1/sqrt(k)
-- which is true, and measured (the null 95%% interval tightened ~2.5-3x). The
error was forgetting that the REAL statistic is also a median. Against a
HOMOGENEOUS alternative (effect present in most crises) a median is indeed far
more powerful. Against a SPARSE one (effect present in a minority) it is
drastically LESS powerful: the real median collapses into the noise faster than
the null tightens.

The alternative here is sparse, and the positive control proves it. 20-day
realized volatility -- which must separate -- scores a panel MEDIAN of 0.32
against its own null median of 0.45 (6.8th percentile, p = 0.93), because per
crisis it scores 3.05 (2008 GFC), 2.20 (2007), 1.99 (COVID) and 0.06-0.14 on
nine others. Most G.10 windows are simply not volatility events in SPY/DIA. A
median reads the 8th-ranked crisis, which sits in the blind majority.

PROVENANCE OF THE COUNT STATISTIC -- this is what makes it usable. It was
selected on the POSITIVE CONTROL ALONE, with the geometric channels untouched:
among {median, mean, max, count} only the count recovered the control
(p = 0.039 vs 0.93 / 0.24 / 0.19). Choosing a statistic by which one flatters
the channels under test would be a forking path; choosing it on a channel whose
answer is known in advance is not. See scripts/diagnostics.py, test 1b.

THE COUNT'S CEILING, AND WHY NO CONFIRMATORY CLAIM IS AVAILABLE HERE. The count
is integer-coarse. Under binomial(15, 0.05) the upper tail is p = 0.171 for 2
hits, 0.036 for 3, 0.0055 for 4, 0.00065 for 5. The BH rank-1 threshold over
our family of 14 is 0.05/14 = 0.00357, so a channel needs FIVE of fifteen
crises clearing their own floor to survive correction, against 0.75 expected by
chance. That threshold is the ceiling and holds on the arithmetic alone.

The positive control clears THREE -- but that is a REFERENCE POINT, not the cap.
Realized vol is a NARROW detector (vol spikes only), and its 3 hits are exactly
the 3 vol events in the panel (2007, 2008, COVID). It is not an upper bound on
achievable detection: a channel clearing those 3 plus 2 slow-grind crises would
reach 5 and clear -- the very profile this project's Task 1 orthogonality result
(|rho| < 0.13) predicts for a channel decorrelated from vol. Consequently:

  * All count results on the geometric channels are EXPLORATORY this round.
  * No raw p-value here may be presented as if FDR correction were merely
    pending. Correction is not pending; it is unreachable at this resolution.
  * NO new hypotheses are pre-registered against this statistic. Pre-registering
    against a test proven unable to deliver a verdict is the appearance of
    discipline, not discipline. Pre-registration is deferred to the next
    protocol that might have power (multi-asset, or false-alarms-per-year).

PERSISTENCE ASYMMETRY -- a corollary that conditions how the count table may be
read ACROSS rows. Each channel's threshold is calibrated to its own null, and
that null inherits the channel's own persistence. A highly autocorrelated
channel therefore faces a structurally HARDER bar than a less autocorrelated
one. This is correct behaviour for a per-channel null -- it is what makes each
p-value valid, and the reason Hammond's 0.53 is never imported -- but it means
"cleared / did not clear" is NOT apples-to-apples across channels, and the
count inherits that asymmetry directly. The per-channel tau is printed beside
the count table so a reader can see which channels faced harder bars. (This is
the same mechanism already invoked AGAINST the SLD channel elsewhere -- lowest
floor, highest N_eff, the easiest bar and it still showed nothing. The corollary
explains that framing rather than contradicting it; the asymmetry runs in SLD's
favour there.)

PRE-REGISTERED HYPOTHESES H1/H2 ARE VOID -- see the block printed by main().
They were specified against the panel median, which its own control then
invalidated. A test that cannot detect realized volatility delivers no verdict
on any channel, so H1 and H2 are recorded as VOID, not FAILED: "failed" would
imply tested and rejected. Same discipline as the E0 shift-null correction,
where a floored p-value was a bound rather than a rejection.

ANTI-FORKING-PATHS CONSTRAINT. We do NOT search for which channel wins which
crisis type. Hammond tested for per-crisis specialization and found none
(p = 0.31), so any pattern visible across 15 crises is almost certainly noise.
The per-crisis table is DESCRIPTIVE ONLY and carries that caveat wherever it is
printed.

------------------------------------------------------------------------------
PROTOCOL

Windows: the 15 post-2005 entries of Hammond's Table G.10, extended by +/-10
trading days (his Sec. 4.1), from the shared registry qgmrd/crises.py. Not
tuned, not added to.

Per crisis: causal (past-fit) preprocessing exactly as scripts/causal_eval.py
does it -- scaler, PCA and the HMM baseline fit only on rows before
(extended window start - 10 business days), then the full timeline transformed
through those past-fit objects. Crises with < 200 pre-cutoff rows are skipped
and the skip is reported. All 7 channels, PLUS realized volatility as the
POSITIVE CONTROL -- reported everywhere, never in the FDR family, because it is
not a hypothesis under test but the instrument check.

Null tests, both from null_model.py, restructured for a panel. Each supplies
both the per-crisis 95th-percentile thresholds AND the null distribution of the
count, from the SAME draw family so each stays self-consistent:

  (a) Random matched-length windows. Per crisis INDEPENDENTLY, draw a window of
      that crisis's own length elsewhere in that crisis's own causal series,
      non-overlapping it. CAVEAT, stated because it cuts against us: drawing
      independently across crises destroys the cross-crisis dependence the real
      15 values have (they come from one market), so the null is TIGHTER than it
      should be -- an easier bar. Null (a) is anti-conservative, for the count
      exactly as it was for the median.

  (b) Circular shift, ONE COMMON shift applied to all 15 series at once, every
      crisis mask held fixed. Preserves the cross-crisis dependence exactly --
      it slides the whole world under a fixed crisis calendar -- so it is the
      conservative one. CAVEAT specific to the count: because one shift moves
      all 15 crises together, the count's null under (b) is LUMPY -- much higher
      variance than under (a) -- and against an already integer-coarse statistic
      that makes the (b) p-values low-resolution. Stated, not hidden.

The two nulls BRACKET the truth. A channel clearing only (a) is a weaker
result than one clearing both, and is reported as such.

Block-bootstrap CIs (HANDOFF Sec. 8 item 1; the Task 2 writeup defers to these
in several places). The null and the CI answer DIFFERENT questions: the null
asks whether a statistic exceeds chance (location), the CI asks how precisely it
is measured (width).

  NOTE: the CI is computed on the MEDIAN only, and is retained as a precision
  statement about that now-superseded statistic. NO CI is computed on the count:
  it is an integer over 0-15, so a percentile interval on it would be theatre
  rather than information. Saying that is more useful than printing one.

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
from qgmrd.baseline import realized_vol_series  # noqa: E402
from qgmrd.crises import CRISIS_WINDOWS, MIN_PRECUTOFF_ROWS  # noqa: E402
from qgmrd.crises import context as crisis_context  # noqa: E402
from qgmrd.data import load_prices  # noqa: E402
from qgmrd.features import build_features  # noqa: E402
from qgmrd.operators import random_hermitian_operators  # noqa: E402

N_WINDOW_DRAWS = 5000     # null (a) panel medians
N_BOOT = 2000             # block-bootstrap replicates
FDR_ALPHA = 0.05
HMM_BLIND_THRESHOLD = 0.2  # control-sanity flag; flagged, never dropped
CONTROL = "realized_vol_20d"   # positive control; reported, never in the family
COUNT_PCT = 95.0               # per-crisis threshold percentile for the count
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


def panel_null_tests(Z: np.ndarray, masks: np.ndarray,
                     n_draws: int = N_WINDOW_DRAWS, seed: int = SEED) -> dict:
    """Both panel nulls for one channel: real median |d| vs. its own floor.

    ``Z`` is (K, M) -- crisis k's own causal z-scored series -- and ``masks`` is
    (K, M) with crisis k's window. Each crisis window is contiguous in the
    finite-trimmed series, so both nulls reduce to window placement and run off
    the same prefix sums.

    (a) random matched-length windows, drawn per crisis INDEPENDENTLY;
    (b) one COMMON circular shift applied to all K series at once.
    See the module docstring for why (a) is anti-conservative and (b) is not.
    """
    K, M_len = Z.shape
    pre = [_prefix_sums(Z[k]) for k in range(K)]
    spans = [(int(np.flatnonzero(masks[k])[0]), int(np.flatnonzero(masks[k])[-1]),
              int(masks[k].sum())) for k in range(K)]
    real_per_crisis = np.array([_window_abs_d(*pre[k], M_len, [spans[k][0]],
                                              spans[k][2])[0] for k in range(K)])
    real_med = float(np.median(real_per_crisis))

    rng = np.random.default_rng(seed)
    da = np.empty((K, n_draws))
    for k in range(K):
        lo, hi, L = spans[k]
        cand = np.array([s for s in range(0, M_len - L + 1)
                         if (s + L - 1 < lo) or (s > hi)], dtype=np.int64)
        da[k] = _window_abs_d(*pre[k], M_len, rng.choice(cand, n_draws), L)
    null_a = np.median(da, axis=0)

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

    return {
        "real_med": real_med,
        "real_per_crisis": real_per_crisis,
        "a_med": float(np.median(null_a)), "a_lo": float(np.percentile(null_a, 2.5)),
        "a_hi": float(np.percentile(null_a, 97.5)),
        "a_pct": float((null_a < real_med).mean() * 100.0), "a_p": p_a,
        "b_med": float(np.median(null_b)),
        "b_pct": float((null_b < real_med).mean() * 100.0),
        "b_p_raw": p_b_raw, "b_p": max(p_b_raw, 1.0 / (n_eff + 1.0)),
        "tau_max": float(max(taus)), "n_eff": float(n_eff),
        "taus": np.array(taus),
        # per-crisis null draws, (K, n) -- the count statistic takes both its
        # thresholds and its own null distribution from these, per family.
        "draws_a": da, "draws_b": db,
    }


def count_statistic(real_per_crisis: np.ndarray, draws: np.ndarray,
                    pct: float = 95.0) -> dict:
    """How many crises does a channel clear ITS OWN per-crisis floor in?

    ``draws`` is (K, n) of null |d| for each crisis from ONE null family; the
    threshold and the null distribution of the count both come from it, so the
    test stays self-consistent within a family.

    This is the headline statistic. Unlike the median it is sensitive to a
    SPARSE alternative -- an effect present in a few crises and absent in the
    rest -- which is what the positive control showed this panel actually has.

    Integer-coarse by construction: under binomial(15, 0.05) the attainable
    p-values are 0.171 (2 hits), 0.036 (3), 0.0055 (4), 0.00065 (5). There is
    nothing between them, and that resolution limit is the reason no
    confirmatory claim is reachable here (see module docstring).
    """
    thresh = np.percentile(draws, pct, axis=1)
    count = int(np.sum(real_per_crisis > thresh))
    null_counts = (draws > thresh[:, None]).sum(axis=0)
    p = (1 + int(np.sum(null_counts >= count))) / (1 + len(null_counts))
    return {
        "count": count, "p": p, "thresh": thresh,
        "null_mean": float(null_counts.mean()),
        "null_95": float(np.percentile(null_counts, 95)),
        "which": np.flatnonzero(real_per_crisis > thresh),
    }


def align_panel(panel: list[dict], channels: list[str]) -> tuple[dict, dict]:
    """(Z, masks) per channel, trimmed to rows finite for EVERY crisis.

    The causal z-score has a warm-up, and the SLD/Berry channels have their own
    leading NaNs. Trimming to the common finite rows makes the circular shift
    preserve the exact marginal and keeps each crisis window contiguous.
    """
    Zs, Ms = {}, {}
    for ch in channels:
        finite = np.ones(len(panel[0]["z"][ch]), dtype=bool)
        for c in panel:
            finite &= ~np.isnan(c["z"][ch])
        Zs[ch] = np.array([c["z"][ch][finite] for c in panel])
        Ms[ch] = np.array([c["ctx"]["mask"][finite] for c in panel])
        for k in range(len(panel)):
            assert Ms[ch][k].sum() > 0, f"{ch}/{panel[k]['name']}: window trimmed away"
    return Zs, Ms


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
    rv = realized_vol_series(prices["SPY"]).reindex(idx).values
    p = min(8, features.shape[1])

    print("\n" + "=" * 100)
    print("MULTI-CRISIS PANEL -- 15 crises, 7 channels + realized-vol CONTROL")
    print("Windows: Hammond Table G.10 post-2005, extended +/-10 trading days")
    print("(his Sec. 4.1), from qgmrd/crises.py. Causal past-fit preprocessing")
    print("per crisis, exactly as scripts/causal_eval.py.")
    print("HEADLINE = the COUNT of crises where a channel clears its OWN")
    print("per-crisis 95th-pct floor. The MEDIAN is retained below, labelled")
    print("SUPERSEDED. The per-crisis table is DESCRIPTIVE ONLY.")
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
        raw = raw_channels(Xp, ops, returns, hmm_fit=pre)
        raw[CONTROL] = rv          # needs no fitting; same series every crisis
        causal = zscored(raw)
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

    # CONTROL last in every table: it is the visibility reference, not a
    # hypothesis under test, and never enters the FDR family.
    tested = [c for c in panel[0]["z"] if c != CONTROL]
    channels = tested + [CONTROL]
    K = len(panel)

    # ---- per-crisis |d| table (descriptive only) ------------------------
    real_d = {ch: np.array([abs(cohens_d(c["z"][ch][c["ctx"]["mask"]],
                                         c["z"][ch][~c["ctx"]["mask"]]))
                            for c in panel]) for ch in channels}

    print("\n" + "=" * 100)
    print("PER-CRISIS causal |d| -- DESCRIPTIVE ONLY, NOT a specialization claim.")
    print("Hammond tested for per-crisis specialization and found none (p = 0.31),")
    print("so any 'channel X owns crisis type Y' pattern below is almost certainly")
    print("noise. Do not mine it.")
    print("The LAST column is the realized-vol CONTROL -- a VISIBILITY REFERENCE:")
    print("it shows which crises exist at all in this instrument. Where it is")
    print("~0.1, SPY/DIA barely registers the event and no channel's number in")
    print("that row should be read as a miss.")
    print("=" * 100)
    hdr = (f"{'crisis':<22}" + "".join(f"{ch[:11]:>12}" for ch in tested)
           + f"{'|CONTROL':>12}" + "  flag")
    print(hdr)
    print("-" * len(hdr))
    flags = []
    for i, c in enumerate(panel):
        row = (f"{c['name']:<22}" + "".join(f"{real_d[ch][i]:>12.2f}" for ch in tested)
               + f"{real_d[CONTROL][i]:>12.2f}")
        blind = real_d[CONTROL][i] < HMM_BLIND_THRESHOLD
        if blind:
            flags.append((c["name"], real_d[CONTROL][i]))
        print(row + ("  <-- NOT VISIBLE IN SPY/DIA" if blind else ""))
    print("-" * len(hdr))
    med = {ch: float(np.median(real_d[ch])) for ch in channels}
    print(f"{'MEDIAN (superseded)':<22}"
          + "".join(f"{med[ch]:>12.2f}" for ch in tested)
          + f"{med[CONTROL]:>12.2f}")

    print(f"\nVISIBILITY flags (realized-vol control |d| < {HMM_BLIND_THRESHOLD}): "
          f"{len(flags)} of {K} crises.")
    for name, v in flags:
        print(f"  {name:<22} control |d| = {v:.2f}")
    print("These are REPORTED, NOT DROPPED. Removing crises after seeing the")
    print("control is a forking path -- and unnecessary here: the COUNT statistic")
    print("calibrates a floor per crisis, so an invisible crisis contributes null")
    print("draws rather than false positives. Filtering would double-solve it.")

    # ---- assemble the finite-aligned panel matrices ---------------------
    Zs, Ms = align_panel(panel, channels)

    # cross-check the vectorised |d| against the reference implementation
    for ch in channels:
        M_len = Zs[ch].shape[1]
        for k in range(K):
            pos = np.flatnonzero(Ms[ch][k])
            c1, c2, t1, t2 = _prefix_sums(Zs[ch][k])
            fast = _window_abs_d(c1, c2, t1, t2, M_len, [pos[0]], len(pos))[0]
            ref = abs(cohens_d(Zs[ch][k][Ms[ch][k]], Zs[ch][k][~Ms[ch][k]]))
            assert abs(fast - ref) < 1e-8, f"{ch}/{k}: {fast} vs {ref}"

    results = {ch: panel_null_tests(Zs[ch], Ms[ch]) for ch in channels}
    counts = {ch: {fam: count_statistic(results[ch]["real_per_crisis"],
                                        results[ch][f"draws_{fam}"], COUNT_PCT)
                   for fam in ("a", "b")} for ch in channels}

    # ---- the positive control and the ceiling ---------------------------
    cc = counts[CONTROL]
    print("\n" + "=" * 100)
    print("POSITIVE CONTROL AND THE POWER CEILING -- read this before any result")
    print("=" * 100)
    print(f"20-day realized volatility, through the identical downstream.")
    print(f"  MEDIAN |d| = {results[CONTROL]['real_med']:.2f} vs. its own null "
          f"median {results[CONTROL]['a_med']:.2f}  (p = "
          f"{results[CONTROL]['a_p']:.4f})  <-- the median CANNOT see it")
    print(f"  COUNT      = {cc['a']['count']}/{K} crises clear their own "
          f"{COUNT_PCT:.0f}th-pct floor  (p = {cc['a']['p']:.4f}, "
          f"null mean {cc['a']['null_mean']:.2f})  <-- the count CAN")
    print("")
    print("THE CEILING. The count is integer-coarse. Attainable p under")
    print("binomial(15, 0.05): 2 hits -> 0.171, 3 -> 0.036, 4 -> 0.0055,")
    print("5 -> 0.00065. The BH rank-1 threshold over our 14-test family is")
    print(f"0.05/14 = {0.05 / 14:.5f}, so a channel needs FIVE of fifteen crises")
    print("to survive correction (0.75 expected by chance). That is the ceiling,")
    print("on the arithmetic alone.")
    print(f"The control clears {cc['a']['count']} -- a REFERENCE POINT, NOT the "
          "cap. Realized vol is a")
    print("NARROW detector (vol spikes only); its 3 hits are the 3 vol events")
    print("(2007, 2008, COVID). A channel clearing those plus 2 slow-grind crises")
    print("would reach 5 -- the profile Task 1 orthogonality (|rho|<0.13) predicts.")
    print("Therefore:")
    print("  * every count result below is EXPLORATORY;")
    print("  * no raw p here may be read as if FDR correction were merely")
    print("    PENDING -- at this resolution correction is UNREACHABLE.")

    # ---- headline: the COUNT --------------------------------------------
    print("\n" + "=" * 100)
    print("HEADLINE -- COUNT of crises clearing their OWN per-crisis floor")
    print("(a) random matched-length windows, per crisis INDEPENDENTLY. Tighter")
    print("    than the dependent reality -> anti-conservative, as for the median.")
    print("(b) ONE COMMON circular shift. Preserves cross-crisis dependence, so it")
    print("    is conservative -- but one shift moves all 15 crises together, so")
    print("    the count's null under (b) is LUMPY (high variance) and, against an")
    print("    already integer-coarse statistic, its p-values are low-resolution.")
    print("PERSISTENCE ASYMMETRY: each threshold is calibrated to the channel's")
    print("own null, which inherits its own persistence, so a high-tau channel")
    print("faces a structurally HARDER bar. Correct per-channel behaviour -- it is")
    print("what makes each p valid -- but 'cleared/not' is NOT apples-to-apples")
    print("across rows. tau is printed so you can see who faced the harder bar.")
    print("=" * 100)
    print(f"{'channel':<20}{'count(a)':>9}{'p(a)':>9}{'count(b)':>10}{'p(b)':>9}"
          f"{'tau':>7}{'median':>9}{'(superseded)':>14}")
    print("-" * 87)
    for ch in sorted(tested, key=lambda c: (-counts[c]["a"]["count"],
                                            -results[c]["real_med"])):
        r, cA, cB = results[ch], counts[ch]["a"], counts[ch]["b"]
        print(f"{ch:<20}{cA['count']:>9d}{cA['p']:>9.4f}{cB['count']:>10d}"
              f"{cB['p']:>9.4f}{r['tau_max']:>7.0f}{r['real_med']:>9.2f}"
              f"{'p=' + format(r['a_p'], '.3f'):>14}")
    print("-" * 87)
    r, cA, cB = results[CONTROL], cc["a"], cc["b"]
    print(f"{CONTROL + ' (CTRL)':<20}{cA['count']:>9d}{cA['p']:>9.4f}"
          f"{cB['count']:>10d}{cB['p']:>9.4f}{r['tau_max']:>7.0f}"
          f"{r['real_med']:>9.2f}{'p=' + format(r['a_p'], '.3f'):>14}")
    print("\nWhich crises each channel cleared (null (a)):")
    for ch in channels:
        w = counts[ch]["a"]["which"]
        tag = " [CONTROL]" if ch == CONTROL else ""
        print(f"  {ch:<20}" + (", ".join(panel[i]["name"] for i in w)
                               if len(w) else "(none)") + tag)

    # ---- BH-FDR over the 14 count tests ---------------------------------
    keys, pvals = [], []
    for ch in tested:                      # CONTROL is never in the family
        keys += [f"{ch}/(a)", f"{ch}/(b)"]
        pvals += [counts[ch]["a"]["p"], counts[ch]["b"]["p"]]
    q = bh_fdr(pvals)

    print("\n" + "=" * 100)
    print(f"Benjamini-Hochberg FDR over {len(pvals)} tests "
          f"(7 channels x 2 nulls). Expected chance survivors at "
          f"alpha={FDR_ALPHA}: {len(pvals) * FDR_ALPHA:.1f}.")
    print("The CONTROL is excluded: it is the instrument check, not a hypothesis.")
    print(f"CEILING REMINDER: rank-1 needs p <= {0.05 / len(pvals):.5f}, i.e. 5 of")
    print("15 crises. The control manages 3. Nothing here can survive, and that")
    print("is a property of the STATISTIC's resolution, not of the channels.")
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
    print("SCOPE: on the MEDIAN only, retained as a precision statement about")
    print("that now-superseded statistic. NO CI is computed on the COUNT -- it is")
    print("an integer over 0-15, so a percentile interval would be theatre rather")
    print("than information. Saying so is more useful than printing one.")
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

    # ---- pre-registered hypotheses: VOID --------------------------------
    print("\n" + "=" * 100)
    print("PRE-REGISTERED HYPOTHESES H1 / H2 -- VOID, NOT FAILED")
    print("=" * 100)
    print("Both were specified against the PANEL MEDIAN:")
    print("  H1: ground energy E0 clears its null on the panel median.")
    print("  H2: the SLD mixed-state QFI channel clears its null on the panel")
    print("      median.")
    print("")
    print("The median was then invalidated by its own positive control: realized")
    print(f"volatility scores a median of {results[CONTROL]['real_med']:.2f} "
          f"against a null median of {results[CONTROL]['a_med']:.2f} "
          f"(p = {results[CONTROL]['a_p']:.2f}).")
    print("A test that cannot detect realized volatility delivers NO VERDICT on")
    print("any channel.")
    print("")
    print("  => H1 is VOID. => H2 is VOID.")
    print("")
    print("VOID, not FAILED. 'Failed' would mean tested and rejected; these were")
    print("tested with a demonstrably invalid instrument, which yields no")
    print("evidence in EITHER direction. This is the same discipline as the E0")
    print("shift-null correction, where a floored p-value was a BOUND rather")
    print("than a rejection. No pass/fail verdict is reported from the median,")
    print("and in particular the earlier claim that E0 'did not replicate' on")
    print("the panel is WITHDRAWN -- it was a median result.")
    print("")
    print("NO NEW HYPOTHESES ARE PRE-REGISTERED AGAINST THE COUNT. The ceiling")
    print("above shows no confirmatory claim is reachable with it on this panel")
    print("(5 of 15 needed, control gets 3). Pre-registering against a test")
    print("proven unable to deliver a verdict is the APPEARANCE of discipline,")
    print("not discipline. Pre-registration is deferred to the next protocol")
    print("that might have power: multi-asset, or false-alarms-per-year.")

    print("\n" + "=" * 100)
    print("STANDING FRAMING DISCIPLINE (HANDOFF Sec. 2d, Open Question 3): a")
    print("channel that does not clear is NOT thereby shown to be signal-free.")
    print("This design can distinguish 'clears the floor' from 'does not clear")
    print("at this power' -- it cannot prove absence. The stronger negative is")
    print("as much an overclaim as the optimistic direction.")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
