"""Multi-crisis panel: the COUNT of crises where a channel clears its OWN
per-crisis 95th-percentile null, across the 15 Table G.10 +/-10 windows.

Causal past-fit preprocessing per crisis (as scripts/causal_eval.py), two nulls
(random matched-length windows; one common circular shift), BH-FDR over the
14-test family, block-bootstrap CIs on the (superseded) median, and realized vol
as the positive control outside the family. The count replaced an earlier median
headline that the control itself invalidated; at 15 crises the count's resolution
tops out below what FDR needs, so all results are EXPLORATORY. Full methodology,
the power ceiling, the persistence-asymmetry caveat, and why H1/H2 are VOID:
see README.

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
    print(f"  MEDIAN |d| = {results[CONTROL]['real_med']:.2f} vs. its own null "
          f"median {results[CONTROL]['a_med']:.2f}  (p = "
          f"{results[CONTROL]['a_p']:.4f})  <-- the median CANNOT see it")
    print(f"  COUNT      = {cc['a']['count']}/{K} crises clear their own "
          f"{COUNT_PCT:.0f}th-pct floor  (p = {cc['a']['p']:.4f}, "
          f"null mean {cc['a']['null_mean']:.2f})  <-- the count CAN")

    # ---- headline: the COUNT --------------------------------------------
    print("\n" + "=" * 100)
    print("HEADLINE -- COUNT of crises clearing their OWN per-crisis floor")
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

    # ---- pre-registered hypotheses: VOID --------------------------------
    print("\n" + "=" * 100)
    print("PRE-REGISTERED HYPOTHESES H1 / H2 -- VOID, NOT FAILED")
    print("=" * 100)
    print("  => H1 is VOID. => H2 is VOID.")

    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
