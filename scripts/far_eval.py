"""Causal false-alarm-rate (FAR) evaluation -- roadmap item 4.

PRE-REGISTERED in FAR_PREREGISTRATION.md. This script implements that document
and nothing beyond it; every knob (calibration block, target rate, alarm
definition, era bins, FDR family) was fixed there BEFORE this code ran. If a
knob must change, that is a new pre-registration, not an edit here.

WHAT THIS ASKS, IN ONE LINE. Cohen's |d| and the panel count ask an offline
"is the crisis distribution different from the rest (including the future)."
FAR asks a deployed operator's question: fix one threshold on PAST CALM DATA,
freeze it, run forward -- how often does each channel cry wolf on calm days
(false alarms/yr), and does it fire during real crises (detection + delay)?
The decision at day t uses only data <= t, so FAR is the first genuinely
real-time metric in this repo (it substantially addresses Gap 2's core defect,
though it is NOT walk-forward -- there are no monthly refits; see Sec. 2b).

THE ARCHITECTURE IS DEPLOY-ONCE (FAR_PREREGISTRATION Sec. 3). One detector per
channel: scaler/PCA fit ONCE on the calm calibration block, frozen, and the full
timeline transformed through them; one causal z-scored series per channel; one
threshold tau per channel, calibrated on the block and frozen. Not the panel's
per-crisis refit -- 15 thresholds would be 15 detectors, and "false alarms/yr of
WHICH detector" is then incoherent. One frozen detector is what "deploy" means
and what makes the number comparable to Hammond's ~1 alarm/yr.

THE CALIBRATION BLOCK (FAR_PREREGISTRATION Sec. 3.1). Rows strictly before the
2007 Quant Meltdown's causal cutoff, i.e. index < 2007-07-04:
    2005-02-01 -> 2007-07-03, 609 rows (~2.42 yr), ZERO crisis days.
The block ends before the first extended crisis window with the standard 10-BDay
buffer, so 20-day rolling features cannot leak the 2007 event backward.

TAU CALIBRATION IS THE WHOLE GAME (FAR_PREREGISTRATION Sec. 4). Per channel, tau
is set so the alarm-EVENT rate on the block's calm days equals the target
(1/yr). An alarm is an UPCROSSING (below->above); a detector sitting above tau is
one alarm until it drops back below. COARSENESS, pre-registered: ~2.2 usable
years and ~2 expected events make tau coarsely determined (~0.4/yr granularity).
That is inherent to a zero-crisis pre-everything block, not a bug; the achieved
in-sample rate is reported up to that granularity, the sweep brackets it, and the
by-era forward FAR shows how it landed.

FAR DOES NOT ESCAPE THE POWER PROBLEM (FAR_PREREGISTRATION Sec. 6). At 1/yr a
window of L trading days is detected by a PURE-NOISE detector with prob
~1-exp(-L/252); summed over the 15 real windows that is ~3-4 free detections.
So detection count is judged against a per-channel CHANCE FLOOR from that
channel's own circular-shift null (masks fixed), which inherits its persistence
-- a persistent channel faces a HIGHER floor, so "cleared/didn't" is NOT
apples-to-apples across channels (tau_autocorr printed beside each count). BH-FDR
over the 7-detector family (6 geometric + HMM; control excluded) is applied
UNCONDITIONALLY, fixed before the numbers were seen.

PRE-REGISTERED EXPECTATION (FAR_PREREGISTRATION Sec. 2). Everything was at chance
under Cohen's d and the count, so the base rate is that FAR shows little
detection too. Run because it asks a sharper question, NOT because a positive is
expected; a good FAR profile where d was unremarkable is a genuine finding, and
nothing improving is a stronger negative (two rulers agree). Tau is never tuned
toward the positive.

    python scripts/far_eval.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from causal_eval import N, SEED, raw_channels, zscored  # noqa: E402
from null_model import bh_fdr, integrated_autocorr_time  # noqa: E402
from qgmrd.baseline import realized_vol_series  # noqa: E402
from qgmrd.crises import CRISIS_WINDOWS  # noqa: E402
from qgmrd.crises import context as crisis_context  # noqa: E402
from qgmrd.crises import crisis_mask  # noqa: E402
from qgmrd.data import load_prices  # noqa: E402
from qgmrd.features import build_features  # noqa: E402
from qgmrd.operators import random_hermitian_operators  # noqa: E402

# ---- pre-registered constants (FAR_PREREGISTRATION.md) ----------------------
TRADING_YEAR = 252.0            # trading days per year (a "year" for FAR)
TARGET_FAR = 1.0               # alarms/yr -- Hammond's deployment figure (headline)
FAR_SWEEP = (0.5, 1.0, 2.0)   # descriptive sensitivity only; 1.0 is the headline
CONTROL = "realized_vol_20d"  # positive control; reported, never in the FDR family
FIRST_CRISIS = "2007 Quant Meltdown"   # defines the calibration cutoff
FDR_ALPHA = 0.05

# 5-year calendar eras, fixed a priori on the decade grid (Sec. 7-iii). Edges
# chosen with NO reference to any realized-FAR value -> no gerrymandered bins.
ERA_EDGES = [2005, 2010, 2015, 2020, 2025, 2027]


# --------------------------------------------------------------------------
# Alarm primitives. An alarm is an UPCROSSING (below-or-NaN -> above tau); a
# detector sitting above tau is ONE alarm until it drops back below.
# --------------------------------------------------------------------------

def above_tau(z: np.ndarray, tau: float) -> np.ndarray:
    """Boolean 'firing' mask; NaN (warm-up) counts as not firing."""
    a = np.zeros(z.shape, dtype=bool)
    finite = ~np.isnan(z)
    a[finite] = z[finite] > tau
    return a


def upcross_indices(above: np.ndarray) -> np.ndarray:
    """Indices where the detector crosses from not-firing to firing.

    Position 0 counts as an upcrossing if it starts already firing (an alarm
    active at the very first evaluated day).
    """
    prev = np.concatenate([[False], above[:-1]])
    return np.flatnonzero(above & ~prev)


def event_far(z_segment: np.ndarray, tau: float, n_eval_days: int) -> float:
    """Alarm EVENTS per year over a segment evaluated on ``n_eval_days`` days."""
    n_up = len(upcross_indices(above_tau(z_segment, tau)))
    return n_up / (n_eval_days / TRADING_YEAR) if n_eval_days > 0 else np.nan


# --------------------------------------------------------------------------
# Calibrate-and-freeze. tau is chosen ONLY from the block's calm z-values so the
# block alarm-event rate is as close to the target as integer-event granularity
# allows (Sec. 4). Monotone: higher tau -> fewer upcrossings.
# --------------------------------------------------------------------------

def calibrate_tau(z_block: np.ndarray, target_far: float) -> tuple[float, float, int]:
    """Return (tau, achieved_block_far, target_count) for one channel.

    ``z_block`` is the block portion of the channel's causal z-series (calm by
    construction). Candidates are the block's own finite z-values; we pick the
    tau whose block upcrossing count is closest to target_far * block_years,
    breaking ties toward the HIGHER tau (fewer alarms -- the conservative side).
    """
    finite = z_block[~np.isnan(z_block)]
    n_eval = finite.size
    block_years = n_eval / TRADING_YEAR
    target_count = max(1, int(round(target_far * block_years)))

    cands = np.unique(finite)                     # ascending
    counts = np.array([len(upcross_indices(above_tau(z_block, t))) for t in cands])
    # closest to target; tie -> higher tau (later in ascending cands)
    err = np.abs(counts - target_count)
    best = np.flatnonzero(err == err.min())[-1]
    tau = float(cands[best])
    achieved = counts[best] / block_years
    return tau, achieved, target_count


# --------------------------------------------------------------------------
# Forward detection / delay, and the circular-shift chance floor.
# --------------------------------------------------------------------------

def detect_and_delay(above: np.ndarray, masks: list[np.ndarray],
                     starts: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per crisis: detected?, delay (trading days), early? (carried in at onset).

    Detected = the detector fires on ANY day inside the window. Delay = first
    in-window firing day minus window start. ``early`` marks windows already
    firing at their first day (alarm carried from the shoulder) -- reported, not
    counted as a false alarm.
    """
    K = len(masks)
    det = np.zeros(K, dtype=bool)
    delay = np.full(K, np.nan)
    early = np.zeros(K, dtype=bool)
    for k in range(K):
        inwin = above & masks[k]
        if inwin.any():
            det[k] = True
            first = int(np.flatnonzero(inwin)[0])
            delay[k] = first - starts[k]
            early[k] = bool(above[starts[k]])
    return det, delay, early


def chance_floor(above: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    """Null distribution of the detection COUNT under all circular shifts.

    Slides the firing series under the FIXED crisis masks by every offset
    1..M-1 (deterministic, full enumeration -- the conservative null (b) of the
    panel). Each shift preserves the firing autocorrelation exactly (up to one
    seam) while destroying alignment to the crisis dates, so the distribution of
    "how many windows contain a firing day by chance" IS the floor -- and it
    inherits this channel's persistence, which is the per-channel asymmetry.
    """
    M = above.size
    shifts = np.arange(1, M)
    pos = above.astype(bool)
    null_counts = np.zeros(shifts.size, dtype=np.int32)
    for mask in masks:
        P = np.flatnonzero(mask)
        # firing at (p - s) mod M for each window position p, over all shifts s
        idx = (P[:, None] - shifts[None, :]) % M
        null_counts += pos[idx].any(axis=0).astype(np.int32)
    return null_counts


# --------------------------------------------------------------------------

def era_label(edges: list[int], i: int) -> str:
    lo, hi = edges[i], edges[i + 1]
    return f"{lo}-{hi - 1}"


def main() -> None:
    prices = load_prices(("SPY", "DIA"))
    features = build_features(prices)
    idx = features.index
    ops = random_hermitian_operators(min(8, features.shape[1]), n=N, seed=SEED)
    returns = np.log(prices["SPY"]).diff().reindex(idx).values
    rv = realized_vol_series(prices["SPY"]).reindex(idx).values
    p = min(8, features.shape[1])

    # ---- calibration block: index < 2007 cutoff (Sec. 3.1) --------------
    ctx0 = crisis_context(idx, *_first_crisis_months())
    cutoff = ctx0["cutoff"]
    block = np.asarray(idx < cutoff)             # calibration rows
    forward = ~block                              # deployed rows (>= cutoff)
    n_block = int(block.sum())

    # every crisis mask over the full calendar, and their union
    masks_full = [crisis_mask(idx, sm, em) for _, sm, em in CRISIS_WINDOWS]
    any_crisis = np.zeros(len(idx), dtype=bool)
    for m in masks_full:
        any_crisis |= m

    print("\n" + "=" * 100)
    print("CAUSAL FALSE-ALARM-RATE (FAR) EVALUATION -- SPY/DIA")
    print("Pre-registered in FAR_PREREGISTRATION.md. Deploy-once: fit + freeze on")
    print("the calm calibration block, run forward. NOT walk-forward.")
    print("=" * 100)
    print(f"calibration block : {idx[block][0].date()} -> {idx[block][-1].date()} "
          f"({n_block} rows, ~{n_block / TRADING_YEAR:.2f} yr)")
    print(f"deployed forward  : {idx[forward][0].date()} -> {idx[forward][-1].date()} "
          f"({int(forward.sum())} rows)")

    # ---- (i) CAUSALITY AUDIT (Sec. 7-i) ---------------------------------
    fit_rows = np.flatnonzero(block)
    n_crisis_in_block = int(any_crisis[block].sum())
    forward_in_fit = int((forward & block).sum())
    assert not any_crisis[block].any(), "calibration block contains crisis days!"
    assert fit_rows.max() < np.flatnonzero(forward).min(), "block overlaps forward!"
    print("\n(i) CAUSALITY AUDIT (Sec. 7-i):")
    print(f"    fit/calibration rows subset of block ... rows "
          f"{fit_rows.min()}..{fit_rows.max()}  [{'PASS' if forward_in_fit == 0 else 'FAIL'}]")
    print(f"    crisis days inside the block ........... {n_crisis_in_block}  "
          f"[{'PASS' if n_crisis_in_block == 0 else 'FAIL'}]")
    print(f"    forward days inside the fit set ........ {forward_in_fit}  "
          f"[{'PASS' if forward_in_fit == 0 else 'FAIL'}]")

    # ---- frozen embedding: fit on block, transform full (Sec. 3) --------
    sc = StandardScaler().fit(features.values[block])
    pc = PCA(n_components=p, random_state=SEED).fit(sc.transform(features.values[block]))
    Xp = normalize(pc.transform(sc.transform(features.values)))
    raw = raw_channels(Xp, ops, returns, hmm_fit=block)   # HMM fit on block too
    raw[CONTROL] = rv                                       # control fits nothing
    Z = zscored(raw)                                        # one series per channel

    tested = [c for c in Z if c != CONTROL]                 # 6 geometric + HMM
    channels = tested + [CONTROL]
    K = len(masks_full)

    # forward-portion series, masks, and the calm (non-crisis) forward set.
    starts_fwd = {}
    masks_fwd = {}
    fwd_pos = np.flatnonzero(forward)
    for k in range(K):
        mk = masks_full[k][forward]
        masks_fwd[k] = mk
        starts_fwd[k] = int(np.flatnonzero(mk)[0]) if mk.any() else -1
    calm_fwd = forward & ~any_crisis

    # a crisis is forward-evaluable iff its window lies in the deployed span
    # (deploy-once keeps all 15: every G.10 window starts after the 2007 cutoff)
    evaluable = [bool((masks_full[k] & forward).any()) for k in range(K)]

    # =====================================================================
    # HEADLINE: target = 1/yr
    # =====================================================================
    print("\n" + "=" * 100)
    print(f"HEADLINE -- target FAR = {TARGET_FAR:.1f} alarm/yr, per-channel tau, frozen")
    print("Detection = detector fires on any day inside a crisis window.")
    print("Chance floor = 95th pct of the per-channel circular-shift null count.")
    print("PERSISTENCE ASYMMETRY: the floor inherits each channel's tau_autocorr,")
    print("so a persistent channel faces a HIGHER floor -- 'cleared/didn't' is NOT")
    print("apples-to-apples across rows. tau_ac printed so the harder bars show.")
    print("=" * 100)

    results = {}
    for ch in channels:
        z = Z[ch]
        tau, block_far, target_count = calibrate_tau(z[block], TARGET_FAR)
        above = above_tau(z, tau)
        above_fwd = above[forward]

        # forward calm FAR (overall + by era)
        n_calm = int(calm_fwd.sum())
        up_all = upcross_indices(above)                    # over full series
        up_calm = up_all[calm_fwd[up_all]]                 # upcrossings on calm days
        calm_far = len(up_calm) / (n_calm / TRADING_YEAR)

        # detection + delay on forward windows
        det, delay, early = detect_and_delay(
            above_fwd, [masks_fwd[k] for k in range(K)],
            [starts_fwd[k] for k in range(K)])
        # only forward-evaluable crises count toward the rate
        ev = np.array(evaluable)
        det_count = int((det & ev).sum())
        n_ev = int(ev.sum())
        med_delay = float(np.nanmedian(delay[det & ev])) if (det & ev).any() else np.nan

        # chance floor + p from the circular-shift null (forward series)
        null_counts = chance_floor(above_fwd, [masks_fwd[k] for k in range(K)])
        floor95 = float(np.percentile(null_counts, 95))
        floor_mean = float(null_counts.mean())
        pval = (1 + int(np.sum(null_counts >= det_count))) / (1 + null_counts.size)
        tau_ac = integrated_autocorr_time(z[forward])[0]

        # dynamic-range / transfer diagnostic: the block's causal-z range vs the
        # deployed range, and the forward FAR the highest admissible (block-max)
        # threshold would still give -- this is what shows deploy-once cannot set
        # the operating point (FAR_PREREGISTRATION Sec. 10).
        blk_zmax = float(np.nanmax(z[block]))
        fwd_zmax = float(np.nanmax(z[forward]))
        up_bmax = upcross_indices(above_tau(z, blk_zmax))
        far_at_bmax = int(calm_fwd[up_bmax].sum()) / (n_calm / TRADING_YEAR)

        # fixed-line separation: does the detector cross tau MORE during crises
        # than during calm? (the informative residue -- Sec. 10)
        crisis_exc = 100.0 * above[forward & any_crisis].mean()
        calm_exc = 100.0 * above[calm_fwd].mean()

        results[ch] = dict(tau=tau, block_far=block_far, target_count=target_count,
                           calm_far=calm_far, det=det, delay=delay, early=early,
                           det_count=det_count, n_ev=n_ev, med_delay=med_delay,
                           floor95=floor95, floor_mean=floor_mean, p=pval,
                           tau_ac=tau_ac, above=above, blk_zmax=blk_zmax,
                           fwd_zmax=fwd_zmax, far_at_bmax=far_at_bmax,
                           crisis_exc=crisis_exc, calm_exc=calm_exc)

    # chance floor summary (reported BEFORE the channel numbers, Sec. 6)
    print("\nCHANCE FLOOR (pure-noise expectation, mean of the shift-null count):")
    for ch in channels:
        r = results[ch]
        tag = " [CONTROL]" if ch == CONTROL else ""
        print(f"  {ch:<20} floor(mean) {r['floor_mean']:.2f}  "
              f"floor(95th) {r['floor95']:.0f}/{r['n_ev']}  "
              f"tau_ac {r['tau_ac']:>4.0f}{tag}")

    print(f"\n{'channel':<20}{'tau':>8}{'blkFAR':>8}{'calmFAR':>9}"
          f"{'detect':>8}{'floor95':>9}{'medDelay':>10}{'p':>9}{'tau_ac':>8}")
    print("-" * 97)
    for ch in sorted(tested, key=lambda c: (-results[c]["det_count"], results[c]["p"])):
        r = results[ch]
        dd = f"{r['med_delay']:.0f}d" if not np.isnan(r["med_delay"]) else "--"
        print(f"{ch:<20}{r['tau']:>8.2f}{r['block_far']:>8.2f}{r['calm_far']:>9.2f}"
              f"{str(r['det_count']) + '/' + str(r['n_ev']):>8}{r['floor95']:>9.0f}"
              f"{dd:>10}{r['p']:>9.4f}{r['tau_ac']:>8.0f}")
    print("-" * 97)
    r = results[CONTROL]
    dd = f"{r['med_delay']:.0f}d" if not np.isnan(r["med_delay"]) else "--"
    print(f"{CONTROL + ' (CTRL)':<20}{r['tau']:>8.2f}{r['block_far']:>8.2f}"
          f"{r['calm_far']:>9.2f}{str(r['det_count']) + '/' + str(r['n_ev']):>8}"
          f"{r['floor95']:>9.0f}{dd:>10}{r['p']:>9.4f}{r['tau_ac']:>8.0f}")

    print("\nWhich crises each channel detected (forward, target 1/yr):")
    names = [n for n, _, _ in CRISIS_WINDOWS]
    for ch in channels:
        w = [names[k] for k in range(K) if results[ch]["det"][k] and evaluable[k]]
        e = [names[k] for k in range(K) if results[ch]["early"][k] and evaluable[k]]
        tag = " [CONTROL]" if ch == CONTROL else ""
        line = ", ".join(w) if w else "(none)"
        if e:
            line += "   [early: " + ", ".join(e) + "]"
        print(f"  {ch:<20}{line}{tag}")

    # ---- DYNAMIC RANGE / TRANSFER (why deploy-once is infeasible) --------
    print("\n" + "=" * 100)
    print("DYNAMIC RANGE / TRANSFER -- the block's causal-z range vs the deployed")
    print("range. A threshold set on the compressed calm-block range sits low in")
    print("the forward distribution and does NOT transfer: forward FAR @ block-tau")
    print("is off the 1/yr target in both directions. 'FAR@blkZmax' is the forward")
    print("FAR the HIGHEST admissible (block-max) threshold would still give.")
    print("=" * 100)
    print(f"{'channel':<20}{'blkZmax':>9}{'fwdZmax':>9}{'FAR@blkTau':>12}{'FAR@blkZmax':>13}")
    print("-" * 63)
    for ch in channels:
        r = results[ch]
        tag = " C" if ch == CONTROL else ""
        print(f"{ch:<20}{r['blk_zmax']:>9.2f}{r['fwd_zmax']:>9.2f}"
              f"{r['calm_far']:>12.2f}{r['far_at_bmax']:>13.2f}{tag}")

    # ---- FIXED-LINE SEPARATION (the informative residue) ----------------
    print("\n" + "=" * 100)
    print("FIXED-LINE SEPARATION -- does the detector cross its tau MORE during")
    print("crises than during calm? This is the clean residue: raw FAR/counts are")
    print("contaminated by drift+chattiness, but the crisis-vs-calm exceedance")
    print("RATIO at a fixed line is interpretable. Ratio ~1 or below = no positive")
    print("separation. Channels with tiny exceedance% fire too rarely to test.")
    print("=" * 100)
    print(f"{'channel':<20}{'tau':>8}{'crisisExc%':>12}{'calmExc%':>11}{'ratio':>9}")
    print("-" * 60)
    for ch in channels:
        r = results[ch]
        ratio = (r["crisis_exc"] / r["calm_exc"]) if r["calm_exc"] > 0 else float("inf")
        tag = " C" if ch == CONTROL else ""
        rs = f"{ratio:.2f}" if np.isfinite(ratio) else "inf"
        print(f"{ch:<20}{r['tau']:>8.2f}{r['crisis_exc']:>12.1f}"
              f"{r['calm_exc']:>11.1f}{rs:>9}{tag}")

    # ---- (ii) IN-SAMPLE BLOCK FAR CHECK (Sec. 7-ii) ---------------------
    print("\n" + "=" * 100)
    print("(ii) IN-SAMPLE CHECK -- block FAR must ~ target by construction")
    print(f"     (up to integer-event granularity of a ~{n_block / TRADING_YEAR:.1f}-yr block;")
    print("      a large miss means the calibration did NOT execute, i.e. a bug)")
    print("=" * 100)
    for ch in channels:
        r = results[ch]
        ok = "OK" if abs(r["block_far"] - TARGET_FAR) <= 0.6 else "CHECK"
        print(f"  {ch:<20} block FAR {r['block_far']:>5.2f}/yr  "
              f"(target {TARGET_FAR:.1f}, {r['target_count']} events)  [{ok}]")

    # ---- (iii) FORWARD CALM FAR BY ERA (Sec. 7-iii) ---------------------
    print("\n" + "=" * 100)
    print("(iii) FORWARD CALM FAR BY 5-YEAR ERA -- the deploy-once drift diagnostic")
    print("Bins fixed a priori on the decade grid (NOT placed after seeing jumps).")
    print("A RISING late-era FAR reflects a frozen threshold under regime drift and")
    print("is NOT read as detection (FAR_PREREGISTRATION Sec. 8, first caveat).")
    print("=" * 100)
    years = idx.year.values
    era_calm = []
    for i in range(len(ERA_EDGES) - 1):
        lo, hi = ERA_EDGES[i], ERA_EDGES[i + 1]
        sel = calm_fwd & (years >= lo) & (years < hi)
        era_calm.append(sel)
    hdr = f"{'channel':<20}" + "".join(f"{era_label(ERA_EDGES, i):>12}"
                                       for i in range(len(era_calm)))
    print(hdr)
    print(f"{'(calm days)':<20}" + "".join(f"{int(s.sum()):>12}" for s in era_calm))
    print("-" * len(hdr))
    for ch in channels:
        above = results[ch]["above"]
        up = upcross_indices(above)
        cells = []
        for s in era_calm:
            nd = int(s.sum())
            n_up = int(s[up].sum())
            cells.append(f"{(n_up / (nd / TRADING_YEAR)) if nd else float('nan'):>12.2f}")
        tag = " C" if ch == CONTROL else ""
        print(f"{ch:<20}" + "".join(cells) + tag)

    # ---- BH-FDR over the 7-detector family (Sec. 6, unconditional) ------
    print("\n" + "=" * 100)
    print("BH-FDR over the 7-detector family (6 geometric + HMM), control EXCLUDED.")
    print("Applied UNCONDITIONALLY -- fixed before the numbers, run regardless.")
    print(f"Expected chance survivors at alpha={FDR_ALPHA}: {len(tested) * FDR_ALPHA:.2f}.")
    print("=" * 100)
    pvals = [results[ch]["p"] for ch in tested]
    q = bh_fdr(pvals)
    print(f"{'detector':<20}{'detect':>9}{'floor95':>9}{'raw p':>10}{'BH q':>10}{'q<.05':>8}")
    print("-" * 66)
    order = np.argsort(q)
    for i in order:
        ch = tested[i]
        r = results[ch]
        print(f"{ch:<20}{str(r['det_count']) + '/' + str(r['n_ev']):>9}"
              f"{r['floor95']:>9.0f}{pvals[i]:>10.4f}{q[i]:>10.4f}"
              f"{('yes' if q[i] < FDR_ALPHA else 'no'):>8}")
    n_pass = int((np.array(q) < FDR_ALPHA).sum())
    print(f"\n{n_pass}/{len(tested)} detectors survive BH-FDR at q < {FDR_ALPHA}.")

    # ---- sensitivity sweep (descriptive only) --------------------------
    print("\n" + "=" * 100)
    print("SENSITIVITY SWEEP -- detection count vs target FAR (DESCRIPTIVE ONLY;")
    print("1/yr is the headline, the sweep is never mined for a flattering target).")
    print("=" * 100)
    print(f"{'channel':<20}" + "".join(f"{'@' + str(t) + '/yr':>12}" for t in FAR_SWEEP))
    print("-" * (20 + 12 * len(FAR_SWEEP)))
    ev = np.array(evaluable)
    for ch in channels:
        z = Z[ch]
        cells = []
        for t in FAR_SWEEP:
            tau, _, _ = calibrate_tau(z[block], t)
            above_fwd = above_tau(z, tau)[forward]
            det, _, _ = detect_and_delay(above_fwd, [masks_fwd[k] for k in range(K)],
                                         [starts_fwd[k] for k in range(K)])
            cells.append(f"{int((det & ev).sum())}/{int(ev.sum())}")
        tag = " C" if ch == CONTROL else ""
        print(f"{ch:<20}" + "".join(f"{c:>12}" for c in cells) + tag)

    print("\n" + "=" * 100)
    print("STANDING DISCIPLINE: a channel not clearing its floor is NOT shown to be")
    print("signal-free -- only that FAR at this power did not detect it. And the 12")
    print("non-vol crises have no working positive control, so a FAR 'detection'")
    print("there is UNVALIDATED (carried from the panel). See FAR_PREREGISTRATION")
    print("Sec. 8. Naming: this is a causal frozen-threshold FAR event study, NOT")
    print("walk-forward.")
    print("=" * 100 + "\n")


def _first_crisis_months() -> tuple:
    for name, sm, em in CRISIS_WINDOWS:
        if name == FIRST_CRISIS:
            return sm, em
    raise KeyError(FIRST_CRISIS)


if __name__ == "__main__":
    main()
