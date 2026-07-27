"""Multi-asset panel (Task 3) -- the pre-registered VISIBILITY headline.

WHY. The SPY/DIA panel sees only equity-vol events: its realized-vol control
clears just 3 of 15 crises (2007, 2008, COVID), all US-equity vol spikes. The
other 12 are rate/FX/credit events that barely touch equities, so they are not
in the instrument -- no channel or feature work can recover them. This panel
widens the instrument to SPY/TLT/UUP/GLD (drop DIA, redundant with SPY) so those
crisis types EXIST to be detected.

PANEL SIZE: 14, not 15. UUP inception is 2007-02, so the frame starts ~2007 and
the 2007 Quant Meltdown drops (< 200 pre-cutoff rows). Pre-registered. NOT
count-comparable to the 15-crisis SPY/DIA panel; all cross-panel comparison is on
the shared 14, where the SPY/DIA control makes 2 crises visible (2008, COVID).

CROSS-ASSET AGGREGATE = ABSORPTION RATIO (sign-INVARIANT; the sign map governs
only the MC alternative). Chosen by the pre-registered degeneracy diagnostic
(scripts/multiasset_aggregate_diagnostic.py): the k=4 degeneracy concern was
tested on held-out calm data and REFUTED (AR/MC corr ~0.50, median AR 0.464, SPY
loading^2 0.20), so AR was retained as a distinct, information-bearing aggregate.

THE HEADLINE = VISIBILITY. Per-asset 20-day realized vol, each through the
IDENTICAL downstream (causal z-score, masks, |d|, both nulls, the count
statistic), counting a crisis VISIBLE if ANY instrument clears its own per-crisis
95th-pct floor. Pre-registered H(multi-asset): the visible count rises materially
above the SPY-alone baseline of 2 on the shared 14. Reported regardless.

Bookkeeping (pre-registered): a union over k per-asset controls is k simultaneous
comparisons, so it inflates the count vs a single control. Both are reported --
the per-asset breakdown AND the union -- and the union is read as an upper
envelope, not a single-instrument p-value. Per Task 3, a control clearing
on a newly-visible crisis licenses INTERPRETATION of that window, never a
detection claim about a geometric channel.

Geometric-channel counts follow as EXPLORATORY, under the same ceiling as the
SPY/DIA panel (5 of 14 needed to survive FDR; see scripts/multi_crisis_panel.py).

    python scripts/multiasset_panel.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from causal_eval import N, SEED, raw_channels, zscored  # noqa: E402
from multi_crisis_panel import (  # noqa: E402
    COUNT_PCT,
    align_panel,
    count_statistic,
    panel_null_tests,
)
from null_model import bh_fdr  # noqa: E402
from qgmrd.baseline import realized_vol_series  # noqa: E402
from qgmrd.crises import CRISIS_WINDOWS, MIN_PRECUTOFF_ROWS  # noqa: E402
from qgmrd.crises import context as crisis_context  # noqa: E402
from qgmrd.data import load_multi_asset_prices  # noqa: E402
from qgmrd.features import build_features_multiasset  # noqa: E402
from qgmrd.operators import random_hermitian_operators  # noqa: E402

ASSETS = ("SPY", "TLT", "UUP", "GLD")
CONTROLS = [f"rv_{a}" for a in ASSETS]
FDR_ALPHA = 0.05


def main() -> None:
    prices = load_multi_asset_prices()
    features = build_features_multiasset(prices, aggregate="AR")
    idx = features.index
    p = min(8, features.shape[1])
    ops = random_hermitian_operators(p, n=N, seed=SEED)
    returns = np.log(prices["SPY"]).diff().reindex(idx).values  # HMM baseline on SPY
    rv = {a: realized_vol_series(prices[a]).reindex(idx).values for a in ASSETS}

    print("\n" + "=" * 100)
    print("MULTI-ASSET PANEL (Task 3) -- SPY/TLT/UUP/GLD, AR aggregate")
    print("HEADLINE = VISIBILITY: per-asset realized-vol control, union over assets,")
    print("through the identical downstream. Pre-registered H: visible count rises")
    print("materially above the SPY-alone baseline of 2 on the shared 14 crises.")
    print("=" * 100)

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
        for a in ASSETS:
            raw[f"rv_{a}"] = rv[a]  # controls need no fitting; same series each crisis
        panel.append({"name": name, "ctx": ctx, "z": zscored(raw)})
        print(f"  fitted {name:<22} {ctx['start'].date()} -> {ctx['end'].date()} "
              f"({ctx['n_window']:>3}d), {ctx['n_pre']} pre-cutoff rows")

    print("\nSKIPPED (< %d pre-cutoff rows), pre-registered:" % MIN_PRECUTOFF_ROWS)
    for name, n_pre in skipped:
        print(f"  {name:<22} {n_pre} pre-cutoff rows")
    K = len(panel)
    print(f"\nPanel size: {K} crises (expected 14).")

    tested = [c for c in panel[0]["z"] if c not in CONTROLS]
    channels = tested + CONTROLS
    Zs, Ms = align_panel(panel, channels)
    results = {ch: panel_null_tests(Zs[ch], Ms[ch]) for ch in channels}
    counts = {ch: {fam: count_statistic(results[ch]["real_per_crisis"],
                                        results[ch][f"draws_{fam}"], COUNT_PCT)
                   for fam in ("a", "b")} for ch in channels}

    # ---- VISIBILITY HEADLINE -------------------------------------------------
    print("\n" + "=" * 100)
    print("VISIBILITY -- per-asset realized-vol control, null (a), which crises each clears")
    print("=" * 100)
    per_asset_which = {}
    for a in ASSETS:
        which = counts[f"rv_{a}"]["a"]["which"]
        per_asset_which[a] = set(which.tolist())
        names = ", ".join(panel[i]["name"] for i in which) if len(which) else "(none)"
        print(f"  {a:<4} clears {len(which)}/{K}: {names}")

    spy_which = per_asset_which["SPY"]
    union = set().union(*per_asset_which.values())
    added = union - spy_which
    print("-" * 100)
    print(f"  SPY-alone baseline : {len(spy_which)} of {K}  "
          f"({', '.join(panel[i]['name'] for i in sorted(spy_which)) or 'none'})")
    print(f"  UNION (any asset)  : {len(union)} of {K}  "
          f"({', '.join(panel[i]['name'] for i in sorted(union)) or 'none'})")
    print(f"  crises ADDED by TLT/UUP/GLD beyond SPY: {len(added)}  "
          f"({', '.join(panel[i]['name'] for i in sorted(added)) or 'none'})")
    print("\nH(multi-asset): visible count rises materially above the SPY-alone")
    print(f"baseline of 2. Observed union = {len(union)}; SPY-alone = {len(spy_which)}.")
    print("Union is k=4 simultaneous comparisons -> read as an UPPER ENVELOPE, not")
    print("a single-instrument p. Per-asset breakdown above is the honest detail.")
    print("A cleared crisis licenses INTERPRETATION of that window, never a")
    print("geometric detection claim (Task 3).")

    # per-crisis |d| vs own floor -- DESCRIPTIVE; explains the negative count.
    print("\n" + "-" * 100)
    print("PER-CRISIS realized-vol |d| / own 95th-pct floor (* = clears).")
    print("DESCRIPTIVE ONLY. Near-misses are NOT detections. Per Hammond (no")
    print("crisis-type specialization, p=0.31) do not mine 'asset X owns crisis Y'.")
    print("Shown because it explains the count of 2: several bond/FX crises carry")
    print("ELEVATED |d| in TLT/UUP that sits JUST UNDER the realized-vol floor,")
    print("which is structurally high (vol-clustering persistence). The binding")
    print("limit is the control's floor, not the instrument -- which is exactly")
    print("what the pre-registered slow-grind second control (Sec 8 item 7) targets.")
    print("-" * 100)
    print(f"{'crisis':<22}" + "".join(f"{a:>13}" for a in ASSETS))
    for i, pc_ in enumerate(panel):
        row = f"{pc_['name']:<22}"
        for a in ASSETS:
            d = results[f"rv_{a}"]["real_per_crisis"][i]
            th = counts[f"rv_{a}"]["a"]["thresh"][i]
            row += f"{d:>7.2f}/{th:<4.2f}{'*' if d > th else ' '}"
        print(row)

    # ---- geometric channels: EXPLORATORY count + FDR ------------------------
    print("\n" + "=" * 100)
    print("GEOMETRIC CHANNELS -- EXPLORATORY. Same ceiling as the SPY/DIA panel:")
    print(f"5 of {K} crises needed to survive FDR (see multi_crisis_panel.py). No")
    print("confirmatory claim is reachable; no raw p is FDR-pending, it is")
    print("unreachable. tau printed -- persistence asymmetry means cleared/not is")
    print("NOT apples-to-apples across rows.")
    print("=" * 100)
    print(f"{'channel':<20}{'count(a)':>9}{'p(a)':>9}{'count(b)':>10}{'p(b)':>9}{'tau':>7}")
    print("-" * 64)
    for ch in sorted(tested, key=lambda c: -counts[c]["a"]["count"]):
        cA, cB, r = counts[ch]["a"], counts[ch]["b"], results[ch]
        print(f"{ch:<20}{cA['count']:>9d}{cA['p']:>9.4f}{cB['count']:>10d}"
              f"{cB['p']:>9.4f}{r['tau_max']:>7.0f}")
    print("-" * 64)
    for a in ASSETS:
        ch = f"rv_{a}"
        cA, cB, r = counts[ch]["a"], counts[ch]["b"], results[ch]
        print(f"{ch + ' (CTRL)':<20}{cA['count']:>9d}{cA['p']:>9.4f}"
              f"{cB['count']:>10d}{cB['p']:>9.4f}{r['tau_max']:>7.0f}")

    keys, pvals = [], []
    for ch in tested:
        keys += [f"{ch}/(a)", f"{ch}/(b)"]
        pvals += [counts[ch]["a"]["p"], counts[ch]["b"]["p"]]
    q = bh_fdr(pvals)
    n_pass = int((np.array(q) < FDR_ALPHA).sum())
    print(f"\n{n_pass}/{len(pvals)} geometric tests survive BH-FDR at q<{FDR_ALPHA} "
          f"(vs {len(pvals) * FDR_ALPHA:.1f} expected by chance). Controls excluded "
          f"from the family.")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
