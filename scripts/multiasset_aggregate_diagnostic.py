"""Pre-registered AR-vs-MC degeneracy diagnostic (HANDOFF Task 3).

Decides the single cross-asset aggregate for the multi-asset panel BEFORE the
panel is run, on a HELD-OUT, genuinely-calm reference -- never on the 14
evaluation crises (choosing a method by its test-set behavior is the optimism
bias avoided everywhere else).

Calm reference = 2014 alone (k=4, all assets live; 2013-taper tail clears ~60
trading days before 2014, so lookbacks are crisis-free). 2004-2006 was preferred
on calmness but declined: UUP is absent pre-2007, making it a k=3 subsystem whose
AR floor (1/3) is not comparable to the k=4 panel's (1/4). 2012/2017 dropped:
mid-2012 Euro haven flows and 2013 taper stress contaminate TLT.

Numerical rule + tie-breaker, committed before running:
  D1 dynamic range : median(AR)          AR-safe <0.80 | ambiguous [0.80,0.85) | degenerate >=0.85
  D2 redundancy    : |corr(AR, MC)|       AR-safe <0.90 | ambiguous [0.90,0.95) | redundant  >=0.95
  -> use AR ONLY if BOTH are AR-safe; any other outcome (degenerate, redundant,
     or ambiguous on either axis) selects MC. Ties break toward MC, the more
     interpretable and less degeneracy-prone object at k=4.
  corroborating (non-decisive) : median SPY top-eigenvector loading^2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qgmrd import crises
from qgmrd.data import load_multi_asset_prices
from qgmrd.multiasset import (
    CORR_WINDOW,
    absorption_ratio_series,
    risk_aligned_mean_corr_series,
    top_eigvec_loading_series,
)

D1_SAFE, D1_HARD = 0.80, 0.85   # median AR: <safe -> AR-safe; >=hard -> degenerate
D2_SAFE, D2_HARD = 0.90, 0.95   # |corr|:   <safe -> AR-safe; >=hard -> redundant


def _band(value: float, safe: float, hard: float) -> str:
    if value < safe:
        return "AR-safe"
    if value < hard:
        return "AMBIGUOUS"
    return "MC"


def main() -> None:
    prices = load_multi_asset_prices()
    logret = np.log(prices).diff()
    idx = logret.index

    ar = absorption_ratio_series(logret)
    mc = risk_aligned_mean_corr_series(logret)
    spy_load = top_eigvec_loading_series(logret, asset="SPY")

    # Union of all 14 extended evaluation windows (2007 is absent from this frame).
    crisis = np.zeros(len(idx), dtype=bool)
    for name, s, e in crises.CRISIS_WINDOWS:
        crisis |= crises.crisis_mask(idx, s, e)
    # A day's 60-day lookback is crisis-clean iff the prior CORR_WINDOW rows carry
    # no crisis day.
    crisis_in_lookback = pd.Series(crisis, index=idx).rolling(CORR_WINDOW).sum().to_numpy()
    lookback_clean = crisis_in_lookback == 0

    in_ref = (idx >= "2014-01-01") & (idx <= "2014-12-31")
    decide = in_ref & lookback_clean & ar.notna().to_numpy() & mc.notna().to_numpy()
    ar_d, mc_d, load_d = ar[decide], mc[decide], spy_load[decide]

    med_ar = float(ar_d.median())
    abs_corr = abs(float(np.corrcoef(ar_d.to_numpy(), mc_d.to_numpy())[0, 1]))
    med_load = float(load_d.median())

    d1_band = _band(med_ar, D1_SAFE, D1_HARD)
    d2_band = _band(abs_corr, D2_SAFE, D2_HARD)
    choice = "AR" if (d1_band == "AR-safe" and d2_band == "AR-safe") else "MC"

    print(f"calm reference = 2014; decision days (lookback-clean, non-NaN): "
          f"{int(decide.sum())} of {int(in_ref.sum())} in-year days")
    print(f"  AR range on 2014 : [{ar_d.min():.3f}, {ar_d.max():.3f}]")
    print()
    print(f"D1  median(AR)        = {med_ar:.3f}   -> {d1_band}   "
          f"(safe <{D1_SAFE}, ambiguous <{D1_HARD}, else degenerate)")
    print(f"D2  |corr(AR, MC)|    = {abs_corr:.3f}   -> {d2_band}   "
          f"(safe <{D2_SAFE}, ambiguous <{D2_HARD}, else redundant)")
    print(f"    median SPY load^2 = {med_load:.3f}   (>0.5 flags SPY domination)")
    print()
    print(f"==> AGGREGATE CHOICE: {choice}   (AR only if both axes AR-safe; ties -> MC)")


if __name__ == "__main__":
    main()
