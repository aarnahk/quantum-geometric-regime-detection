"""Crisis window registry -- ONE definition, used by every script in the repo.

WINDOW CONVENTION (decided before any result was computed; see README):
Hammond's Table G.10 windows, extended by +/-10 TRADING days per his Sec. 4.1
convention. Table G.10 gives windows at month granularity ("2020-02 to
2020-04"); the extension pads each side by 10 rows of the trading calendar.

This module exists because the convention was previously inconsistent: an
earlier ad-hoc COVID window (2020-02-19 to 2020-04-30) sat inside G.10's
2020-02..2020-04, while the 2022 and China windows matched G.10 *unextended*.
Two conventions in one repo is worse than either one alone, so the definition
now lives in exactly one place and every script imports it.

The 15 windows below are POST-2005 (the span of our pinned price snapshot) and
are verbatim from Table G.10 -- Hammond's own post-2005 null panel. They are
not to be modified, tuned, or added to: choosing windows after seeing results
is the forking path the honesty architecture exists to prevent.

A knock-on benefit of the extension: the past-fit cutoff is computed from the
EXTENDED window start, so the preprocessing fit now ends ~20 trading days
before the nominal crisis month rather than ~10. That is comfortably clear of
the longest rolling feature window (20 days), which strengthens the Gap 1
argument rather than weakening it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (name, start_month, end_month) -- verbatim from Hammond Table G.10, post-2005.
CRISIS_WINDOWS: list[tuple[str, str, str]] = [
    ("2007 Quant Meltdown", "2007-08", "2007-09"),
    ("2008 GFC",            "2008-09", "2009-03"),
    ("2010 Flash Crash",    "2010-05", "2010-06"),
    ("2011 Euro Crisis",    "2011-07", "2011-10"),
    ("2013 Taper Tantrum",  "2013-05", "2013-07"),
    ("2015 China Crash",    "2015-07", "2015-09"),
    ("2016 Brexit",         "2016-06", "2016-07"),
    ("2018 Volmageddon",    "2018-01", "2018-04"),
    ("2018 Q4 Selloff",     "2018-10", "2018-12"),
    ("2019 Repo Crisis",    "2019-09", "2019-10"),
    ("2020 COVID",          "2020-02", "2020-04"),
    ("2021 Meme/Archegos",  "2021-01", "2021-04"),
    ("2022 Rate Hikes",     "2022-01", "2022-10"),
    ("2023 SVB",            "2023-03", "2023-04"),
    ("2024 Carry Unwind",   "2024-07", "2024-08"),
]

EXTEND_TRADING_DAYS = 10   # Hammond Sec. 4.1: pad each side by 10 trading days
CUTOFF_BUFFER_DAYS = 10    # business days before window start to stop fitting
MIN_PRECUTOFF_ROWS = 200   # skip a crisis with fewer past rows than this

# The three crises carried over from the pre-panel work (Task 2 / null models),
# now expressed in the G.10 +/-10 convention like everything else.
LEGACY_THREE = ["2020 COVID", "2022 Rate Hikes", "2015 China Crash"]


def get(name: str) -> tuple[str, str, str]:
    """Look up one registry entry by name."""
    for entry in CRISIS_WINDOWS:
        if entry[0] == name:
            return entry
    raise KeyError(f"unknown crisis {name!r}; known: {[c[0] for c in CRISIS_WINDOWS]}")


def month_bounds(start_month: str, end_month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Calendar bounds of a "YYYY-MM to YYYY-MM" G.10 window (inclusive)."""
    lo = pd.Timestamp(f"{start_month}-01")
    hi = pd.Timestamp(f"{end_month}-01") + pd.offsets.MonthEnd(0)
    return lo, hi


def crisis_mask(index: pd.DatetimeIndex, start_month: str, end_month: str,
                extend: int = EXTEND_TRADING_DAYS) -> np.ndarray:
    """Boolean mask over ``index`` for a G.10 window extended by +/-``extend``.

    ``index`` is the trading calendar (the feature-frame index), so extending by
    positions in it is extension in TRADING days, which is what Hammond's
    Sec. 4.1 convention means. Clipped at the ends of available data. Returns an
    all-False mask if the window falls entirely outside ``index``.
    """
    lo, hi = month_bounds(start_month, end_month)
    core = np.asarray((index >= lo) & (index <= hi))
    out = np.zeros(len(index), dtype=bool)
    pos = np.flatnonzero(core)
    if pos.size == 0:
        return out
    out[max(0, pos[0] - extend) : min(len(index) - 1, pos[-1] + extend) + 1] = True
    return out


def context(index: pd.DatetimeIndex, start_month: str, end_month: str,
            extend: int = EXTEND_TRADING_DAYS,
            buffer_days: int = CUTOFF_BUFFER_DAYS) -> dict:
    """Everything a script needs for one crisis, derived once, in one place.

    Returns the extended-window mask, its first/last dates, the past-fit cutoff
    (``window_start - buffer_days`` business days), the pre-cutoff row mask, and
    the pre-cutoff row count. Every script uses this so the window convention
    and the cutoff rule cannot drift apart between them.
    """
    mask = crisis_mask(index, start_month, end_month, extend=extend)
    if not mask.any():
        return {"mask": mask, "n_window": 0, "n_pre": 0}
    pos = np.flatnonzero(mask)
    w_start, w_end = index[pos[0]], index[pos[-1]]
    cutoff = w_start - pd.tseries.offsets.BDay(buffer_days)
    pre = np.asarray(index < cutoff)
    return {
        "mask": mask,
        "n_window": int(mask.sum()),
        "start": w_start,
        "end": w_end,
        "cutoff": cutoff,
        "pre": pre,
        "n_pre": int(pre.sum()),
    }
