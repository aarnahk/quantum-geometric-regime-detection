"""Classical baselines and the harness control.

``hmm_high_variance_prob`` : 2-state Gaussian HMM posterior, the classical
                             *baseline* detector.
``realized_vol_series``    : trailing realized volatility, the **primary
                             harness control** (see below).
``drawdown_series``        : trailing drawdown depth, the **second control**,
                             sensitive to slow-grind declines vol misses.
``trailing_return_series`` : fixed-window cumulative return, the **third
                             control**: decline-sensitive like drawdown, but
                             short-memory like realized vol.

Note: like the paper's RF baseline, the HMM is fit globally here (v0). A causal
variant that fits only on pre-crisis data (the fair benchmark) is implemented
in scripts/causal_eval.py (Task 2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def realized_vol_series(close: pd.Series, window: int = 20) -> pd.Series:
    """Trailing realized volatility -- THE POSITIVE CONTROL for this repo.

    Takes a close-price Series and returns a Series on the SAME index, so
    callers reindex onto the feature calendar themselves
    (``realized_vol_series(prices["SPY"]).reindex(idx).values``).

    Rolling standard deviation of log returns. Definitionally elevated during a
    volatility crisis, so pushing it through the identical downstream (causal
    z-score, crisis masks, Cohen's |d|, the two nulls) asks the harness to find
    something that is unambiguously there.

    Why this is the primary control rather than the Gaussian HMM:

      * **It fits nothing.** No scaler, no PCA, no EM. That isolates the
        EVALUATION from the EMBEDDING -- a failure here cannot be blamed on the
        preprocessing, the operators, or the geometry.
      * **The HMM is not a trustworthy instrument.** Diagnostics found 12 of 15
        per-crisis fits reporting convergence on a *negative* final
        log-likelihood delta (EM oscillating at the tolerance floor), and its
        posterior saturation swings with the fit window -- the 2007 fit pins
        94.6% of days near zero and scores |d| = 0.17 where realized vol scores
        2.20 on the same window.

    A positive control can only ever validate the HARNESS. It cannot validate a
    channel, and a channel beating it is not thereby a detection.

    Caveat that matters when reading its floor: realized vol is strongly
    autocorrelated (volatility clustering), so random null windows tend to land
    inside high-vol clumps and its noise floor is structurally HIGH. Compare its
    tau/floor against other channels' before concluding anything from a failure
    to clear -- see the persistence-asymmetry caveat in scripts/null_model.py.
    """
    return np.log(pd.Series(close).astype(float)).diff().rolling(window).std()


def drawdown_series(close: pd.Series, window: int = 252) -> pd.Series:
    """Trailing drawdown depth -- the SECOND POSITIVE CONTROL, sensitive to
    slow-grind declines that realized vol is structurally blind to.

    Takes a close-price Series and returns a Series on the SAME index, so
    callers reindex onto the feature calendar themselves, exactly like
    ``realized_vol_series``.

    ``1 - price / trailing_peak``: 0 at a new high, growing as price falls
    below its trailing ``window``-day peak. Depth-based rather than spike-based,
    so it stays elevated for the DURATION of a decline rather than firing only
    on sharp days -- the shape a months-long grind (2022 Rate Hikes) needs, and
    realized vol (squared daily returns) cannot express regardless of window.

    ``window=252`` (~1 trading year): resets on a market-cycle timescale, not
    an all-time high (elevated for years post-crash) or a single quarter (peak
    erodes mid-decline, understating depth). For crises near/above 252 days
    (2022 runs ~210 with G.10 padding), the peak can still erode late in the
    window -- understates rather than inflates, a conservative direction.

    Caveat: drawdown is highly persistent by construction (underwater until
    price recovers), so expect a noise floor at least as high as realized
    vol's, maybe higher.

    Circularity risk: G.10 windows were chosen because those periods were
    market declines, so this control is more at risk than realized vol of
    "detecting" crises by construction. Read its null floors (does it separate
    crisis windows from ordinary declines, not just calm ones?) before treating
    a clean |d| as validation.
    """
    price = pd.Series(close).astype(float)
    peak = price.rolling(window, min_periods=1).max()
    return 1.0 - price / peak


def trailing_return_series(close: pd.Series, window: int = 20) -> pd.Series:
    """Trailing ``window``-day cumulative return, sign-flipped -- the THIRD
    CONTROL: positive when price is down over the window, negative when up.

    Unlike drawdown's peak-tracking (persistent underwater until a NEW high),
    both endpoints of this window roll forward each day, so it stops reflecting
    a decline once the decline exits the window -- much shorter memory even at
    similar window length. Signed, not clipped: direction shows up in Cohen's d
    the same way as the geometric channels.
    """
    log_price = np.log(pd.Series(close).astype(float))
    return -(log_price - log_price.shift(window))


def hmm_high_variance_prob(
    returns: np.ndarray, n_states: int = 2, seed: int = 42
) -> np.ndarray:
    """Posterior probability of the high-variance regime at each timestep."""
    r = np.asarray(returns, dtype=float).reshape(-1, 1)
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=200,
        random_state=seed,
    )
    model.fit(r)
    variances = model.covars_.reshape(n_states)
    high = int(np.argmax(variances))
    return model.predict_proba(r)[:, high]
