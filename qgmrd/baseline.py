"""Classical baselines and the harness controls.

``hmm_high_variance_prob`` : Gaussian HMM baseline detector.
``realized_vol_series``    : primary control, validates on vol crises.
``drawdown_series``        : second control, targets slow-grind declines, fails.
``trailing_return_series`` : third control, same target, non-circular, works better.

The HMM here is fit globally (v0); a causal variant fit only on pre-crisis
data is in scripts/causal_eval.py (Task 2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def realized_vol_series(close: pd.Series, window: int = 20) -> pd.Series:
    """Trailing realized volatility, THE POSITIVE CONTROL for this repo.

    Rolling std of log returns: definitionally elevated during a vol crisis,
    so running it through the same downstream (z-score, masks, Cohen's d,
    both nulls) tests whether the harness can find something unambiguous.

    Chosen over the Gaussian HMM baseline: it fits nothing (isolates
    evaluation from embedding), and the HMM's fits are unstable (12 of 15
    crises converge worse than they started; 2007 pins 94.6% of days near
    zero, scoring |d| = 0.17 vs. this control's 2.20 on the same window).

    A control validates the harness, not a channel; beating it isn't a
    detection. Caveat: strongly autocorrelated (vol clustering), so its own
    noise floor is structurally high. Compare tau against other channels
    before reading a failure to clear as meaningful.
    """
    return np.log(pd.Series(close).astype(float)).diff().rolling(window).std()


def drawdown_series(close: pd.Series, window: int = 252) -> pd.Series:
    """Trailing drawdown depth, the SECOND CONTROL: catches slow-grind
    declines vol can't see.

    ``1 - price / trailing_peak``: 0 at a new high, growing as price falls
    below its trailing peak. Stays elevated for the whole DURATION of a
    decline rather than firing only on sharp days, the shape a months-long
    grind (2022) needs.

    ``window=252`` (~1yr): resets on a market-cycle scale, long enough not to
    erode mid-decline, short enough not to stay elevated for years after a
    crash.

    Caveats: persistent by construction (underwater until price recovers),
    so expect a noise floor at least as high as vol's. Also a circularity
    risk: crisis windows were chosen because they were declines, so a clean
    |d| here is weaker evidence than for vol. Check its null floors, not
    just the raw score.
    """
    price = pd.Series(close).astype(float)
    peak = price.rolling(window, min_periods=1).max()
    return 1.0 - price / peak


def trailing_return_series(close: pd.Series, window: int = 20) -> pd.Series:
    """Trailing cumulative return over ``window`` days, sign-flipped: the
    THIRD CONTROL, positive when price is down over the window.

    Unlike drawdown's peak-tracking (stuck underwater until a new high), both
    endpoints roll forward daily, so memory is bounded by the window itself,
    much shorter even at the same length. Signed, not clipped, like the
    geometric channels.
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
