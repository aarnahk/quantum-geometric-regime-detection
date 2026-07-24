"""Classical baselines and the harness control.

``hmm_high_variance_prob`` : 2-state Gaussian HMM posterior, the classical
                             *baseline* detector.
``realized_vol_series``    : trailing realized volatility, the **primary
                             harness control** (see below).

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
