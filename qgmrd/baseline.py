"""Gaussian HMM baseline.

Fits a 2-state Gaussian HMM to a univariate return series and returns the
posterior probability of the high-variance state, to be compared against the
geometric observables under the same Cohen's d crisis-window metric.

Note: like the paper's RF baseline, this is fit globally here (v0). A causal
variant that fits only on pre-crisis data (the fair benchmark) is implemented
in scripts/causal_eval.py (Task 2).
"""

from __future__ import annotations

import numpy as np
from hmmlearn.hmm import GaussianHMM


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
