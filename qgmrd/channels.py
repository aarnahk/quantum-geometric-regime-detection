"""v2 observable channels over a time series of feature vectors.

berry_phase_rate_series : |F_01(x_t) - F_01(x_{t-1})| (paper Eq. 5), Berry
                          curvature in the top two PCA directions via the
                          gauge-invariant plaquette.
qfi_logdet_series       : log pseudo-det of the QFI matrix at each t
                          (the 'QFI Determinant' walk-forward channel).
ground_energy_series    : E_0(x_t), the cheapest detector in the paper
                          (Sec. 6.3) - one eigendecomposition per step.
"""

from __future__ import annotations

import numpy as np

from .embedding import error_hamiltonian
from .geometry import berry_curvature_plaquette, qfi_log_pdet


def ground_energy_series(X: np.ndarray, ops: np.ndarray) -> np.ndarray:
    """E_0(x_t) for each row of X."""
    T = X.shape[0]
    out = np.empty(T)
    for t in range(T):
        H = error_hamiltonian(X[t], ops)
        out[t] = np.linalg.eigvalsh(H)[0]
    return out


def berry_phase_rate_series(
    X: np.ndarray, ops: np.ndarray, a: int = 0, b: int = 1, eps: float = 1e-3
) -> np.ndarray:
    """Berry Phase Rate: absolute step-to-step change of F_ab (paper Eq. 5)."""
    T = X.shape[0]
    F = np.empty(T)
    for t in range(T):
        F[t] = berry_curvature_plaquette(X[t], ops, a=a, b=b, eps=eps)
    rate = np.empty(T)
    rate[0] = np.nan
    rate[1:] = np.abs(np.diff(F))
    return rate


def qfi_logdet_series(X: np.ndarray, ops: np.ndarray) -> np.ndarray:
    """log pseudo-determinant of the QFI matrix at each timestep."""
    T = X.shape[0]
    out = np.empty(T)
    for t in range(T):
        out[t] = qfi_log_pdet(X[t], ops)
    return out
