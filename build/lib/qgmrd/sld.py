"""Mixed-state QFI via the symmetric logarithmic derivative (v3).

This module extends the pipeline beyond the pure-state shortcut 4g = F_Q,
using the SLD formalism of Dingilian, Kurella et al. (J. Chem. Phys. 2026,
doi:10.1063/5.0316287), Eqs. (8)-(12): diagonalize rho, build the SLD in its
eigenbasis, and evaluate F_Q = Tr(L^2 rho).

Finance transposition: a rolling window of embedded market states is
naturally a MIXED state,

    rho_t = (1/w) sum_{s=0}^{w-1} |psi(x_{t-s})><psi(x_{t-s})|,

("what the market has looked like lately," as a density matrix). Its SLD-QFI
with respect to time measures how statistically distinguishable the recent
ensemble is becoming from its immediate past - a regime detector that the
pure-state formula cannot express. In the w=1 limit the window state is pure
and the SLD-QFI reduces to 4 g_aa (tested in tests/test_sld.py), closing the
loop between the microscopy formalism and the QCML metric.
"""

from __future__ import annotations

import numpy as np

from .embedding import error_hamiltonian


def _ground(x: np.ndarray, ops: np.ndarray) -> np.ndarray:
    _, evecs = np.linalg.eigh(error_hamiltonian(x, ops))
    return evecs[:, 0]


def window_density_matrix(states: np.ndarray, weights=None) -> np.ndarray:
    """Density matrix rho = sum_s w_s |psi_s><psi_s| from a stack of pure states.

    Parameters
    ----------
    states : (w, n) complex array, one unit vector per row.
    weights : optional length-w nonnegative weights; uniform if None.
    """
    states = np.asarray(states)
    w, n = states.shape
    if weights is None:
        weights = np.full(w, 1.0 / w)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
    rho = np.zeros((n, n), dtype=complex)
    for s in range(w):
        rho += weights[s] * np.outer(states[s], states[s].conj())
    return 0.5 * (rho + rho.conj().T)  # enforce Hermiticity against roundoff


def sld_operator(rho: np.ndarray, drho: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Symmetric logarithmic derivative L solving d_rho = (L rho + rho L)/2.

    Built in the eigenbasis of rho (JCP Eq. 10):

        L = sum_{k,k' : D_k + D_k' > tol} 2 <k|drho|k'> / (D_k + D_k') |k><k'|

    Pairs in the joint kernel of rho are excluded (they carry no information).
    """
    evals, U = np.linalg.eigh(rho)
    dr = U.conj().T @ drho @ U          # drho in the eigenbasis of rho
    denom = evals[:, None] + evals[None, :]
    mask = denom > tol
    L_eig = np.zeros_like(dr)
    L_eig[mask] = 2.0 * dr[mask] / denom[mask]
    L = U @ L_eig @ U.conj().T
    return 0.5 * (L + L.conj().T)


def sld_qfi(rho: np.ndarray, drho: np.ndarray, tol: float = 1e-12) -> float:
    """Quantum Fisher information F_Q = Tr(L^2 rho) (JCP Eq. 8)."""
    L = sld_operator(rho, drho, tol=tol)
    return float(np.real(np.trace(L @ L @ rho)))


def sld_qfi_feature(
    x: np.ndarray, ops: np.ndarray, a: int, eps: float = 1e-5
) -> float:
    """SLD-QFI of the PURE embedded state w.r.t. feature direction a.

    Exists to verify the pure-state limit: this must equal 4 g_aa from
    geometry.metric_pt (Proposition 2 of the regime paper meeting Eq. 8 of
    the JCP paper). drho is computed by symmetric finite differences of the
    projector, which is gauge-invariant (no phase alignment needed).
    """
    xp = x.copy()
    xm = x.copy()
    xp[a] += eps
    xm[a] -= eps
    pp = _ground(xp, ops)
    pm = _ground(xm, ops)
    psi = _ground(x, ops)
    rho = np.outer(psi, psi.conj())
    drho = (np.outer(pp, pp.conj()) - np.outer(pm, pm.conj())) / (2 * eps)
    return sld_qfi(rho, drho)


def sld_qfi_time_series(
    X: np.ndarray, ops: np.ndarray, window: int = 20
) -> np.ndarray:
    """SLD-QFI of the rolling-window density matrix w.r.t. TIME.

    At each t (for t >= window):
        rho_t   = uniform mixture of the last `window` ground states,
        drho/dt ~ rho_t - rho_{t-1}   (one-day finite difference),
        score_t = Tr(L^2 rho_t).

    High values mean the recent ensemble of market states is changing in a
    statistically distinguishable way - the mixed-state analog of the pure
    QFI channel, and the v3 detector. NaN before `window` history exists.
    """
    T, _ = X.shape
    states = np.stack([_ground(X[t], ops) for t in range(T)])

    out = np.full(T, np.nan)
    rho_prev = None
    for t in range(window - 1, T):
        rho_t = window_density_matrix(states[t - window + 1 : t + 1])
        if rho_prev is not None:
            drho = rho_t - rho_prev            # per-day change of the ensemble
            out[t] = sld_qfi(rho_t, drho)
        rho_prev = rho_t
    return out
