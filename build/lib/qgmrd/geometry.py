"""Quantum geometry of the QCML embedding (v2).

Implements the differential-geometric layer of the paper:

- metric_pt  : quantum metric g_ab via second-order perturbation theory
               (paper Eq. 10), one eigendecomposition per point.
- metric_fd  : the same metric via phase-aligned finite differences of the
               ground state (paper Eq. 3). Independent computation path.
- The two must agree: 4 g_ab = [F_Q]_ab (Proposition 2). qfi_matrix returns
  the quantum Fisher information matrix as 4g.
- qcrb_bounds: the quantum Cramer-Rao lower bounds Var(x_a) >= 1/4 [g^-1]_aa,
               the estimation-theory reading of the metric.
- berry_curvature_plaquette : gauge-invariant Wilson-loop Berry curvature
               (paper Eq. 6), no phase fixing required.
- curvature_gap_bound : the Theorem 4 upper bound |F_ab| <= 2||dH_a|| ||dH_b|| / gap^2.

Conventions: dH_a = -(A_a - x_a I) for the error Hamiltonian (paper App. A).
"""

from __future__ import annotations

import numpy as np

from .embedding import error_hamiltonian


def _eigh(x: np.ndarray, ops: np.ndarray):
    H = error_hamiltonian(x, ops)
    return np.linalg.eigh(H)


def dH(x: np.ndarray, ops: np.ndarray, a: int) -> np.ndarray:
    """Partial derivative of H(x) in feature direction a: -(A_a - x_a I)."""
    n = ops.shape[1]
    return -(ops[a] - x[a] * np.eye(n))


def metric_pt(x: np.ndarray, ops: np.ndarray) -> np.ndarray:
    """Quantum metric g_ab via the perturbation-theory sum over states (Eq. 10).

    g_ab = Re sum_{n>=1} <psi0|dH_a|psi_n><psi_n|dH_b|psi0> / (E_n - E_0)^2
    """
    p = ops.shape[0]
    evals, evecs = _eigh(x, ops)
    psi0 = evecs[:, 0]
    denom = (evals[1:] - evals[0]) ** 2  # (n-1,)

    # v_a[n] = <psi_n|dH_a|psi0> for excited n
    V = np.empty((p, len(denom)), dtype=complex)
    for a in range(p):
        V[a] = evecs[:, 1:].conj().T @ (dH(x, ops, a) @ psi0)

    g = np.real(np.einsum("am,bm,m->ab", V.conj(), V, 1.0 / denom))
    return 0.5 * (g + g.T)  # symmetrize against roundoff


def _ground(x: np.ndarray, ops: np.ndarray) -> np.ndarray:
    _, evecs = _eigh(x, ops)
    return evecs[:, 0]


def _phase_align(psi_ref: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Rotate psi's global phase so <psi_ref|psi> is real and positive."""
    ov = np.vdot(psi_ref, psi)
    if np.abs(ov) < 1e-14:
        return psi
    return psi * (ov.conj() / np.abs(ov))


def metric_fd(x: np.ndarray, ops: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Quantum metric via phase-aligned finite differences (Eq. 3).

    g_ab = Re <d_a psi|(1 - |psi><psi|)|d_b psi>, with each shifted ground
    state phase-aligned to psi(x) before differencing. Independent of
    metric_pt; their agreement is the Prop. 2 consistency check.
    """
    p = ops.shape[0]
    psi = _ground(x, ops)
    dpsi = np.empty((p, psi.shape[0]), dtype=complex)
    for a in range(p):
        xa = x.copy()
        xa[a] += eps
        psi_a = _phase_align(psi, _ground(xa, ops))
        dpsi[a] = (psi_a - psi) / eps

    proj_dpsi = dpsi - np.outer(dpsi @ psi.conj(), psi)  # (1-|psi><psi|) d_a psi
    g = np.real(proj_dpsi.conj() @ proj_dpsi.T)
    return 0.5 * (g + g.T)


def qfi_matrix(x: np.ndarray, ops: np.ndarray) -> np.ndarray:
    """Quantum Fisher information matrix F_Q = 4 g (Proposition 2)."""
    return 4.0 * metric_pt(x, ops)


def qcrb_bounds(g: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Quantum Cramer-Rao lower bounds: Var(x_a) >= 1/4 [g^+]_aa.

    Uses the pseudo-inverse so rank-deficient metrics (common when the data
    manifold has low intrinsic dimension) are handled gracefully.
    """
    g_inv = np.linalg.pinv(g, rcond=rcond, hermitian=True)
    return 0.25 * np.diag(g_inv)


def qfi_log_pdet(x: np.ndarray, ops: np.ndarray, rcond: float = 1e-10) -> float:
    """log pseudo-determinant of F_Q: total statistical volume (paper Sec. 5.2).

    Spikes when many feature directions become simultaneously distinguishable.
    """
    F = qfi_matrix(x, ops)
    evals = np.linalg.eigvalsh(F)
    pos = evals[evals > rcond * evals.max()] if evals.max() > 0 else evals[evals > 0]
    if len(pos) == 0:
        return -np.inf
    return float(np.sum(np.log(pos)))


def berry_curvature_plaquette(
    x: np.ndarray, ops: np.ndarray, a: int = 0, b: int = 1, eps: float = 1e-3
) -> float:
    """Gauge-invariant plaquette (Wilson loop) Berry curvature F_ab (Eq. 6).

    F_ab = -(1/eps^2) Im log( <p0|pa><pa|pab><pab|pb><pb|p0> )

    where p0, pa, pab, pb are ground states at the four corners of the
    plaquette x, x+eps e_a, x+eps e_a+eps e_b, x+eps e_b. The product of
    overlaps is invariant under any per-corner phase, so no gauge fixing is
    needed.
    """
    ea = np.zeros_like(x)
    eb = np.zeros_like(x)
    ea[a] = eps
    eb[b] = eps

    p0 = _ground(x, ops)
    pa = _ground(x + ea, ops)
    pab = _ground(x + ea + eb, ops)
    pb = _ground(x + eb, ops)

    loop = (
        np.vdot(p0, pa) * np.vdot(pa, pab) * np.vdot(pab, pb) * np.vdot(pb, p0)
    )
    return float(-np.angle(loop) / eps**2)


def curvature_gap_bound(x: np.ndarray, ops: np.ndarray, a: int = 0, b: int = 1) -> float:
    """Theorem 4 upper bound: 2 ||dH_a||_op ||dH_b||_op / gap^2."""
    evals, _ = _eigh(x, ops)
    gap = evals[1] - evals[0]
    na = np.linalg.norm(dH(x, ops, a), ord=2)
    nb = np.linalg.norm(dH(x, ops, b), ord=2)
    return float(2.0 * na * nb / gap**2)
