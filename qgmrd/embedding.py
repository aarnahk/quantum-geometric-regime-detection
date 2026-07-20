"""The QCML error Hamiltonian and its ground state.

For a feature vector x = (x_1, ..., x_p) and fixed Hermitian operators {A_k},

    H(x) = (1/2) * sum_k (A_k - x_k I)^2                     (paper Eq. 1)

The quasi-coherent state |psi(x)> is the ground state (lowest-eigenvalue
eigenvector) of H(x). Because H is Hermitian we diagonalize with
``np.linalg.eigh``, which returns real ascending eigenvalues and an
orthonormal eigenbasis.
"""

from __future__ import annotations

import numpy as np


def error_hamiltonian(x: np.ndarray, ops: np.ndarray) -> np.ndarray:
    """Build H(x) for feature vector ``x`` and operator stack ``ops`` (p,n,n)."""
    x = np.asarray(x, dtype=float)
    p, n, _ = ops.shape
    if x.shape[0] != p:
        raise ValueError(f"x has length {x.shape[0]} but there are {p} operators")
    eye = np.eye(n)
    H = np.zeros((n, n), dtype=complex)
    for k in range(p):
        D = ops[k] - x[k] * eye
        H += D @ D
    return 0.5 * H


def ground_state(x: np.ndarray, ops: np.ndarray):
    """Return (eigenvalues, ground_state_vector) for H(x).

    eigenvalues : ascending real array of length n (the full spectrum).
    ground_state_vector : complex unit vector, the lowest-energy eigenvector.
    """
    H = error_hamiltonian(x, ops)
    evals, evecs = np.linalg.eigh(H)  # ascending
    return evals, evecs[:, 0]
