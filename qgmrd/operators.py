"""Construction of the fixed Hermitian operators {A_k}.

The paper (Sec. 2, "Operator construction") offers two heuristics: PCA-inspired
Pauli-scaled operators, and random Hermitian operators. It reports that random
operators avoid the near-degeneracy PCA-inspired ones can produce and are in
fact best for the Berry channel, so we use them as the v0 default. All
randomness is seeded for reproducibility (the paper fixes seed 42).
"""

from __future__ import annotations

import numpy as np


def random_hermitian_operators(p: int, n: int, seed: int = 42) -> np.ndarray:
    """Return ``p`` random Hermitian ``n x n`` operators.

    Each operator is A = (M + M^dagger) / 2 with the entries of M drawn from a
    standard complex normal distribution.

    Parameters
    ----------
    p : number of operators (one per feature coordinate).
    n : Hilbert-space dimension.
    seed : RNG seed for reproducibility.

    Returns
    -------
    ndarray of shape (p, n, n), complex, each slice Hermitian.
    """
    rng = np.random.default_rng(seed)
    ops = np.empty((p, n, n), dtype=complex)
    for k in range(p):
        M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        ops[k] = (M + M.conj().T) / 2.0
    return ops
