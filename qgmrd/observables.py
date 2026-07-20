"""Geometric observables computable from a single eigendecomposition.

spectral_entropy : Shannon entropy of the normalized excitation spectrum
                   (paper Eq. 7). High = disordered/crisis; low = ordered.
reduced_purity   : purity tr(rho_A^2) of the reduced density matrix obtained
                   by tracing out subsystem B (paper Eq. 9). Low purity =
                   loss of factor separability, a crisis signature. Depends on
                   the bipartition choice (the paper flags this explicitly).
"""

from __future__ import annotations

import numpy as np


def spectral_entropy(evals: np.ndarray) -> float:
    """Shannon entropy of excitation-energy weights (paper Eq. 7).

    w_n = (E_n - E_0) / sum_m (E_m - E_0), over the excited states n >= 1.
    """
    evals = np.asarray(evals, dtype=float)
    gaps = evals[1:] - evals[0]           # excitation energies above ground
    total = gaps.sum()
    if total <= 0:
        return 0.0
    w = gaps / total
    w = w[w > 0]
    return float(-(w * np.log(w)).sum())


def reduced_purity(psi: np.ndarray, dim_a: int = 2) -> float:
    """Purity tr(rho_A^2) of the reduced density matrix (paper Eq. 9).

    Splits the n-dim state into subsystems A (dim ``dim_a``) and B (dim n/dim_a),
    forms rho_A = tr_B |psi><psi|, and returns tr(rho_A^2) in (0, 1].
    Purity 1 means the ground state factorizes across the cut.
    """
    psi = np.asarray(psi).ravel()
    n = psi.shape[0]
    if n % dim_a != 0:
        raise ValueError(f"dim_a={dim_a} does not divide Hilbert dim n={n}")
    dim_b = n // dim_a
    mat = psi.reshape(dim_a, dim_b)
    rho_a = mat @ mat.conj().T            # (dim_a, dim_a), trace 1
    return float(np.real(np.trace(rho_a @ rho_a)))
