"""Quantum-Geometric Market Regime Detection (v0).

A minimal, honest reproduction of the QCML geometric-observable pipeline
(Hammond 2026, arXiv:2605.17117) restricted to the two observables that
fall out of a single eigendecomposition per timestep: spectral entropy and
reduced-density-matrix purity. Berry-phase rate and the QFI/Cramer-Rao
framing are deferred to v1/v2 (see README roadmap).
"""

from .operators import random_hermitian_operators
from .embedding import error_hamiltonian, ground_state
from .observables import spectral_entropy, reduced_purity
from .zscore import causal_zscore
from .features import build_features
from .pipeline import embed_series, cohens_d
from .baseline import hmm_high_variance_prob
from .data import synthetic_prices
from .geometry import (
    metric_pt,
    metric_fd,
    qfi_matrix,
    qcrb_bounds,
    berry_curvature_plaquette,
)
from .channels import (
    berry_phase_rate_series,
    qfi_logdet_series,
    ground_energy_series,
)
from .sld import (
    window_density_matrix,
    sld_operator,
    sld_qfi,
    sld_qfi_time_series,
)

__all__ = [
    "random_hermitian_operators",
    "error_hamiltonian",
    "ground_state",
    "spectral_entropy",
    "reduced_purity",
    "causal_zscore",
    "build_features",
    "embed_series",
    "cohens_d",
    "hmm_high_variance_prob",
    "synthetic_prices",
    "metric_pt",
    "metric_fd",
    "qfi_matrix",
    "qcrb_bounds",
    "berry_curvature_plaquette",
    "berry_phase_rate_series",
    "qfi_logdet_series",
    "ground_energy_series",
    "window_density_matrix",
    "sld_operator",
    "sld_qfi",
    "sld_qfi_time_series",
]
