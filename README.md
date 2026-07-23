# Quantum-Geometric Market Regime Detection

A from-scratch reproduction and extension of the QCML geometric-observable
pipeline for detecting market regime shifts (Hammond 2026,
[arXiv:2605.17117](https://arxiv.org/abs/2605.17117)), reframed through
Quantum Fisher Information and the Cramer-Rao bound as estimation on a
statistical manifold.

**Status: v3.** Seven channels are implemented and causally z-scored: spectral
entropy, reduced-density-matrix purity, ground-state energy, Berry-phase rate,
QFI log-determinant, and a novel SLD mixed-state QFI channel — benchmarked
against a Gaussian HMM. The quantum-metric identity 4g = F_Q is verified
numerically. Nothing here claims prediction: following the source paper, these
are *contemporaneous detection* observables, not forecasters.

## Results (SPY/DIA, offline event study)

Cohen's |d|, COVID-2020 window vs. rest. **Offline**: preprocessing is fit on
all data, so these measure crisis-window separability (an event study), not
causal out-of-sample detection. The causal walk-forward version is the v4
milestone.

| method | type | \|d\| |
|---|---|---|
| Gaussian HMM (high-var prob) | baseline | 1.15 |
| Reduced purity | geometric v0 | 1.15 |
| Spectral entropy | geometric v0 | 1.01 |
| Ground energy E0 | geometric v2 | 0.96 |
| Berry phase rate | geometric v2 | 0.71 |
| SLD mixed-state QFI (w=20) | geometric v3 | 0.53 |
| QFI log-det | geometric v2 | 0.10 |

On COVID (a volatility-driven crash), volatility-sensitive channels lead, as
expected; the geometric channels span medium-to-large effects and no single
channel dominates — consistent with Hammond (2026). Proposition 2 identity
verified: corr(g_FD, g_PT) = 1.000000000, max rel. error ~1e-5, confirming
4g = F_Q via two independent metric computations.

## What it does

For each trading day, a rolling feature vector `x_t` defines an error
Hamiltonian `H(x) = 1/2 sum_k (A_k - x_k I)^2`. Its ground state `|psi(x)>`
(via `np.linalg.eigh`) is a unit vector in `C^n`; geometric observables are
read off the ground state and its spectrum:

- **Spectral entropy** — Shannon entropy of the excitation spectrum. High = disordered/crisis.
- **Reduced purity** — `tr(rho_A^2)` after tracing out a subsystem. Low = factor structure breaking down.

Both are converted to causal (past-only) z-scores and compared against a
Gaussian HMM under a Cohen's d crisis-window separability metric.

## Quickstart

```bash
uv venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -e ".[dev]"                # installs numpy/scipy/pandas/sklearn/hmmlearn/pytest
python scripts/run_demo.py                # runs on synthetic data, prints a Cohen's d table
pytest -q                                 # smoke tests
```

Example output (synthetic regime-switch data; a sharp injected vol/corr spike,
so absolute d-values are inflated relative to real crises):

```
method                          type             |d|
----------------------------------------------------
Gaussian HMM (high-var prob)    baseline        2.14
Spectral entropy                geometric       1.71
Reduced purity                  geometric       1.49
```

To run on real data, `uv pip install ".[data]"` and swap `synthetic_prices`
for `load_yfinance(("SPY", "DIA"))` in `scripts/run_demo.py`.

## Honest scope (what v0 is and isn't)

- Operators are **fixed random Hermitian** (paper-endorsed for detection), not
  gradient-learned. Reconstruction-loss operator learning is out of scope.
- Preprocessing (scaler, PCA) is fit **globally / offline**. This is an event
  study, not a causal walk-forward — the same distinction the paper draws
  between its offline Table 3 and its load-bearing walk-forward Table 5.
- Absolute Cohen's d on synthetic data is not comparable to the paper's real-
  crisis numbers; it exists to prove the pipeline produces separable signal.

## v3: SLD mixed-state QFI (novel channel)

A rolling window of embedded market states is a MIXED state
`rho_t = (1/w) sum |psi(x_s)><psi(x_s)|` - "what the market has looked like
lately" as a density matrix. Its quantum Fisher information w.r.t. time is
computed via the symmetric logarithmic derivative (SLD) formalism of
Dingilian, Kurella et al., J. Chem. Phys. (2026), doi:10.1063/5.0316287,
Eqs. (8)-(12): diagonalize rho, build L in its eigenbasis, evaluate
`Tr(L^2 rho)`. This measures how statistically distinguishable the recent
ensemble is becoming from its immediate past - a detector the pure-state
shortcut `4g = F_Q` cannot express. In the w=1 limit the SLD-QFI provably
reduces to `4 g_aa` (tests/test_sld.py::test_pure_state_limit), closing the
loop between the microscopy estimation theory and the QCML metric.

## Roadmap

- **v1-v3 — DONE.** Embedding, 7 channels, SLD mixed-state QFI, offline
  evaluation on real SPY/DIA (see Results). Identities tested in
  `tests/test_geometry.py` and `tests/test_sld.py`.
- **v4 — causal walk-forward (next).** Fit scaler/PCA/operators only on
  pre-crisis rows and re-evaluate — the honest deployment estimate that
  upgrades every number above from "separability" to out-of-sample detection.
- **v5 — multi-crisis panel.** Repeat across the paper's crisis windows
  (2022 rate hikes, 2015 China, 2018 Q4) with block-bootstrap CIs.

## Layout

```
qgmrd/
  operators.py    random Hermitian operators {A_k}
  embedding.py    H(x) and ground state via eigh
  observables.py  spectral entropy, reduced purity
  zscore.py       causal expanding-window z-score (Algorithm 1)
  features.py     returns / vol / momentum / cross-corr
  pipeline.py     embed_series + Cohen's d
  baseline.py     Gaussian HMM
  data.py         synthetic generator + yfinance loader
scripts/run_demo.py
tests/test_smoke.py
```
