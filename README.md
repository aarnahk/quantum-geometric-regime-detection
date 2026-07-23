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
causal out-of-sample detection. The causal (past-fit) preprocessing version is
below (Task 2); closing the remaining event-study metric gap (Gap 2) is the
later milestone.

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

Channel correlation (SPY/DIA, causal z-scores, Pearson and Spearman): the
SLD mixed-state QFI channel correlates with every other channel at
|ρ| < 0.13 (both metrics), roughly half Hammond's own geometric-classical
decorrelation benchmark (mean |ρ| ≈ 0.22). Its highest correlation is with
spectral entropy (ρ = 0.122 Pearson, 0.127 Spearman) — its closest
conceptual relative — and even that is well within noise. Combined with its
COVID Cohen's d = 0.53 (a real, medium effect, not noise-level), this
answers Open Question 1: the mixed-state generalization is empirically
distinct from the pure-state channels, not merely a redundant formalism
exercise reusing spectral information under a different name.

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

**Novelty, verified against source (not just paper text).** Cloned Hammond's
repo and grepped `qcml_geometry/*.py` for `sld`, `symmetric_log`,
`logarithmic_derivative`: zero matches. His nearest neighbor,
`QuantumRelativeEntropyDetector`, builds a mixed state the same way this
repo does — an averaged `sum |psi_s><psi_s|` over past states — but computes
quantum relative entropy from it, not SLD-based QFI. Those are different
Riemannian metrics (Kubo–Mori vs. SLD/Bures) that coincide only when states
commute, so this is a source-confirmed distinction, not a
paper-vs-paper inference. One design difference: his reference state uses an
expanding window (all past states), this repo's `rho_t` uses a fixed
rolling window `w`.

## Channel correlation matrix (Open Question 1, answered)

```bash
python scripts/channel_correlations.py
```

Pairwise Pearson and Spearman correlation of all seven causal z-scored
channels on real SPY/DIA, over the full series (not just the COVID window) —
this tests day-to-day agreement between detectors, a stronger redundancy
check than co-spiking during one crisis.

**SLD's largest |correlation| with any other channel is ~0.13** (vs.
spectral entropy), with every other SLD pair below that — decorrelated even
relative to Hammond's reported mean |rho| ~ 0.22 between geometric and
classical channel families. Pearson and Spearman agree closely on every SLD
pair (largest gap 0.06), so this isn't an artifact of a few extreme days.
**Answer: the SLD channel is empirically non-redundant, not a repackaging of
the pure-state channels it generalizes.** Combined with its d = 0.53 on
COVID (real, if mid-pack, separability), this is the empirical case for the
extension — not just the pure-state-limit proof.

For context, the other six channels correlate with each other much more
strongly (e.g. reduced purity vs. `E0` at -0.90, Berry phase rate vs. QFI
log-det at ~0.70-0.74, spectral entropy vs. the HMM baseline at ~0.64-0.71) —
they're largely re-detecting the same crisis from different angles, which is
what makes SLD's near-zero correlation with all of them notable rather than
just noisy.

Caveat: Pearson and Spearman diverge by >0.1 for two pairs, both involving
Berry phase rate (vs. spectral entropy, vs. HMM) — a monotonic-but-nonlinear
relationship, plausibly because Berry rate is a step-difference observable.
Doesn't affect the SLD conclusion but is a caveat on Berry-rate comparisons
specifically.

This result is offline (global scaler/PCA fit, same caveat as the table
above) — it answers whether channels agree with each other, not whether any
one of them is causally clean; that's Task 2.

## Causal (past-fit) preprocessing (Task 2)

```bash
python scripts/causal_eval.py
```

Every |d| above is **offline**: scaler and PCA are fit on the whole series,
crisis included. This script removes that look-ahead — per crisis it fits
`StandardScaler`, `PCA`, and the HMM baseline only on rows before
`cutoff = crisis_start − 10 business days`, then transforms the full timeline
through those past-fit objects. Operators are data-independent and unchanged.
The 10-day buffer stops 20-day rolling-vol features from leaking the crisis
backward. This closes **Gap 1** (leaky preprocessing); **Gap 2** (Cohen's *d*
still scores the crisis window against future days) remains, so the causal
column is still offline separability, not real-time detection.

**No confidence intervals yet.** Every number below is a point estimate on a
single realized path of three crises. Block-bootstrap CIs are a pending task;
until they land, **no delta in this table is certified distinguishable from
noise**, and "barely moved" means the point estimate barely moved, not a
statistical claim.

Offline → causal Cohen's |d| (SPY/DIA):

| channel | COVID 2020 | Rate Hikes 2022 | China 2015 |
|---|---|---|---|
| Reduced purity | 1.15 → 1.13 | 0.95 → 0.90 | 0.97 → 0.94 |
| Spectral entropy | 1.01 → 0.97 | 1.18 → 1.22 | 0.71 → 0.49 |
| Ground energy E0 | 0.96 → 0.95 | 1.64 → 1.61 | 0.48 → 0.65 |
| Berry phase rate | 0.71 → 0.70 | 0.99 → 0.91 | 0.24 → 0.79 |
| SLD QFI (w=20) | 0.53 → 0.66 | 0.01 → 0.07 | 0.56 → 0.54 |
| QFI log-det | 0.10 → 0.22 | 1.20 → 1.09 | 0.82 → 0.18 |
| Gaussian HMM | 1.15 → 1.13 | 1.07 → 1.06 | 0.04 → 0.02 |

**China 2015 is uninterpretable and is not analyzed channel-by-channel.** The
HMM control scores 0.04 offline — it essentially cannot see a crisis in
SPY/DIA over Jul–Sep 2015 (mostly quiet, with one violent week around Aug
20–26 that Cohen's *d* dilutes against the calm remainder). Independently, the
causal embedding agrees least with the offline one there (row-wise cosine mean
0.93, min 0.41 — some days near-orthogonal). With a blind control and the
largest embedding divergence, no individual channel's swing in that column is
trustworthy; those numbers are reported for completeness only.

**Does the preprocessing change reach the channels?** (`causal_eval.py` prints
all of this.) The preprocessing *parameters* diverge modestly and in a
concentrated way: the top 3 PCA axes are near-identical across crises
(|cos| ≥ 0.997), and the scaler-scale shift is dominated by a single feature —
rolling cross-correlation `xcorr20` (Δscale/scale = 0.41 / 0.36 / 0.60, since
correlations reprice hard in a crisis) — while the other ten features shift
≤ 0.09 (median ~0.05, only 1/11 above 0.10). That divergence *does* reach the
daily embedding (row-wise causal-vs-offline cosine mean 0.98 COVID, 0.99 2022,
0.93 China; min 0.82 on COVID). **But it does not reach the z-scored channel
series** on the two interpretable crises: causal-vs-offline series
correlations are 0.94–0.999 on COVID and 2022 (most ≥ 0.98; the frame-
sensitive Berry and QFI-log-det lowest at 0.94–0.96). So the puzzle of "the
daily embedding differs yet |d| barely moves" resolves as **channel
robustness, not metric coarseness**: the trailing causal z-score washes out
per-day embedding jitter into a detector series ~0.98 correlated with the
offline one. Where |d| barely moves, the underlying series barely move too —
the "Cohen's *d* is too coarse to see series-level change" failure mode is
*not* what the data show here. (On China the series correlations do drop to
0.78–0.90, i.e. the perturbation reaches the series there — but China is
uninterpretable for the separate reason above.)

**Reduced purity shows no collapse under causal fitting** — its point estimate
stays high (1.13 / 0.90 / 0.94, ≤ 0.05 from the offline values). The right
comparison is Hammond's Table 3 reduced-purity median of *d* ≈ 0.83, which is
**already a per-crisis past-fit number** (his §5.1: scaler, PCA, *and*
operators fit only on pre-crisis data) — the same protocol structure
`causal_eval.py` uses. All three of our causal values *exceed* that median, but
"above" is not self-evidently good: these are three point estimates from a
materially different pipeline set against a 17-crisis median. The differences
are real, not cosmetic:

- **Features:** 11 raw features here vs. his ~52 enriched.
- **Dimensionality:** 8 PCA components here vs. his 15.
- **Crisis panel:** three crises here, a subset of his 17.
- **Operator treatment:** Hammond fits operators on pre-crisis data too (his
  §5.1 names scaler, PCA, *and* operators; Table F.9 lists op ∈ {rnd, pca},
  i.e. data-dependent and refit per crisis). Ours are fixed seeded random
  Hermitian, never fit to data at all — cleaner for causality (zero look-ahead
  by construction; see "Honest scope" and the operator note), but a genuine
  protocol difference, not a match.

A similar number out of a different feature set, dimensionality, crisis panel,
and operator treatment is weaker evidence than the numerical closeness
suggests — as easily coincidence as replication. Honest ceiling: **no collapse
observed; values in the same range as his Table 3 median, under a protocol
matching his in structure but not in features, dimensionality, crisis panel, or
operator treatment.**

The often-quoted *d* ≈ 0.26 for reduced purity is a **different protocol**, not
this one: Hammond's §5.2 "frozen holdout" — an expanding-window fit anchored at
2005, one-year evaluation windows, monthly operator refits, scored *only on the
evaluation year* (7 crisis pairs). That scoring change closes **Gap 2** (no
longer comparing the crisis window against all other days, including future
ones), not just Gap 1. This repo has not built that protocol — it is roadmap
item 5 (expanding-window walk-forward). So 0.26 is out of Task 2's reach by
construction, not a number we aimed at and missed. (The paper is internally
inconsistent on the mechanism: §5.2 attributes the drop to "restricting
preprocessing to past data only," but the Table 3 caption says its 0.83
*already* restricts preprocessing to past data. The protocol difference —
expanding-window plus eval-year-only scoring — is what separates 0.26 from
0.83, not the past-fit restriction. Recorded as a reading of the source, not a
criticism.)

**Purity as a "leading" channel — a caveat from Hammond's own null test, and
one we have not run.** Hammond's null-model analysis (§5.1) found reduced
purity's real median *d* = 0.73 against a null median of ~0.53, 95% interval
[0.30, 0.89], **p = 0.18** — its offline lead was never established as
distinguishable from the noise floor, and the 0.26 figure sits *below* the
null's lower bound of 0.30. This bears directly on the table above, where
purity is among the top channels: it can lead a ranking without clearing noise.
And it applies more sharply to us — **we have no null model at all.** Hammond
at least tested against random-window and circular-shift nulls; none of our
seven channels has been tested against any null yet (roadmap). Read every |d|
here with that missing floor in mind.

**Favorable result, reported for the same reason unfavorable ones are.** On
the rate-driven 2022 crisis the geometric channels lead the table — ground
energy E0 at 1.64 (the highest single |d| here), QFI log-det 1.20, spectral
entropy 1.18, all above the HMM baseline (1.07). Hammond reports the Berry
channel leading on rate events; here E0 leads instead. One window, no CIs —
reported for symmetry, not as a ranking claim.

**The SLD channel, held to the same skepticism as every other channel:** it is
**blind on Rate Hikes 2022** (0.01 offline, 0.07 causal — flatly, it does not
see that crisis), flat on China (0.56 → 0.54, and China is uninterpretable
regardless), and rises on COVID (0.53 → 0.66). That COVID rise is **not**
claimed as robustness: the QFI log-det channel — near-blind at 0.10 — moved by
the same +0.12 in the same window, its z-series is 0.97 correlated with the
offline one, and COVID has real preprocessing divergence, so a mid-pack
channel moving ±0.13 there is consistent with preprocessing sensitivity, not
signal. SLD builds ρ_t from ground states that depend on the PCA frame, so it
is frame-sensitive too — less directly than Berry, but not invariant. Net:
causal fitting neither clearly helps nor hurts SLD, and no such claim is
defensible without CIs.

## Roadmap

- **v1-v3 — DONE.** Embedding, 7 channels, SLD mixed-state QFI, offline
  evaluation on real SPY/DIA (see Results), channel correlation matrix
  answering Open Question 1 (SLD is decorrelated, not redundant — see
  above). Identities tested in `tests/test_geometry.py` and
  `tests/test_sld.py`.
- **v4 — causal (past-fit) preprocessing — DONE.** Fits scaler/PCA/HMM only on
  pre-crisis rows, transforms the full timeline, re-evaluates (see "Causal
  (past-fit) preprocessing" above). Closes Gap 1 (leaky preprocessing); Gap 2
  (event-study metric) remains, so numbers stay "separability," not yet
  out-of-sample detection.
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
