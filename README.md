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
| Spectral entropy | geometric v0 | 1.02 |
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
spectral entropy (ρ = 0.121 Pearson, 0.126 Spearman) — its closest
conceptual relative — and even that is well within noise. This answers Open
Question 1 on the axis it was posed: the mixed-state generalization is
empirically **distinct** from the pure-state channels, not a redundant
formalism exercise reusing spectral information under a different name.

**That orthogonality result stands; the effect-size half of the argument does
not.** An earlier version of this section read the SLD channel's COVID
Cohen's *d* as "a real, medium effect, not noise-level." The null-model tests
below falsify that: the SLD channel does not clear its own noise floor on any
window tested. Decorrelated-and-undetected is the accurate description. See
"Null-model tests" below.

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

To run on real data, swap `synthetic_prices` for `load_prices(("SPY", "DIA"))`
in `scripts/run_demo.py`. No network needed — it reads the pinned snapshot.

## Reproducibility

Every real-data number in this README comes from a **pinned price snapshot**
committed at `data/spy_dia_close.csv` (SPY/DIA close, 2005-01-03 →
2026-06-30, `end` exclusive), loaded by `qgmrd.data.load_prices`. The three
analysis commands below reproduce their tables exactly, offline:

```bash
python scripts/channel_correlations.py   # correlation matrix
python scripts/causal_eval.py            # offline vs. causal |d|
python scripts/null_model.py             # per-channel null floors
```

Why a snapshot rather than a pinned end date. `yfinance` returns
dividend/split-**adjusted** closes, so every new distribution retroactively
rescales the entire price history — the past itself changes. Cohen's *d*
compares a crisis window against every other day, so unpinned data quietly
shifts every *d*, correlation, and null floor in this repo between runs. An
`end=` date freezes the last row but not the earlier ones; only a snapshot
freezes the numbers. (Observed drift before pinning was ≤ 0.0002 in p-values —
small, but it meant no table here mapped to a command a reader could run and
match, which the repo promises.) `load_yfinance` still exists for a deliberate
re-pull via `load_prices(refresh=True)`; expect numbers to move and every
script to need a re-run.

Two related pins: `auto_adjust=True` is passed explicitly rather than left to
the `yfinance` default, which has changed across versions; and the package must
be installed **editable** (`uv pip install -e .`) — a stale non-editable copy
in `site-packages` will silently shadow the repo.

### Snapshot provenance

| | |
|---|---|
| file | `data/spy_dia_close.csv` |
| SHA-256 | `c8fe62c2bd7a49333d057f15377d6d700751b750e195b8acee443286a227e99f` |
| rows | 5406 |
| columns | `SPY`, `DIA` (close) |
| first / last | 2005-01-03 / 2026-06-30 |
| source | `yfinance`, `auto_adjust=True` (pinned explicitly) |

```bash
shasum -a 256 data/spy_dia_close.csv   # must match the value above
```

Yahoo's terms restrict redistribution, so **this file may be removed from the
repo later.** If it is, the checksum is what remains useful: it lets you verify
whether your own fetch reproduces the data these numbers were computed from.

Be aware that **a later fetch will not match** — that is the entire reason the
snapshot exists. `auto_adjust=True` returns dividend/split-adjusted closes, so
every distribution SPY and DIA pay after 2026-06-30 retroactively rescales the
whole history. A fetch in 2027 returns a differently-adjusted 2005–2026 series,
a different checksum, and slightly different numbers from the ones printed
here. A mismatch therefore means "the adjustment factors have moved on," not
necessarily "something is wrong" — but it does mean your numbers and this
README's numbers are no longer the same computation.

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
the pure-state channels it generalizes.**

This is a claim about *redundancy*, and it survives. It is **not** a claim
that the channel detects anything: an earlier draft paired it with "d = 0.53
on COVID (real, if mid-pack, separability)" to build an empirical case for
the extension, and the null-model tests below remove that half of the
argument. A channel can be perfectly decorrelated from every other channel
and still be indistinguishable from noise — orthogonality and detection are
independent properties, and here only the first is established.

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
| Reduced purity | 1.15 → 1.14 | 0.95 → 0.90 | 0.97 → 0.94 |
| Spectral entropy | 1.02 → 0.97 | 1.18 → 1.22 | 0.71 → 0.49 |
| Ground energy E0 | 0.96 → 0.95 | 1.64 → 1.61 | 0.48 → 0.64 |
| Berry phase rate | 0.71 → 0.70 | 0.99 → 0.91 | 0.24 → 0.79 |
| SLD QFI (w=20) | 0.53 → 0.66 | 0.01 → 0.06 | 0.56 → 0.55 |
| QFI log-det | 0.10 → 0.21 | 1.20 → 1.09 | 0.82 → 0.17 |
| Gaussian HMM | 1.15 → 1.12 | 1.07 → 1.06 | 0.05 → 0.02 |

**China 2015 is uninterpretable and is not analyzed channel-by-channel.** The
HMM control scores 0.05 offline — it essentially cannot see a crisis in
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
rolling cross-correlation `xcorr20` (Δscale/scale = 0.40 / 0.35 / 0.60, since
correlations reprice hard in a crisis) — while the other ten features shift
≤ 0.09 (median 0.04–0.05, only 1/11 above 0.10). That divergence *does* reach the
daily embedding (row-wise causal-vs-offline cosine mean 0.98 COVID, 0.99 2022,
0.93 China; min 0.83 on COVID). **But it does not reach the z-scored channel
series** on the two interpretable crises: causal-vs-offline series
correlations are 0.94–1.00 on COVID and 2022 (most ≥ 0.98; the frame-
sensitive Berry and QFI-log-det lowest at 0.94–0.96). So the puzzle of "the
daily embedding differs yet |d| barely moves" resolves as **channel
robustness, not metric coarseness**: the trailing causal z-score washes out
per-day embedding jitter into a detector series ~0.98 correlated with the
offline one. Where |d| barely moves, the underlying series barely move too —
the "Cohen's *d* is too coarse to see series-level change" failure mode is
*not* what the data show here. (On China the series correlations do drop as low as
0.79, i.e. the perturbation reaches the series there — but China is
uninterpretable for the separate reason above.)

**Reduced purity shows no collapse under causal fitting** — its point estimate
stays high (1.14 / 0.90 / 0.94, ≤ 0.05 from the offline values). The right
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

**Purity as a "leading" channel — a caveat from Hammond's own null test, now
run on our channels too.** Hammond's null-model analysis (§5.1) found reduced
purity's real median *d* = 0.73 against a null median of ~0.53, 95% interval
[0.30, 0.89], **p = 0.18** — its offline lead was never established as
distinguishable from the noise floor, and the 0.26 figure sits *below* the
null's lower bound of 0.30. This bears directly on the table above, where
purity is among the top channels: it can lead a ranking without clearing noise.
We have now run the same two nulls per channel — see "Null-model tests" below.
**The short version: almost nothing in the table above clears its own floor.**
Read every |d| here with that in mind.

**Favorable result, reported for the same reason unfavorable ones are.** On
the rate-driven 2022 crisis the geometric channels lead the table — ground
energy E0 at 1.64 (the highest single |d| here), QFI log-det 1.20, spectral
entropy 1.18, all above the HMM baseline (1.07). Hammond reports the Berry
channel leading on rate events; here E0 leads instead. One window, no CIs —
reported for symmetry, not as a ranking claim.

**The SLD channel, held to the same skepticism as every other channel:** it is
**blind on Rate Hikes 2022** (0.01 offline, 0.06 causal — flatly, it does not
see that crisis), flat on China (0.56 → 0.55, and China is uninterpretable
regardless), and rises on COVID (0.53 → 0.66). That COVID rise is **not**
claimed as robustness: the QFI log-det channel — near-blind at 0.10 — moved by
a comparable +0.11 in the same window, its z-series is 0.94 correlated with
the offline one, and COVID has real preprocessing divergence, so a mid-pack
channel moving ±0.13 there is consistent with preprocessing sensitivity, not
signal. SLD builds ρ_t from ground states that depend on the PCA frame, so it
is frame-sensitive too — less directly than Berry, but not invariant. Net:
causal fitting neither clearly helps nor hurts SLD, and no such claim is
defensible without CIs.

## Null-model tests (HANDOFF §8.1b)

```bash
python scripts/null_model.py
```

Every |d| above was reported without a noise floor. In an autocorrelated,
heavy-tailed score series, *any* contiguous window separates from the rest by
some amount for free. These tests measure that free lunch per channel and ask
whether the real crisis windows beat it. Two nulls, run on the **causal
(past-fit) z-scored series**:

- **(a) Random matched-length windows** — series fixed, window moves. Draw a
  window of the same non-NaN length as the crisis, placed elsewhere and
  non-overlapping it; |d| vs. the rest; 5000 seeded draws. Tests whether the
  crisis *dates* are special or whether any same-length window separates as
  well.
- **(b) Circular shift** — window fixed, series slides. Roll the series by
  every offset (all M−1, deterministic), keep the crisis mask fixed. Tests
  whether the score's *alignment* to the crisis is real or accidental. A
  circular shift preserves the marginal and autocorrelation exactly (up to one
  seam) while destroying correspondence to the dates.

**The floor is computed per channel, never imported.** Hammond's ~0.53 came
from his pipeline; a floor depends on each channel's own distributional shape.
Ours range from **0.23 to 0.71** across channels and windows — straddling his
0.53 in both directions — so importing a single number would have been wrong
for most channels.

### Headline

**1 of 28 primary-family tests survives Benjamini–Hochberg FDR at q < 0.05:**
ground energy `E0` on Rate Hikes 2022, |d| = 1.61 against a null median of
0.35 — where **not one of the 5000 random matched-length windows reached it**
(the 100th percentile of its own floor; raw p = 0.0002, q = 0.006). At
α = 0.05 across 28 tests the expected number of chance survivors is **1.4**,
so **one survivor out of 28 is, as a count, barely distinguishable from the
expected false-positive rate.** This is not a detection result.

**But the count understates this particular hit, and saying so is part of
reporting symmetrically.** Under a true null, chance survivors scatter near the
q ≈ 0.05 boundary — marginal is what they look like by construction. This one
is not marginal: **no null window out of 5000 reached it.** A hit that deep in
the tail is atypical for a chance survivor, and "1 vs. 1.4 expected" — a
statement about counts — says nothing about where in the tail the one landed.

**The circular-shift null is indeterminate here, not a failure.** Its reported
q = 0.277 must not be read as the shift test rejecting `E0`. About 0.3% of
circular shifts exceeded the real value (raw p ≈ 0.003, which would survive FDR
comfortably); we then floor the p-value at 1/(N_eff+1) ≈ 0.020, because with
N_eff ≈ 50 the shift null cannot resolve below that. **A floored p-value is not
a measurement — it means "at most 0.020, cannot resolve lower."** So the shift
null did not adjudicate `E0` in either direction, and **what prevents
adjudication is our own conservatism, not the data.** We still report the
floored value: quoting the raw 0.003 would be false precision on a null whose
effective sample size is 50. But a bound must be read as a bound.

What does keep the verdict at "not a detection result" is simpler and does not
depend on the shift null at all: **a single window, no CI, and no replication
across crises.** One window cannot establish a channel. **Consequence for
planning: `E0` on the rate-driven crisis is the single result most worth
prioritizing in the multi-crisis panel**, and it is pre-registered here as the
hypothesis to test rather than something to rediscover after the fact.

### Primary family (COVID 2020 + Rate Hikes 2022)

`pct` = percentile of the real |d| within its own null (high = strong);
`p` = fraction of null draws at or above it (low = strong). They are
complements, not the same number.

| crisis | channel | dir | \|d\| | null med | null 95% | pct | p (a) | p (b) |
|---|---|---|---|---|---|---|---|---|
| COVID | reduced purity | + | 1.14 | 0.54 | [0.03, 1.46] | 88 | 0.124 | 0.124 |
| COVID | Gaussian HMM | + | 1.12 | 0.57 | [0.04, 2.12] | 87 | 0.135 | 0.133 |
| COVID | spectral entropy | + | 0.97 | 0.65 | [0.03, 1.87] | 68 | 0.325 | 0.326 |
| COVID | ground energy E0 | − | 0.95 | 0.59 | [0.02, 1.69] | 74 | 0.257 | 0.246 |
| COVID | Berry phase rate | − | 0.70 | 0.63 | [0.03, 2.09] | 55 | 0.452 | 0.434 |
| COVID | **SLD QFI (w=20)** | + | 0.66 | 0.47 | [0.02, 1.48] | 69 | 0.314 | 0.316 |
| COVID | QFI log-det | − | 0.21 | 0.66 | [0.03, 1.78] | 17 | 0.835 | 0.832 |
| 2022 | **ground energy E0** | − | **1.61** | 0.35 | [0.01, 1.38] | 100 | **0.0002** | 0.020 |
| 2022 | spectral entropy | + | 1.22 | 0.38 | [0.02, 1.76] | 81 | 0.186 | 0.172 |
| 2022 | QFI log-det | − | 1.09 | 0.55 | [0.03, 1.53] | 82 | 0.182 | 0.173 |
| 2022 | Gaussian HMM | + | 1.06 | 0.43 | [0.02, 1.81] | 88 | 0.120 | 0.107 |
| 2022 | Berry phase rate | − | 0.91 | 0.50 | [0.03, 1.31] | 81 | 0.192 | 0.195 |
| 2022 | reduced purity | + | 0.90 | 0.28 | [0.01, 1.14] | 93 | 0.074 | 0.065 |
| 2022 | **SLD QFI (w=20)** | + | 0.06 | 0.23 | [0.02, 0.77] | 12 | 0.879 | 0.873 |

### The SLD channel, reported as pre-registered

The commitment to report this channel's result regardless of outcome was
recorded before the analysis was run (HANDOFF §2d). The outcome is
unfavorable and is stated in the committed language:

> The SLD mixed-state QFI channel is **mathematically correct**
> (`tests/test_sld.py::test_pure_state_limit` proves it reduces to `4g_aa` in
> the pure-state limit), **conceptually novel** (source-verified absent from
> Hammond's repo under any name), **decorrelated from the other channels**
> (Task 1, |ρ| < 0.13), and **not demonstrated to detect above chance on any
> window tested** — COVID p ≈ 0.31, Rate Hikes 2022 p ≈ 0.88, China 2015
> p ≈ 0.40, on both nulls.

Two details make this a *stronger* negative than the p-values alone suggest,
and both cut against the channel:

1. **SLD has the lowest floor of any channel** (null median 0.23–0.47 vs. up
   to 0.71 for others). It faced the easiest bar and still did not clear it.
2. **SLD has by far the least autocorrelated series** (τ ≈ 26 vs. ≈ 400 for
   spectral entropy), giving it the *highest* effective sample size
   (N_eff ≈ 204 vs. ≈ 13) and therefore the most statistical power of any
   channel here. It is the channel best positioned to show an effect, and it
   shows none.

On Rate Hikes 2022 it lands at the 12th percentile of its own null — below the
floor's median, consistent with the existing README note that it is flatly
blind to that crisis. Task 1's orthogonality finding is unaffected; what is
withdrawn is the claim that its COVID *d* represented "a real, medium effect,
not noise-level."

### Reduced purity: an independent reproduction of Hammond's null result

Purity's COVID |d| = 1.14 is among the highest in the table, and its
p ≈ 0.12 (both nulls) does not clear the floor. Hammond found the same thing
on his own pipeline: real median *d* = 0.73, **p = 0.18**. Two independent
implementations — different features (11 vs. ~52), different dimensionality
(8 vs. 15), different operator treatment (fixed vs. refit) — reach the same
verdict: **reduced purity leads rankings without clearing noise.** This is a
genuine reproduction result, and it is the clearest positive finding in this
section, even though what it reproduces is a negative.

### The control fails too — and what that does and does not license

The Gaussian HMM baseline also fails to clear its floor on COVID
(|d| = 1.12, p ≈ 0.13). This matters for interpretation: a well-understood
classical detector, on the crisis it should find easiest, does not clear
either. **So this is not geometric channels being uniquely weak — the whole
panel is underpowered on single short windows.**

Stated precisely, because these are different claims and only the weaker one
is supported: **this analysis cannot distinguish "no signal" from "signal too
weak to detect with three crises."** Nothing here establishes that the
geometric channels do not detect regimes. What it establishes is that
**single-window Cohen's *d* on three crises cannot demonstrate that they do.**
Every effect size in this repo should be read as un-cleared until a
higher-powered protocol says otherwise.

### Methodology caveats

- **The floor is conservative, not clean.** Random null windows are drawn from
  the whole 2005–present series, so some land on 2008, 2011, or 2018 — real
  crises, not null periods. That inflates the floor. We keep them: it pushes
  in the safe direction (harder to clear), but it means a failing channel was
  measured against "any other turbulent stretch," not against calm.
- **Circular-shift p-values are floored at 1/(N_eff+1).** Adjacent offsets on
  an autocorrelated series are near-duplicate draws, so M shifts are worth far
  fewer than M independent samples. We estimate N_eff = M/τ (Geyer
  initial-positive-sequence). For spectral entropy and QFI log-det, τ ≈ 400–442
  and **N_eff ≈ 12–13**, so their shift p-values cannot resolve below ≈ 0.07 no
  matter how many offsets are enumerated. Quoting an unfloored p there would be
  false precision.
- **|d| is folded.** Absolute Cohen's *d* discards direction, so a channel
  moving the *wrong* way during a crisis still registers as detecting. Folding
  is kept for consistency with the rest of the repo; the `dir` column shows
  each real effect's sign so the reader can check.
- **China 2015 is exploratory and excluded from the FDR family.** It carries
  the pre-existing uninterpretability caveat (blind HMM control, d = 0.05 offline, 0.02 causal).
  Its numbers are printed by the script for completeness. No channel clears
  there either (best: reduced purity, p ≈ 0.17).
- **Multiple comparisons were pre-specified**, not chosen after seeing
  results: primary family = COVID + 2022 (28 tests), China held out, raw p and
  BH q-values both reported.
- **Data is pinned.** All figures come from the committed snapshot
  `data/spy_dia_close.csv` (SPY/DIA close, 2005-01-03 → 2026-06-30). See
  "Reproducibility" below — re-running these commands reproduces these numbers
  exactly.

**Implementation cross-check:** the two nulls are structurally different
constructions, yet their p-values agree to within ~0.01 on every channel
(e.g. COVID purity 0.127 vs. 0.125). Two independent routes to the same floor
is evidence the floor estimate is not an artifact of either one.

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
- **v4.5 — null-model tests — DONE.** Per-channel noise floors via random
  matched-length windows and circular shift (see above). Result: 1 of 28
  primary tests survives FDR, against ~1.4 expected by chance. This
  reprioritizes everything below.
- **v5 — multi-crisis panel — NOW THE CRITICAL PATH, not a nice-to-have.**
  Repeat across the paper's crisis windows (2018 Q4, 2011, 2008, …) with
  block-bootstrap CIs. The null results make this the gating task rather than
  an incremental one: **no per-channel detection claim in this repo is
  supportable until power rises**, and more crisis windows is the only lever
  that raises it. Note that Hammond reports a **median across 17 crises**, not
  a single-window result — the comparison this repo has been making to his
  numbers is a comparison of a point estimate to a median, and closing that
  gap is what the panel is for.

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
