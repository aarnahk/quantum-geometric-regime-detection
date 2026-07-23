# Quantum-Geometric Market Regime Detection

A from-scratch reproduction and extension of the QCML geometric-observable
pipeline for detecting market regime shifts (Hammond 2026,
[arXiv:2605.17117](https://arxiv.org/abs/2605.17117)), reframed through
Quantum Fisher Information and the Cramer-Rao bound as estimation on a
statistical manifold.

**Status: v5.** Seven channels are implemented and causally z-scored: spectral
entropy, reduced-density-matrix purity, ground-state energy, Berry-phase rate,
QFI log-determinant, and a novel SLD mixed-state QFI channel — benchmarked
against a Gaussian HMM, evaluated with causal (past-fit) preprocessing across a
15-crisis panel with per-channel noise floors and block-bootstrap CIs. The
quantum-metric identity 4g = F_Q is verified numerically. Nothing here claims
prediction: following the source paper, these are *contemporaneous detection*
observables, not forecasters.

## Headline finding

Across a **15-crisis panel** with causal (past-fit) preprocessing, **no channel's
median Cohen's |d| clears its own noise floor** — 0 of 14 tests survive
Benjamini–Hochberg FDR, against 0.7 expected by chance. That includes the
Gaussian HMM control, and it includes the two channels that were
**pre-registered** as hypotheses before the panel was run. See
"[Multi-crisis panel](#multi-crisis-panel-the-headline-result)" below, which is
the load-bearing section of this README; everything above it is single-crisis
work that the panel supersedes.

## Crisis window convention (changed — read before comparing to earlier numbers)

**Every number in this README uses Hammond's Table G.10 windows extended by
±10 trading days** (his §4.1 convention), defined once in `qgmrd/crises.py` and
imported by every script.

This is a **change**, and earlier versions of these tables used a different,
ad-hoc definition. The old windows were inconsistent with G.10 *and* with each
other: COVID was hand-set to 2020-02-19 → 2020-04-30 (inside G.10's
2020-02..2020-04), while 2022 and China matched G.10 **unextended**. Two
conventions in one repo is worse than either alone, so one was adopted
uniformly and every table was recomputed under it. The decision was made and
written down **before** any result under the new convention was computed.

The change is not cosmetic — it moves numbers substantially, and always
downward for COVID:

| COVID 2020, causal \|d\| | old ad-hoc window | G.10 ±10 |
|---|---|---|
| Reduced purity | 1.14 | 0.56 |
| Gaussian HMM | 1.12 | 0.75 |
| Spectral entropy | 0.97 | 0.79 |

The G.10 ±10 COVID window is 2020-01-17 → 2020-05-14 (82 trading days), which
brackets the crash with a calm January and most of the April–May recovery. That
dilutes a sharp event against its own quiet shoulders. This is a real cost of
the convention and is not worked around: the alternative — hand-trimming each
window to where the effect looks strongest — is precisely the tuning the rest
of this repo is built to avoid.

One benefit: because the past-fit cutoff is computed from the **extended**
window start, preprocessing now stops ~20 trading days before the nominal
crisis month rather than ~10, which is comfortably clear of the 20-day rolling
feature windows.

## Results (SPY/DIA, offline event study)

Cohen's |d|, COVID-2020 window vs. rest. **Offline**: preprocessing is fit on
all data, so these measure crisis-window separability (an event study), not
causal out-of-sample detection. The causal (past-fit) preprocessing version is
below (Task 2); closing the remaining event-study metric gap (Gap 2) is the
later milestone.

| method | type | \|d\| |
|---|---|---|
| Gaussian HMM (high-var prob) | baseline | 0.78 |
| Spectral entropy | geometric v0 | 0.76 |
| Reduced purity | geometric v0 | 0.63 |
| Berry phase rate | geometric v2 | 0.58 |
| Ground energy E0 | geometric v2 | 0.43 |
| QFI log-det | geometric v2 | 0.25 |
| SLD mixed-state QFI (w=20) | geometric v3 | 0.08 |

**These are single-window point estimates and none of them clears its own noise
floor** (see "Null-model tests"). Read the ordering as descriptive, not as a
ranking. Proposition 2 identity verified: corr(g_FD, g_PT) = 1.000000000, max
rel. error ~1e-5, confirming 4g = F_Q via two independent metric computations.

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
python scripts/multi_crisis_panel.py     # 15-crisis panel (the headline)
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

## Honest scope (what this is and isn't)

- Operators are **fixed random Hermitian** (paper-endorsed for detection), not
  gradient-learned. Reconstruction-loss operator learning is out of scope.
- Preprocessing (scaler, PCA, HMM) is fit **causally / past-only** in every
  result that carries a claim (Task 2 onward); the "offline" columns are kept
  as the leaky comparison, labelled as such. This closes **Gap 1** but not
  **Gap 2** — Cohen's *d* still scores the crisis window against future days,
  so this is a **causal event study, not a walk-forward**. That is the
  distinction the paper draws between its offline Table 3 and its load-bearing
  Table 5, and the word "walk-forward" is not used for this repo's protocol.
- Absolute Cohen's d on synthetic data is not comparable to the paper's real-
  crisis numbers; it exists to prove the pipeline produces separable signal.
- **No channel in this repo has been shown to detect above chance.** See the
  multi-crisis panel. Effect sizes are reported throughout; none of them clears
  its own noise floor after multiple-comparison correction.

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
on COVID (real, if mid-pack, separability)" to build an empirical case for the
extension. The null-model tests below removed that half of the argument, and
the window-convention change removed the number itself — SLD's COVID |d| under
G.10 ±10 is 0.08 offline, 0.25 causal. A channel can be perfectly decorrelated
from every other channel
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

**No confidence intervals on *this* table.** Every number below is a point
estimate on a single realized path of three crises, so **no delta here is
certified distinguishable from noise**, and "barely moved" means the point
estimate barely moved, not a statistical claim. Block-bootstrap CIs were the
pending task this deferred to; they have since been built, but on the
**15-crisis panel median** rather than on per-crisis deltas — see
"[Multi-crisis panel](#multi-crisis-panel-the-headline-result)". A CI on a
single-crisis delta was never going to be informative at these widths, which is
why the panel was built instead.

Offline → causal Cohen's |d| (SPY/DIA), **G.10 ±10 windows**:

| channel | COVID 2020 | Rate Hikes 2022 | China 2015 |
|---|---|---|---|
| Reduced purity | 0.63 → 0.56 | 0.90 → 0.85 | 0.93 → 0.90 |
| Spectral entropy | 0.76 → 0.79 | 1.18 → 1.23 | 0.70 → 0.48 |
| Ground energy E0 | 0.43 → 0.41 | 1.60 → 1.57 | 0.49 → 0.66 |
| Berry phase rate | 0.58 → 0.57 | 0.96 → 0.88 | 0.22 → 0.77 |
| SLD QFI (w=20) | 0.08 → 0.25 | 0.03 → 0.09 | 0.51 → 0.51 |
| QFI log-det | 0.25 → 0.32 | 1.15 → 1.04 | 0.74 → 0.23 |
| Gaussian HMM | 0.78 → 0.75 | 1.06 → 1.06 | 0.08 → 0.09 |

**China 2015 is uninterpretable and is not analyzed channel-by-channel.** The
HMM control scores 0.08 offline / 0.09 causal — it essentially cannot see a
crisis in SPY/DIA over Jun–Oct 2015 (mostly quiet, with one violent week around
Aug 20–26 that Cohen's *d* dilutes against the calm remainder). Independently,
the causal embedding agrees least with the offline one there (row-wise cosine
mean 0.93, min 0.43 — some days near-orthogonal). With a blind control and the
largest embedding divergence, no individual channel's swing in that column is
trustworthy; those numbers are reported for completeness only. The panel below
shows this is **not** a China-specific quirk: **5 of 15 crises have a blind HMM
control**, so a third of Hammond's post-2005 window list is simply not visible
in SPY/DIA.

**Does the preprocessing change reach the channels?** (`causal_eval.py` prints
all of this.) The preprocessing *parameters* diverge modestly and in a
concentrated way: the PCA axes stay closely aligned (min |cos| 0.98 / 0.99 /
0.94 across the three crises), and the scaler-scale shift is dominated by a
single feature — rolling cross-correlation `xcorr20` (Δscale/scale =
0.40 / 0.35 / 0.60, since correlations reprice hard in a crisis) — while the
other ten features shift ≤ 0.09 (median 0.04–0.05, only 1/11 above 0.10). That
divergence *does* reach the daily embedding (row-wise causal-vs-offline cosine
mean 0.98 COVID, 0.99 2022, 0.93 China; min 0.83 on COVID). **But it does not
reach the z-scored channel series** on the two interpretable crises:
causal-vs-offline series correlations are 0.94–1.00 on COVID and 2022 (most
≥ 0.98; the frame-sensitive Berry and QFI-log-det lowest at 0.94–0.96). So the
puzzle of "the daily embedding differs yet |d| barely moves" resolves as
**channel robustness, not metric coarseness**: the trailing causal z-score
washes out per-day embedding jitter into a detector series ~0.98 correlated
with the offline one. Where |d| barely moves, the underlying series barely move
too — the "Cohen's *d* is too coarse to see series-level change" failure mode is
*not* what the data show here. (On China the series correlations do drop as low
as 0.79, i.e. the perturbation reaches the series there — but China is
uninterpretable for the separate reason above.)

**Reduced purity shows no collapse under causal fitting** — every point
estimate moves ≤ 0.07 from its offline value (0.56 / 0.85 / 0.90). The right
comparison is Hammond's Table 3 reduced-purity median of *d* ≈ 0.83, which is
**already a per-crisis past-fit number** (his §5.1: scaler, PCA, *and*
operators fit only on pre-crisis data) — the same protocol structure
`causal_eval.py` uses. Two of our three causal values sit near it and one
(COVID) well below; the **15-crisis panel median is 0.71**, which is the
type-correct comparison to his 17-crisis median and lands modestly under it.
"Near his number" is not self-evidently good regardless, because the pipelines
differ materially:

- **Features:** 11 raw features here vs. his ~52 enriched.
- **Dimensionality:** 8 PCA components here vs. his 15.
- **Crisis panel:** fifteen crises here (his post-2005 G.10 list), against his
  17-crisis median which spans pre-2005 events our snapshot cannot reach.
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
operator treatment.** And note that "same range as his median" is a statement
about magnitude only — on our panel that magnitude does **not** clear purity's
own noise floor after FDR correction (see below).

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
energy E0 at 1.60 (the highest single |d| here), spectral entropy 1.18, QFI
log-det 1.15, all above the HMM baseline (1.06). Hammond reports the Berry
channel leading on rate events; here E0 leads instead. One window, no CIs —
reported for symmetry, not as a ranking claim. **The panel below retracts any
temptation to read more into it:** E0's 15-crisis median is 0.41, *below* its
own null median, so 2022 was not the tip of a pattern.

**The SLD channel, held to the same skepticism as every other channel:** it is
**effectively blind on Rate Hikes 2022** (0.03 offline, 0.09 causal) and on
COVID under the G.10 ±10 window (0.08 → 0.25), and flat on China (0.51 → 0.51,
which is uninterpretable regardless). The COVID rise is **not** claimed as
robustness: the QFI log-det channel moved by a comparable +0.07 in the same
window, its z-series is 0.94 correlated with the offline one, and COVID has
real preprocessing divergence, so a low-|d| channel moving ±0.17 there is
consistent with preprocessing sensitivity, not signal. SLD builds ρ_t from
ground states that depend on the PCA frame, so it is frame-sensitive too — less
directly than Berry, but not invariant. Net: causal fitting neither clearly
helps nor hurts SLD.

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
Ours range from **0.23 to 0.69** across channels and windows — straddling his
0.53 in both directions — so importing a single number would have been wrong
for most channels.

### Headline

**1 of 28 primary-family tests survives Benjamini–Hochberg FDR at q < 0.05:**
ground energy `E0` on Rate Hikes 2022, |d| = 1.57 against a null median of
0.31 — where **not one of the 5000 random matched-length windows reached it**
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
q = 0.277 must not be read as the shift test rejecting `E0`. Under 2% of
circular shifts exceeded the real value; we then floor the p-value at
1/(N_eff+1) ≈ 0.020, because with N_eff ≈ 50 the shift null cannot resolve
below that. **A floored p-value is not a measurement — it means "at most 0.020,
cannot resolve lower."** So the shift null did not adjudicate `E0` in either
direction, and **what prevents adjudication is our own conservatism, not the
data.**

> **Resolved by the panel — read this before quoting `E0` from this section.**
> The verdict above rested on "a single window, no CI, and no replication
> across crises," with the multi-crisis panel pre-registered as the test. That
> test has now been run, and **`E0` did not replicate**: its 15-crisis median
> |d| is 0.41 against a null median of 0.49 — *below* its own floor, at the
> 31st percentile, q = 0.95. The 2022 hit is best read as the chance survivor
> the count always suggested it was. This subsection is kept as the record of
> what was claimed before the panel, and of the fact that it was framed as a
> hypothesis rather than a finding.

### Primary family (COVID 2020 + Rate Hikes 2022), G.10 ±10 windows

`pct` = percentile of the real |d| within its own null (high = strong);
`p` = fraction of null draws at or above it (low = strong). They are
complements, not the same number.

| crisis | channel | dir | \|d\| | null med | null 95% | pct | p (a) | p (b) |
|---|---|---|---|---|---|---|---|---|
| COVID | spectral entropy | + | 0.79 | 0.59 | [0.03, 1.85] | 62 | 0.382 | 0.372 |
| COVID | Gaussian HMM | + | 0.75 | 0.55 | [0.02, 1.89] | 82 | 0.177 | 0.177 |
| COVID | Berry phase rate | − | 0.57 | 0.60 | [0.03, 1.98] | 48 | 0.519 | 0.496 |
| COVID | reduced purity | + | 0.56 | 0.46 | [0.02, 1.36] | 58 | 0.416 | 0.411 |
| COVID | ground energy E0 | − | 0.41 | 0.49 | [0.02, 1.56] | 43 | 0.568 | 0.563 |
| COVID | QFI log-det | − | 0.32 | 0.66 | [0.03, 1.71] | 24 | 0.757 | 0.736 |
| COVID | **SLD QFI (w=20)** | + | 0.25 | 0.38 | [0.01, 1.22] | 35 | 0.648 | 0.656 |
| 2022 | **ground energy E0** | − | **1.57** | 0.31 | [0.02, 1.37] | 100 | **0.0002** | 0.020 |
| 2022 | spectral entropy | + | 1.23 | 0.39 | [0.02, 1.76] | 82 | 0.181 | 0.164 |
| 2022 | Gaussian HMM | + | 1.06 | 0.43 | [0.00, 1.82] | 88 | 0.121 | 0.107 |
| 2022 | QFI log-det | − | 1.04 | 0.55 | [0.03, 1.49] | 80 | 0.206 | 0.202 |
| 2022 | Berry phase rate | − | 0.88 | 0.51 | [0.02, 1.34] | 78 | 0.217 | 0.218 |
| 2022 | reduced purity | + | 0.85 | 0.27 | [0.01, 1.14] | 92 | 0.085 | 0.074 |
| 2022 | **SLD QFI (w=20)** | + | 0.09 | 0.23 | [0.01, 0.75] | 24 | 0.765 | 0.754 |

**Note how wide those null 95% intervals are** — up to [0.03, 1.98]. That is
the free lunch a single short window gets from an autocorrelated, fat-tailed
series, and it is why a single-crisis test has almost no resolution. Narrowing
it is the entire point of the panel below.

### The SLD channel, reported as pre-registered

The commitment to report this channel's result regardless of outcome was
recorded before the analysis was run (HANDOFF §2d). The outcome is
unfavorable and is stated in the committed language:

> The SLD mixed-state QFI channel is **mathematically correct**
> (`tests/test_sld.py::test_pure_state_limit` proves it reduces to `4g_aa` in
> the pure-state limit), **conceptually novel** (source-verified absent from
> Hammond's repo under any name), **decorrelated from the other channels**
> (Task 1, |ρ| < 0.13), and **not demonstrated to detect above chance on any
> window tested** — COVID p ≈ 0.65, Rate Hikes 2022 p ≈ 0.76, China 2015
> p ≈ 0.40, on both nulls, **nor on the 15-crisis panel median** (p ≈ 0.28–0.30,
> q = 0.95).

Two details make this a *stronger* negative than the p-values alone suggest,
and both cut against the channel:

1. **SLD has the lowest floor of any channel** (null median 0.23–0.42 vs. up
   to 0.69 for others). It faced the easiest bar and still did not clear it.
2. **SLD has by far the least autocorrelated series** (τ ≈ 26 vs. ≈ 400 for
   spectral entropy), giving it the *highest* effective sample size
   (N_eff ≈ 205 vs. ≈ 13) and therefore the most statistical power of any
   channel here. It is the channel best positioned to show an effect, and it
   shows none.

Task 1's orthogonality finding is unaffected; what is withdrawn is the claim
that its COVID *d* represented "a real, medium effect, not noise-level."

**One thing that must be said in the other direction, for symmetry.** On the
panel, SLD's median |d| (0.46) is the only channel other than reduced purity to
sit *above* its own null median (0.39), and it sits above the HMM control's
median (0.34). That is **not** a detection claim — p ≈ 0.28 does not clear
anything, and ranking channels by median is exactly the mining this repo
refuses to do. It is recorded because the pre-registered commitment was to
report the SLD result as it came out, and "did not clear, but was not among the
channels that landed below their own floor" is the accurate description.

### Reduced purity: an independent reproduction of Hammond's null result

Purity's 2022 |d| = 0.85 sits at the 92nd percentile of its own floor, and its
p ≈ 0.07–0.09 still does not clear after correction. Hammond found the same
thing on his own pipeline: real median *d* = 0.73, **p = 0.18**. Two
independent implementations — different features (11 vs. ~52), different
dimensionality (8 vs. 15), different operator treatment (fixed vs. refit) —
reach the same verdict: **reduced purity leads rankings without clearing
noise.** This is a genuine reproduction result, and it is the clearest positive
finding in this section, even though what it reproduces is a negative. The
panel sharpens it rather than overturning it: purity's 15-crisis median is the
closest any channel comes to clearing (raw p ≈ 0.034–0.038, better than
Hammond's 0.18 on the same question) and still fails FDR at q = 0.26.

### The control fails too — and what that does and does not license

The Gaussian HMM baseline also fails to clear its floor on COVID
(|d| = 0.75, p ≈ 0.18) or on 2022 (|d| = 1.06, p ≈ 0.11), and its panel median
(0.34) is the **lowest of all seven channels** and below its own null. This
matters for interpretation: a well-understood classical detector, on crises it
should find easiest, does not clear either. **So this is not geometric channels
being uniquely weak — the whole panel is underpowered, or the metric is too
blunt, or both.**

Stated precisely, because these are different claims and only the weaker one
is supported: **this analysis cannot distinguish "no signal" from "signal too
weak to detect with three crises."** Nothing here establishes that the
geometric channels do not detect regimes. What it establishes is that
**single-window Cohen's *d* on three crises cannot demonstrate that they do.**
Every effect size in this repo should be read as un-cleared until a
higher-powered protocol says otherwise. The panel below *is* that
higher-powered protocol, and it does not rescue any channel — which narrows,
but does not eliminate, the "too weak to detect" branch.

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
- **The "rest" group is contaminated with other crises, and every absolute
  \|d\| in this repo is biased downward as a result.** The 15 G.10 ±10 windows
  cover **1412 of 5386 trading days — 26.2% of the series**, so when a channel
  is scored on one crisis, roughly a quarter of the comparison group is *other*
  crisis days. The crisis and non-crisis means are pulled together and |d|
  shrinks. This matters mainly for **comparing our absolute |d| to Hammond's**,
  where his rest-group construction may differ. It largely does **not** affect
  the null tests: null windows are scored against the same contaminated rest
  group, so the floor absorbs the same bias and the comparison stays like-for-
  like. Not corrected — correcting it would mean choosing which days count as
  "clean," which is a modelling decision with its own forking paths.
- **|d| is folded.** Absolute Cohen's *d* discards direction, so a channel
  moving the *wrong* way during a crisis still registers as detecting. Folding
  is kept for consistency with the rest of the repo; the `dir` column shows
  each real effect's sign so the reader can check.
- **China 2015 is exploratory and excluded from the FDR family.** It carries
  the pre-existing uninterpretability caveat (blind HMM control, d = 0.08
  offline, 0.09 causal). Its numbers are printed by the script for
  completeness. No channel clears there either (best: reduced purity,
  p ≈ 0.14–0.15).
- **Multiple comparisons were pre-specified**, not chosen after seeing
  results: primary family = COVID + 2022 (28 tests), China held out, raw p and
  BH q-values both reported.
- **Data is pinned.** All figures come from the committed snapshot
  `data/spy_dia_close.csv` (SPY/DIA close, 2005-01-03 → 2026-06-30). See
  "Reproducibility" below — re-running these commands reproduces these numbers
  exactly.

**Implementation cross-check:** the two nulls are structurally different
constructions, yet their p-values agree to within ~0.02 on every channel
(e.g. COVID purity 0.416 vs. 0.411). Two independent routes to the same floor
is evidence the floor estimate is not an artifact of either one.

## Multi-crisis panel (the headline result)

```bash
python scripts/multi_crisis_panel.py
```

Everything above this section is single-window work. The null-model tests found
that essentially nothing clears its floor on one crisis — **including the HMM
control** — which cannot distinguish "no signal" from "not enough power." A
panel is the only lever that raises power, so this was flagged as the critical
path, and this section is what came back.

### Why the statistic is the MEDIAN across crises

The headline is each channel's **median |d| across the 15 crises**, not its
per-crisis values. Three reasons, all fixed before the run:

1. **Power.** A single crisis's |d| sits inside an enormous floor — the null
   95% intervals in the table above run to [0.03, 1.98], because an
   autocorrelated, fat-tailed series lets *any* contiguous window separate by
   luck. The median of *k* draws concentrates around the true median at rate
   1/√k, so the null's spread shrinks while the real value does not move.
   Measured: the null 95% interval narrows from ~1.2–2.0 wide per crisis to
   **0.29–0.67 wide** on the panel median, a **~2.5–3× tightening**. Luck does
   not repeat in the same direction eight times out of fifteen.
2. **Multiple comparisons collapse.** Per-crisis testing would be
   15 × 7 × 2 = **210 tests**, expecting ~10.5 chance survivors at α = 0.05 —
   an uncountable result. Testing the median is 7 channels × 2 nulls =
   **14 tests**, expecting **0.7**. A survivor becomes countable as evidence
   instead of drowning in expected noise.
3. **It matches Hammond.** His Table 3 is a 17-crisis median. Every comparison
   this repo has made to his figures until now has been a point estimate
   against a median — a type error the panel fixes.

Median, not mean, because |d| has a heavy right tail: a mean would let one
lucky crisis carry the panel, and would be dragged by the control-blind crises
without the option of leaving them in.

### Result

15 crises (Hammond Table G.10 post-2005, ±10 trading days), causal past-fit
preprocessing per crisis, all 7 channels. **No crises skipped** — the smallest
pre-cutoff sample is 609 rows against a 200-row minimum.

| channel | **median \|d\|** | null median | null 95% | pct | p (a) | p (b) | BH q | 95% CI |
|---|---|---|---|---|---|---|---|---|
| Reduced purity | **0.71** | 0.44 | [0.22, 0.74] | 96 | 0.038 | 0.034 | 0.26 | [0.47, 0.94] |
| Berry phase rate | 0.51 | 0.55 | [0.27, 0.84] | 40 | 0.601 | 0.549 | 0.95 | [0.29, 0.73] |
| SLD QFI (w=20) | 0.46 | 0.39 | [0.20, 0.62] | 70 | 0.299 | 0.278 | 0.95 | [0.26, 0.62] |
| Ground energy E0 | 0.41 | 0.49 | [0.23, 0.81] | 31 | 0.692 | 0.674 | 0.95 | [0.29, 0.80] |
| Spectral entropy | 0.40 | 0.57 | [0.27, 0.94] | 15 | 0.849 | 0.814 | 0.95 | [0.27, 0.73] |
| QFI log-det | 0.36 | 0.59 | [0.30, 0.94] | 6 | 0.936 | 0.950 | 0.95 | [0.20, 0.56] |
| Gaussian HMM | 0.34 | 0.40 | [0.27, 0.56] | 20 | 0.799 | 0.765 | 0.95 | [0.19, 0.71] |

**0 of 14 tests survive BH-FDR at q < 0.05**, against 0.7 expected by chance.
Four of seven channels — including the HMM control — have a panel median
**below their own null median**.

**The design worked; the result is negative.** The floor tightened by the
predicted factor and in the predicted direction. Nothing cleared it. That is a
substantially stronger negative than the single-crisis version, because the
"underpowered" escape hatch is now much narrower — though not closed, and the
framing discipline below still applies.

### Pre-registered hypotheses, reported as committed

Both were recorded in `scripts/multi_crisis_panel.py`'s docstring before the
panel was run. Both are **unfavorable**, and both are reported in the committed
language.

> **H1 — ground energy `E0` clears its null on the panel median. FAILED, and
> decisively.** Prior: `E0` was the only single-crisis survivor (2022,
> q = 0.006, at the 100th percentile of its own floor). On the panel its median
> |d| is **0.41 against a null median of 0.49** — *below* its own floor, at the
> 31st percentile, p ≈ 0.67–0.69, q = 0.95. **The 2022 result did not
> replicate.** With one survivor observed against 1.4 expected by chance, the
> honest reading was always that it might be a chance survivor; the panel is
> the test that says so.

> **H2 — the SLD mixed-state QFI channel clears its null on the panel median.
> FAILED.** Prior: it cleared nothing on single crises despite the easiest bar
> (lowest floor) and the most power (least autocorrelated series). On the panel
> its median |d| is 0.46 against a null median of 0.39 — above its floor but
> only at the 70th percentile, p ≈ 0.28–0.30, q = 0.95. **Not demonstrated to
> detect above chance**, on any window or on the panel.

### Per-crisis table — descriptive only

`multi_crisis_panel.py` prints the full 15 × 7 grid. It is **not** mined for
which channel wins which crisis type: Hammond tested for exactly that
specialization and **found none (p = 0.31)**, so any such pattern across 15
crises is almost certainly noise. Two things in it are worth stating because
they are properties of the *panel*, not of any channel:

- **5 of 15 crises have a blind HMM control** (|d| < 0.2): 2007 Quant Meltdown
  (0.17), 2015 China (0.09), 2018 Volmageddon (0.06), 2019 Repo Crisis (0.09),
  2024 Carry Unwind (0.16). A third of Hammond's post-2005 window list is
  effectively invisible in SPY/DIA — several of those events were not primarily
  US large-cap equity events. **They are flagged and kept, not dropped.**
  Dropping crises after seeing the control is a forking path; the median is
  robust to a minority of bad crises by construction, which is part of why it
  is the headline.
- The per-crisis spread is enormous — the HMM control alone ranges from 2.84
  (2008 GFC) to 0.06 (Volmageddon). This is the variance the median exists to
  absorb, and it is why per-crisis numbers should not be quoted individually.

### Two nulls, and why they are not symmetric

- **(a) Random matched-length windows**, drawn per crisis **independently**,
  5000 panel draws. Independence across crises destroys the cross-crisis
  dependence that the real 15 values have (they come from one market), which
  makes this null median *tighter* than it should be — **an easier bar. Null
  (a) is anti-conservative**, and is reported as such rather than quietly used.
- **(b) Circular shift**, with **one common shift applied to all 15 series at
  once** and every crisis mask held fixed. This preserves the cross-crisis
  dependence exactly — it slides the whole world under a fixed crisis calendar
  — so it is the **conservative** one. Its p is floored at 1/(N_eff+1) as
  elsewhere, using the most autocorrelated of the 15 series.

The two **bracket** the truth, and a channel clearing only (a) would be a
weaker result than one clearing both. In the event, the two agree closely on
every channel (largest gap 0.05) and nothing clears either.

### Block-bootstrap CIs — a different question from the null

The null asks whether the median beats chance (**location**); the CI asks how
precisely the median is measured (**width**). A channel can clear its null with
a CI too wide to be useful, or have a tight CI around a median sitting squarely
in noise. HANDOFF §8 item 1 specifies these, and the Task 2 write-up defers to
them in several places ("no delta in this table is certified distinguishable
from noise") — this pays that off.

**Construction, and why it is not the obvious one.** One circular block
resample of the **whole timeline** per replicate, with each crisis's labels
riding along on the blocks; all 15 |d| are recomputed on that single resampled
series and *then* the median is taken. The tempting alternative — resample each
crisis independently and take the median — is **wrong here**, because the 15
real |d| are not independent: each is scored against a "rest" group containing
almost the entire series, including the other 14 crisis windows, so they share
nearly all their data. Independent resampling would understate the variance of
their median and produce a CI that is too narrow — the same failure mode as
using an iid bootstrap on an autocorrelated series.

Block length comes from the **Politis–White (2004) automatic rule**, not by
hand: block length is the one free knob, and a knob tuned by eye on a reported
statistic is exactly the fitting this repo avoids. It lands at 144–180 days per
channel.

Three caveats, all printed by the script:

- **Conditional on this crisis set.** The CI propagates within-series sampling
  error only. It does **not** propagate "which 15 crises" uncertainty — the set
  is fixed by Table G.10.
- **Politis–White saturates on every channel.** The autocorrelation never drops
  below the rule's threshold inside the rule's own search range, so each block
  length is a **lower bound** on that channel's persistence, and each CI is if
  anything **too narrow**.
- **Short crises are drawn all-or-nothing.** With blocks of ~150–180 days
  against windows as short as 62, a crisis can miss a replicate entirely; it is
  then dropped from that replicate's median. On average **11.5 of 15 crises**
  contribute per replicate (5th percentile: 9). This widens the interval — the
  safe direction — but the CI is a median over ~11–12 crises per replicate, not
  always 15.

### What this does and does not license

**It does not establish that the geometric channels fail to detect regimes.**
The honesty architecture cuts both ways: the strong negative is as much an
overclaim as the optimistic direction would be. What the panel establishes is
narrower and firmer than the single-crisis version:

> At 15-crisis power, with causal past-fit preprocessing and a per-channel
> noise floor, **no channel in this repo — geometric or classical — separates
> crisis windows from arbitrary same-length windows by a margin distinguishable
> from chance.**

The most likely explanations, none of which this analysis can adjudicate
between, are that the effects are genuinely small; that Cohen's *d* against a
rest group 26% composed of other crises is too blunt an instrument; that
SPY/DIA is too narrow an instrument set (a third of the panel is invisible to
the control); or that the event-study metric itself (Gap 2) is the wrong target
and a deployment metric like false-alarms-per-year would behave differently.
Those are the next tests, not conclusions.

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
- **v5 — multi-crisis panel — DONE, and negative.** 15 crises (Hammond Table
  G.10 post-2005, ±10 trading days), causal past-fit preprocessing, median |d|
  as the headline, both nulls run on the median, Politis–White block-bootstrap
  CIs. Result: **0 of 14 tests survive FDR**; both pre-registered hypotheses
  (E0, SLD) fail; the HMM control has the lowest panel median of all seven.
  See "[Multi-crisis panel](#multi-crisis-panel-the-headline-result)". This
  also closed the point-estimate-vs-median mismatch in every comparison to
  Hammond's figures.
- **v6 — what the panel makes next.** The panel removed power as the
  explanation of last resort, so the remaining candidates are *what is being
  measured* rather than *how much data*: (i) **Task 3.5**, the classical
  Bures/MMD baseline, which is now the decisive test for whether the SLD
  channel is more than a relabeled classical statistic; (ii) **Task 4**,
  multi-asset — a third of the panel is invisible to the control in SPY/DIA
  alone; (iii) **roadmap item 4/5**, false-alarms-per-year under an
  expanding-window protocol, which closes Gap 2 and changes the metric rather
  than the sample.

## Layout

```
qgmrd/
  operators.py    random Hermitian operators {A_k}
  embedding.py    H(x) and ground state via eigh
  observables.py  spectral entropy, reduced purity
  geometry.py     metric (2 paths), QFI, QCRB, Berry plaquette
  channels.py     Berry rate / QFI log-det / E0 time series
  sld.py          SLD, mixed-state QFI (the extension)
  zscore.py       causal expanding-window z-score (Algorithm 1)
  features.py     returns / vol / momentum / cross-corr
  crises.py       THE crisis window registry (G.10 +/-10 trading days)
  pipeline.py     embed_series + Cohen's d
  baseline.py     Gaussian HMM
  data.py         synthetic generator + pinned snapshot loader
scripts/
  run_demo.py             synthetic smoke run
  channel_correlations.py Task 1 — redundancy matrix
  causal_eval.py          Task 2 — offline vs. causal past-fit |d|
  null_model.py           per-channel noise floors, single crises
  multi_crisis_panel.py   15-crisis panel — the headline result
tests/
  test_smoke.py  test_geometry.py  test_sld.py
```

`qgmrd/crises.py` is the single source of truth for crisis windows. Every
script imports it, so the window convention cannot drift between them — which
is the bug the convention change fixed.
