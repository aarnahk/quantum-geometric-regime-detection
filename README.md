# Quantum-Geometric Market Regime Detection

A from-scratch reproduction and extension of the QCML geometric-observable
pipeline for detecting market regime shifts (Hammond 2026,
[arXiv:2605.17117](https://arxiv.org/abs/2605.17117)), reframed through
Quantum Fisher Information and the Cramer-Rao bound as estimation on a
statistical manifold.

**Reproduction vs. extension — stated plainly.** Five of the six geometric
channels — spectral entropy, reduced-density-matrix purity, ground-state energy
(used here as a full detector channel, though the source paper treats ground
energy as an incidental quantity rather than a headline detector — flagged
because it is the one channel that ever clears its floor in the count analysis
below), Berry-phase rate, and the QFI log-determinant — sit inside Hammond's
published channel taxonomy. They are **reproductions, not original in concept**:
implemented independently from the paper's equations (his source code was never
read while building them), but not claimed as novel. Only the **SLD mixed-state
QFI channel** is an extension beyond his framework (see
[v3](#v3-sld-mixed-state-qfi-novel-channel)). The same skepticism is applied to
the extension as to the reproductions throughout.

**Status: v5.5.** The detector set is **six geometric channels** — the five
reproductions above plus the novel SLD channel — and **one classical baseline**,
a Gaussian HMM; all seven are causally (past-fit) z-scored. They are evaluated
across a 15-crisis panel with per-channel noise floors and two **positive
controls**, one of which — 20-day realized volatility — rides in the panel and
null tables as an **eighth series outside the FDR family**, an instrument check
rather than a detector under test. The quantum-metric identity 4g = F_Q is
verified numerically. Nothing here claims prediction: following the source
paper, these are *contemporaneous detection* observables, not forecasters.

## Headline finding: this test has almost no power

**The evaluation used here does not have the resolution to confirm detection of
anything on this panel.** Under Benjamini–Hochberg correction across the 14-test
family (seven detectors — six geometric channels plus the Gaussian HMM baseline
— × two nulls), rank 1 requires p ≤ 0.05/14 = 0.0036 — which, for the integer-coarse
count statistic, means a channel must clear its own floor in **5 of 15 crises**,
against **0.75 expected by chance**. That threshold is the ceiling, and it holds
regardless of what any detector achieves.

For reference, 20-day realized volatility clears **3 of 15**, p = 0.039 — short
of the bar. But realized vol is a **narrow** detector: it registers volatility
spikes only, and its three hits are exactly the three vol events in the panel
(2007, 2008, COVID). Its 3 is **not** an upper bound on achievable detection —
it is what a vol-specific detector scores on a panel where only 3 of 15 events
are vol spikes. A channel that caught those three *plus* two slow-grind crises
would reach 5 and clear, which is precisely the profile this project's own
orthogonality result (Task 1, |ρ| < 0.13) predicts a geometric channel should
have.

So **no confirmatory claim about any channel is reachable on this panel**, and
none is made. This supersedes an earlier headline ("0 of 14 median tests survive
FDR ⇒ no channel detects"), which was **wrong**: the identical test says
realized volatility does not detect crises either. The statistic, not the
channels, was what that result measured. Details in
"[Multi-crisis panel](#multi-crisis-panel)"; the correction is worked through in
"[Why the statistic is the COUNT](#why-the-statistic-is-the-count-and-why-the-median-was-wrong)".

What the harness *can* do is established separately and does hold — **but only
on volatility events.** On synthetic data with a crisis injected into all 15
windows it detects comfortably (realized vol q = 0.008, Gaussian HMM q = 0.0032),
and realized vol validates the real pipeline on the three real vol spikes. **On
the other 12 crises there is no working positive control** — see "[The harness is
validated on volatility events only](#the-harness-is-validated-on-volatility-events-only)".
The code is correct; the resolution, and the reach of the control, are the
limits.

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
python scripts/null_model.py             # per-channel null floors + control
python scripts/multi_crisis_panel.py     # 15-crisis panel (the headline)
python scripts/diagnostics.py            # positive controls + bug checks
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
- **No channel in this repo has been shown to detect above chance — and the
  tests used here do not have the resolution to show it either way.** The
  positive control (20-day realized volatility) reaches only p = 0.039 on the
  panel, below what BH correction over 14 tests requires. Effect sizes are
  reported throughout; read them as un-cleared *and* as untested at any useful
  power, not as evidence of absence.

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

Pairwise Pearson and Spearman correlation of all seven causally z-scored
detectors (six geometric channels plus the Gaussian HMM baseline) on real
SPY/DIA, over the full series (not just the COVID window) —
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
Gaussian HMM baseline scores 0.08 offline / 0.09 causal — it essentially cannot
see a crisis in SPY/DIA over Jun–Oct 2015 (mostly quiet, with one violent week
around Aug 20–26 that Cohen's *d* dilutes against the calm remainder).
Independently, the causal embedding agrees least with the offline one there
(row-wise cosine mean 0.93, min 0.43 — some days near-orthogonal). With a blind
baseline and the largest embedding divergence, no individual channel's swing in
that column is trustworthy; those numbers are reported for completeness only.
The panel below shows this is **not** a China-specific quirk: under the
**realized-volatility control**, the panel script flags **6 of 15 crises at
|d| < 0.2** as barely present in SPY/DIA (five of them below 0.1), so roughly
40% of Hammond's post-2005 window list scarcely registers in this instrument
set. (The HMM baseline goes blind on a different, overlapping set of crises —
see the panel's visibility discussion, where the two instruments' disagreement
is itself the point.)

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

**1 of 28 primary-family tests survives Benjamini–Hochberg FDR at q < 0.05**
(the family is seven detectors — six geometric channels plus the Gaussian HMM
baseline — × two crises × two nulls; the realized-vol control is reported
separately, outside it): ground energy `E0` on Rate Hikes 2022, |d| = 1.57 against a null median of
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

> **Correction — an earlier version of this box said the panel showed `E0` "did
> not replicate." That claim is WITHDRAWN.** It rested on the panel *median*,
> which the median's own positive control then invalidated (realized volatility
> scores a median of 0.32 against a 0.45 floor, p = 0.93). A test that cannot
> detect realized volatility delivers no verdict on any channel, so it delivered
> none on `E0` — in **either** direction. The panel neither confirms nor
> refutes the 2022 result. See "[H1 and H2 are VOID, not
> failed](#h1-and-h2-are-void-not-failed)".
>
> What the panel *can* say, under the count statistic: `E0` clears its own floor
> in exactly one crisis — **2022 Rate Hikes, the same window** — against 0.75
> expected by chance. One hit is not evidence. It is also not a refutation.

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
series, and it is why a single-crisis test has almost no resolution.

### Does the single-crisis test have a working instrument?

The panel's median was invalidated by a positive control, so the same question
must be asked here. Realized volatility, same downstream, reported outside the
FDR family:

| crisis | channel | \|d\| | (a) floor | (b) floor | p (a) | p (b) | tau | clears? |
|---|---|---|---|---|---|---|---|---|
| COVID | **realized vol (CONTROL)** | 1.99 | 0.44 | 0.45 | **0.027** | **0.034** | 186 | **YES** |
| 2022 | **realized vol (CONTROL)** | 0.53 | 0.39 | 0.36 | 0.310 | 0.275 | 186 | no |
| 2022 | ground energy E0 | 1.57 | 0.31 | 0.30 | 0.0002 | 0.020 | 108 | YES |

**On COVID the control clears, so the single-crisis machinery works on at least
one window.** That is a materially better position than the panel median, which
cleared nothing.

**On 2022 the control fails, and reading that correctly requires separating two
cases with opposite consequences:**

- **(a) the statistic is invalid** — what happened with the panel median. Then
  the test is broken and `E0`'s 2022 result is uninterpretable.
- **(b) the control is poorly suited to this window** — either realized vol's own
  persistence gives it a structurally unreachable floor, or 2022 was a ten-month
  grind rather than a vol spike, so vol genuinely should not separate. Then the
  test is fine and `E0` stands.

**The numbers favour (b), and rule out the persistence route within it.** On
2022 realized vol's floor (0.39) sits *mid-pack* among the eight series (the
seven detectors plus the realized-vol control itself)
(range 0.23–0.55) and its tau (186) is likewise mid-pack (range 26–417) — so its
failure is **not** explained by an unusually hard bar. What is small is its
*signal*: |d| = 0.53, the 69th percentile, and six channels exceed it on that
window. The window is demonstrably not inert — `E0` reaches the 100th percentile
there against a reachable 0.31 floor.

**The diagnosis is nonetheless not settled**, and is reported as ambiguous. What
would distinguish the two remaining readings is a control that responds to a
slow-grind crisis, and we do not have one; realized vol is by construction a
control for *volatility* events. So:

- `E0`'s 2022 result is **not withdrawn** — case (a) is not supported.
- It is **not confirmed** either. It remains 1 survivor against 1.4 expected by
  chance, now with the added caveat that no working control certifies that
  particular window.

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

### The HMM control fails too — but it has been demoted, and here is why

The Gaussian HMM baseline fails to clear its floor on COVID (|d| = 0.75,
p ≈ 0.18) or on 2022 (|d| = 1.06, p ≈ 0.11). The conclusion that used to be
drawn from this — *not geometric channels being uniquely weak* — still holds,
but **the HMM is no longer the primary control**, and the reason it was
promoted to that role was never checked until now.

Two findings from `scripts/diagnostics.py` disqualify it as a reference
instrument:

1. **Its convergence is not clean.** All 15 per-crisis fits report
   `monitor_.converged == True`, so the long-standing "not converging" warning is
   not a hard failure — but **12 of 15 report convergence on a *negative* final
   log-likelihood delta.** EM is oscillating at the tolerance floor rather than
   converging monotonically. (Corroborating detail: the warning text itself is
   not bitwise reproducible across runs, differing at ~1e-13.)
2. **Its posterior saturation swings with the fit window.** Because the causal
   HMM is fit only on pre-cutoff returns, an early crisis calibrates it on a calm
   world. The **2007 fit pins 94.6% of all days near zero** and scores |d| = 0.17
   — on a window where realized volatility scores **2.20**. A control that is
   blind where the unambiguous signal is loudest is not measuring what a control
   should measure.

**Realized volatility is now the primary control**: it fits nothing, so it
isolates the evaluation from the embedding, and it has no EM to oscillate. The
HMM remains in the tables as a classical *baseline* — a competitor detector,
which is what Hammond uses it for — not as the instrument check.

Caveat that must travel with realized vol: it is strongly autocorrelated
(volatility clustering), so random null windows tend to land inside high-vol
clumps and its floor is structurally high. Check its tau and floor against the
other channels' before reading any failure of it as a broken test — see the
2022 diagnosis above, where exactly that check ruled the persistence
explanation out.

Stated precisely, because these are different claims and only the weaker one
is supported: **this analysis cannot distinguish "no signal" from "signal too
weak to detect with three crises."** Nothing here establishes that the
geometric channels do not detect regimes. What it establishes is that
**single-window Cohen's *d* on three crises cannot demonstrate that they do.**
Every effect size in this repo should be read as un-cleared until a
higher-powered protocol says otherwise. The panel below was built to be that
protocol; its own control shows it does not have the resolution to settle the
question either.

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
  BH q-values both reported. The realized-vol control is reported but **excluded
  from the family** — it is the instrument check, not a hypothesis under test.
- **Early crises may carry inflated z-scores from a short normalization
  history.** `causal_zscore` normalizes against *all* prior smoothed values, so
  the earliest crises are scored against a denominator estimated from a short and
  calm history. Measured: the expanding SD at the 2007 crisis midpoint is
  **0.0019 against ~0.009 for every later crisis — roughly 5×smaller**. The
  symptom is visible in the control: realized volatility scores |d| = **2.20 on
  the 2007 Quant Meltdown**, a stat-arb deleveraging event barely visible in
  SPY/DIA, which is implausibly high on its face. Flagged, not corrected —
  correcting it means choosing a minimum-history rule, which is a protocol change
  with its own forking paths. Read 2007 numbers, including the control's, with
  this in mind. (Tested separately for a general time trend and found none: the
  strongest per-channel Spearman is reduced purity at ρ = −0.49, p = 0.066 over
  15 points, and the expanding SD does not rise with date after 2008 — it drifts
  slowly down. So this is a 2007-specific artifact, not systematic decay.)
- **Data is pinned.** All figures come from the committed snapshot
  `data/spy_dia_close.csv` (SPY/DIA close, 2005-01-03 → 2026-06-30). See
  "Reproducibility" below — re-running these commands reproduces these numbers
  exactly.

**Implementation cross-check:** the two nulls are structurally different
constructions, yet their p-values agree to within ~0.02 on every channel
(e.g. COVID purity 0.416 vs. 0.411). Two independent routes to the same floor
is evidence the floor estimate is not an artifact of either one.

## Multi-crisis panel

```bash
python scripts/multi_crisis_panel.py
```

### Positive controls — read these first, they license everything below

Every result in this repo is a non-detection. That pattern has two completely
different explanations which, until controls existed, predicted **identical
output**: either the channels carry no detectable signal, or the evaluation
cannot detect anything at all. Two controls separate them
(`python scripts/diagnostics.py`).

**Control 1 — realized volatility through the identical downstream.** SPY's
20-day trailing realized vol, same causal z-score, same 15 masks, same Cohen's
|d|, same two nulls, same correction. It fits nothing, so it isolates the
*evaluation* from the *embedding*.

| statistic | realized vol | its own null | p |
|---|---|---|---|
| **median \|d\|** (old headline) | 0.32 | 0.45 | **0.93** |
| **count** of crises clearing own 95th-pct floor | **3 / 15** | 0.75 expected | **0.039** |

The median cannot see realized volatility. Per crisis it scores 3.05 (2008
GFC), 2.20 (2007), 1.99 (COVID) — and 0.06–0.14 on nine others, because **most
G.10 windows are not volatility events in SPY/DIA**. Not a window artifact:
under bare G.10 without the ±10 extension the median is 0.33 with the same
three crises above \|d\| = 1.

**Control 2 — end-to-end synthetic.** Synthetic returns on the *real* trading
calendar with vol/correlation spikes injected on the *real* 15 windows, so
every line of the panel runs unchanged and only the prices are manufactured.
The machinery detects it: realized vol median 0.90 (q = 0.008), Gaussian HMM
q = 0.0032 (clears both nulls), spectral entropy and reduced purity close behind
(q = 0.058, 0.087). A single-crisis synthetic run scores every channel between
0.89 and 4.71.

Supporting checks, all clean: masks align (2008 GFC and 2007 peak *exactly* at
shift 0 under a ±60-day sweep); zero NaN days inside any crisis window across
all channels; the vectorised z-score matches the production one to 7.6e-15;
smoothing width `w = 20` is at or near optimal, so Algorithm 1's trailing mean
is not diluting short windows.

**Conclusion: the code is correct and the statistic was not.**

### The harness is validated on volatility events only

Realized volatility is a **volatility** detector, so it can only ever validate
the harness on volatility events. Its three panel hits (2007, 2008 GFC, COVID)
are exactly the three vol spikes in the 15-crisis set. On the **other 12
crises** — including 2022 Rate Hikes, a ten-month grind where realized vol
*fails* (|d| = 0.53, p ≈ 0.31) but `E0` clears its floor — **there is no working
positive control at all.**

This is the concrete limitation of the current controls, and it cuts in a
specific direction: the one window where a geometric channel clears and the vol
control does not is precisely the window the vol control cannot speak to. So the
harness is *known* to work where the signal is a vol spike, and *unvalidated*
everywhere else.

What would fix it is a **second control sensitive to slow-grind crises** — a
trailing-drawdown measure, or a term-structure / vol-of-vol statistic — run
through the identical downstream. If such a control cleared on 2022 and the
other grind windows, the harness would be validated on the crisis types the
geometric channels are most likely (by Task 1 orthogonality) to be responding
to. This is on the roadmap and should precede any further reading of the
per-channel results on non-vol crises.

### Why the statistic is the COUNT (and why the median was wrong)

The headline is now the **count of crises in which a channel exceeds its own
per-crisis 95th-percentile null**.

The original argument for the median was that the median of *k* draws
concentrates the **null** at rate 1/√k — true, and measured: the null 95%
interval tightened ~2.5–3×. **The error was forgetting that the real statistic
is also a median.** Both halves must be stated:

- Against a **homogeneous** alternative (effect present in most crises), a
  median is indeed far more powerful.
- Against a **sparse** one (effect present in a minority), it is drastically
  *less* powerful: the real median collapses into the noise faster than the null
  tightens. It reads the 8th-ranked crisis, which sits in the blind majority.

The alternative here is sparse, and the control is what proved it.

**Provenance, which is what makes the new statistic usable.** The count was
selected **on the positive control alone, with the geometric channels
untouched**: among {median, mean, max, count} only the count recovered realized
vol (p = 0.039 vs 0.93 / 0.24 / 0.19). Choosing a statistic by which one
flatters the channels under test would be a forking path; choosing it on a
channel whose answer is known in advance is not.

### The power ceiling

The count is integer-coarse. Under binomial(15, 0.05) the attainable p-values
are:

| crises cleared | p | vs BH rank-1 at m = 14 (0.00357) |
|---|---|---|
| 2 | 0.171 | fail |
| 3 | 0.036 | fail |
| 4 | 0.0055 | fail |
| 5 | 0.00065 | pass |

There is nothing in between. **A channel needs 5 of 15 to survive correction**,
against 0.75 expected by chance — that is the ceiling, and it stands on the
arithmetic alone, independent of any detector. Consequently every count result
below is **exploratory**, and **no raw p-value here may be read as if FDR
correction were merely pending** — at this resolution correction is unreachable,
not outstanding.

Realized vol's 3 hits are a **reference point, not the cap.** It is a narrow
detector (vol spikes only), and 3 is what it scores on a panel with 3 vol
events — not a ceiling on what a broader detector could reach. A channel
clearing those three plus two slow-grind crises would hit 5 and clear, which is
the profile Task 1's orthogonality result (|ρ| < 0.13) predicts for a channel
that is genuinely decorrelated from vol.

### Result (exploratory)

15 crises, causal past-fit preprocessing, none skipped (smallest pre-cutoff
sample 609 rows against a 200 minimum).

| channel | count (a) | p (a) | count (b) | p (b) | tau | median (superseded) |
|---|---|---|---|---|---|---|
| Berry phase rate | 1 | 0.54 | 1 | 0.55 | 371 | 0.51 (p = 0.60) |
| Ground energy E0 | 1 | 0.54 | 1 | 0.57 | 136 | 0.41 (p = 0.69) |
| Gaussian HMM | 1 | 0.54 | 1 | 0.57 | 190 | 0.34 (p = 0.80) |
| Reduced purity | 0 | 1.00 | 0 | 1.00 | 111 | 0.71 (p = 0.038) |
| SLD QFI (w=20) | 0 | 1.00 | 0 | 1.00 | 63 | 0.46 (p = 0.30) |
| Spectral entropy | 0 | 1.00 | 0 | 1.00 | 483 | 0.40 (p = 0.85) |
| QFI log-det | 0 | 1.00 | 0 | 1.00 | 442 | 0.36 (p = 0.94) |
| **realized vol (CONTROL)** | **3** | **0.039** | **3** | **0.023** | 186 | 0.32 (p = 0.93) |

**0 of 14 tests survive BH-FDR** — but per the ceiling, that outcome was
guaranteed by the statistic's resolution and carries no information about the
channels. The informative comparison is the raw count: **no geometric channel
clears more than one crisis, against 0.75 expected by chance, while the control
clears three.** Which crises: `E0` → 2022 Rate Hikes; Berry and the HMM → 2008
GFC; control → 2007, 2008 GFC, COVID.

Note that reduced purity has the **best median (0.71) and zero count hits**. The
two statistics measure genuinely different things: purity sits moderately
elevated across many crises without ever spiking past its own 95th percentile in
any single one. Reported descriptively, not as a ranking.

**Persistence asymmetry — how to read this table across rows.** Each channel's
threshold is calibrated to its own null, and that null inherits the channel's own
persistence, so a **highly autocorrelated channel faces a structurally harder
bar**. This is correct behaviour for a per-channel null — it is what makes each
p-value valid, and the reason Hammond's 0.53 is never imported — but it means
"cleared / did not clear" is **not apples-to-apples across channels**, and the
count inherits that asymmetry directly. The `tau` column is printed so the
disparity is visible: spectral entropy and QFI log-det face tau ≈ 440–480, SLD
only 63.

This is the same mechanism already invoked *against* the SLD channel elsewhere
in this README ("lowest floor, highest N_eff — the easiest bar, and it still
showed nothing"). The corollary explains that passage rather than contradicting
it: there the asymmetry runs in SLD's favour, which is why that negative stands.

### H1 and H2 are VOID, not failed

Both were pre-registered against the **panel median**:

> H1: ground energy `E0` clears its null on the panel median.
> H2: the SLD mixed-state QFI channel clears its null on the panel median.

The median was then invalidated by its own control (0.32 against a 0.45 floor,
p = 0.93). **A test that cannot detect realized volatility delivers no verdict on
any channel.**

**H1 is VOID. H2 is VOID.** Void, not *failed* — "failed" would mean tested and
rejected; these were tested with a demonstrably invalid instrument, which yields
no evidence in either direction. This is the same discipline as the `E0`
shift-null correction elsewhere in this README, where a floored p-value was a
**bound** rather than a rejection. In particular, **the earlier claim that `E0`
"did not replicate" on the panel is withdrawn** — that was a median result.

**No new hypotheses are pre-registered against the count.** The ceiling shows no
confirmatory claim is reachable with it here. Pre-registering against a test
proven unable to deliver a verdict is the *appearance* of discipline, not
discipline. Pre-registration is deferred to the next protocol that might have
power: multi-asset, or false-alarms-per-year.

### The synthetic-panel finding — the most informative result about the extension this round

This one does **not** depend on the median and is not void. On the synthetic
panel the alternative is homogeneous by construction (a spike is injected into
all 15 windows), so the median is a valid statistic *there* — and the harness
demonstrably detects (realized vol 0.90, HMM q = 0.0032). Under those
conditions:

> **The SLD mixed-state QFI channel lands at 0.29, below its own null (p = 0.77).
> QFI log-det lands at 0.33, likewise below its own null (p = 0.85).** Both fail
> to respond to a crisis that spectral entropy (0.81), reduced purity (0.74),
> Berry (0.66) and `E0` (0.65) all register.

This is the most informative result about the extension this round, **with
external validity limited by the synthetic construction**: `synthetic_prices()`
injects a volatility/correlation spike, so what it establishes is that SLD does
not respond to *that pattern*. How well an injected spike resembles real regime
change is unestablished, and this is not a claim about real crises.

### Per-crisis table — descriptive only, with a visibility column

`multi_crisis_panel.py` prints the full grid. It is **not** mined for which
channel wins which crisis type: Hammond tested for exactly that specialization
and **found none (p = 0.31)**, so any such pattern across 15 crises is almost
certainly noise.

The realized-vol column is a **visibility reference**: it shows which crises
exist at all in this instrument. The panel script flags **6 of 15 crises at
control |d| < 0.2** — 2010 Flash Crash (0.14), 2015 China (0.06), 2018
Volmageddon (0.07), 2018 Q4 (0.06), 2019 Repo (0.06), 2023 SVB (0.09) — five of
them below 0.1, so roughly 40% of Hammond's post-2005 list is barely present in
SPY/DIA. Where the control is ~0.1, no channel's number in that row should be
read as a miss.

**Two visibility instruments disagree — and the disagreement is the point.** The
Gaussian HMM baseline has its own blind set, and it is *not* the vol control's.
The two agree the market is quiet on China (vol 0.06 / HMM 0.09), Volmageddon
(0.07 / 0.06) and 2019 Repo (0.06 / 0.09), but part ways sharply elsewhere: the
HMM goes dark on 2007 (|d| = 0.17) and 2024 Carry (0.16) where the vol control
screams on 2007 (2.20) and sees 2024 (0.32); conversely the vol control is blind
on 2018 Q4 (0.06) and SVB (0.09), both of which the HMM registers (0.83, 0.34).
Neither is "the" visibility measure — that two narrow instruments disagree about
which crises even register is itself a fact about how thin SPY/DIA is, not a
contradiction. (The 2007 split is the short-normalization-history artifact
flagged under Null-model methodology; realized vol's 2.20 there is implausibly
high on its face.) The vol control is the one carried in the FDR-external column
above because it fits nothing; the HMM figure is its own baseline blind set.

**All 15 are kept, unfiltered.** The count statistic calibrates a floor *per
crisis*, so an invisible crisis contributes null draws rather than false
positives — sparsity is handled by construction. Filtering would double-solve it,
and any filtering criterion is a forking path.

### Two nulls, and why they are not symmetric

- **(a) Random matched-length windows**, drawn per crisis **independently**.
  Independence destroys the cross-crisis dependence the real 15 values have
  (they come from one market), making the null tighter than it should be — **an
  easier bar. Anti-conservative**, for the count exactly as it was for the median.
- **(b) One common circular shift** applied to all 15 series at once, masks
  fixed. Preserves cross-crisis dependence exactly, so it is the **conservative**
  one. Caveat specific to the count: because one shift moves all 15 crises
  together, **the count's null under (b) is lumpy** — much higher variance than
  under (a) — and against an already integer-coarse statistic that makes the (b)
  p-values low-resolution. Read them as coarse, not precise.

### Block-bootstrap CIs — scope

Computed on the **median only**, and retained as a precision statement about
that now-superseded statistic (widths 0.36–0.52; Politis–White block lengths
144–180). **No CI is computed on the count**: it is an integer over 0–15, so a
percentile interval would be theatre rather than information.

Two caveats on the median CIs, both printed by the script: Politis–White's
autocorrelation search **saturates on every channel**, so each block length is a
lower bound on that channel's persistence and each interval is if anything too
narrow; and with blocks of ~150–180 against windows as short as 62 days, a crisis
can miss a replicate entirely, so on average **11.5 of 15 crises** contribute per
replicate (5th percentile 9).

### What this does and does not license

**It does not establish that the geometric channels fail to detect regimes**, and
it does not establish that they detect them. The honesty architecture cuts both
ways. What the panel establishes is narrower and firmer:

> At the resolution available here — 15 crises, a count statistic whose own
> positive control reaches only p = 0.039 — **this repo cannot support a
> confirmatory detection claim about any channel, geometric or classical.** What
> it can say is that no geometric channel clears more than one crisis where
> realized volatility clears three.

The remaining candidate explanations, none of which this analysis adjudicates
between: the effects are genuinely small; Cohen's *d* against a rest group 26%
composed of other crises is too blunt; SPY/DIA is too narrow an instrument set
(a third of the panel is barely visible in it); or the event-study metric itself
(Gap 2) is the wrong target and a deployment metric like false-alarms-per-year
would behave differently. Those are the next tests, not conclusions.

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
- **v5 — multi-crisis panel — DONE; result is a POWER finding, not a channel
  finding.** 15 crises (Hammond Table G.10 post-2005, ±10 trading days), causal
  past-fit preprocessing, both nulls, Politis–White block-bootstrap CIs. The
  panel shipped first with a **median** headline; its own positive control then
  invalidated that statistic (realized vol median 0.32 vs a 0.45 floor,
  p = 0.93), so the headline is now the **count** of crises clearing their own
  per-crisis floor. What the panel established is that **the evaluation tops out
  at p = 0.039 on a signal nobody disputes**, so no confirmatory claim about any
  channel is reachable here. Pre-registered H1/H2 are **VOID, not failed**. See
  "[Multi-crisis panel](#multi-crisis-panel)".
- **v5.5 — harness diagnostics — DONE.** `scripts/diagnostics.py`: two positive
  controls plus five bug checks (HMM convergence, mask alignment, smoothing
  ablation, normalization drift, NaN audit). Established that the **code is
  correct** — synthetic panel detects an injected crisis, masks peak exactly at
  zero shift, no missing days — and that the **statistic was the problem**. Also
  demoted the HMM from control to baseline and flagged the 2007 short-history
  artifact.
- **v6 — what comes next, now that power is the binding constraint.** More
  crises will not help; the count already saturates its resolution at 15. The
  live options change *what is measured*: (i) **a second positive control
  sensitive to slow-grind crises** (trailing drawdown, or a term-structure /
  vol-of-vol statistic) — realized vol validates the harness on vol events only,
  so the other 12 crises, including the 2022 window where `E0` clears, are
  currently unvalidated; this should come *first*, before further reading of
  per-channel results on non-vol crises; (ii) **Task 4, multi-asset** — a third
  of the panel is barely visible in SPY/DIA at all, so widening the instrument
  set raises the number of crises that *exist* to be detected, which is the one
  lever that moves the count's ceiling; (iii) **roadmap item 4/5**,
  false-alarms-per-year under an expanding-window protocol, which closes Gap 2
  and replaces the event-study metric entirely; (iv) **Task 3.5**, the classical
  Bures/MMD baseline, still the decisive test for whether the SLD channel is more
  than a relabeled classical statistic. **Pre-registration is deferred to
  whichever of these is built** — see the panel section for why registering
  hypotheses against the current statistic would be theatre.

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
  baseline.py     Gaussian HMM (baseline) + realized_vol_series (CONTROL)
  data.py         synthetic generator + pinned snapshot loader
scripts/
  run_demo.py             synthetic smoke run
  channel_correlations.py Task 1 — redundancy matrix
  causal_eval.py          Task 2 — offline vs. causal past-fit |d|
  null_model.py           per-channel noise floors, single crises + control
  multi_crisis_panel.py   15-crisis panel — count statistic, ceiling
  diagnostics.py          positive controls + bug checks; statistic provenance
tests/
  test_smoke.py  test_geometry.py  test_sld.py
```

`qgmrd/crises.py` is the single source of truth for crisis windows. Every
script imports it, so the window convention cannot drift between them — which
is the bug the convention change fixed.
