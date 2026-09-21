# Quantum-Geometric Market Regime Detection

A from-scratch reproduction and extension of the QCML geometric-observable
pipeline for detecting market regime shifts (Hammond 2026,
[arXiv:2605.17117](https://arxiv.org/abs/2605.17117)), reframed through Quantum
Fisher Information and the Cramér–Rao bound as estimation on a statistical
manifold. **Status: v5.11.**

**Headline: this test has almost no power.** Six geometric channels plus a
Gaussian-HMM baseline are evaluated across a 15-crisis panel (SPY/DIA, Hammond
Table G.10 ±10) with per-channel noise floors and two positive controls. **No
confirmatory detection claim is reachable at this resolution:** the count
statistic needs [5 of 15 crises](#power-ceiling) to survive FDR (0.75 expected by
chance), and the positive control, 20-day realized volatility, tops out at 3 of
15, **p = 0.039**, short of the bar. The code is correct; the resolution, and the
reach of the control ([validated on volatility events
only](#validated-on-volatility-events-only)), are the limits. Every effect size
is un-cleared *and* untested at useful power, not evidence of absence. Nothing
claims prediction: these are contemporaneous *detection* observables.

## Results summarized

- **No confirmatory detection is reachable at this resolution.** The count statistic needs 5 of 15 crises to survive FDR; the realized-volatility control reaches 3 of 15 (p = 0.039), below the bar, and no geometric channel clears more than one crisis.
- **The SLD mixed-state channel is decorrelated but undetected.** The novel extension correlates with every other channel at |ρ| < 0.13 (empirically distinct, not a redundant repackaging of the pure-state channels), yet does not clear its noise floor on any window tested. Orthogonality and detection are independent properties, and only the first is established.
- **Ground energy `E0` is the only test to survive FDR correction** (2022 Rate Hikes, 1 of 28 primary-family tests), but at 1 survivor against 1.4 expected by chance, it is not read as a detection.
- **Reduced purity reproduces Hammond's null result:** it leads rankings without clearing noise (p = 0.18 in his pipeline, p ≈ 0.07 to 0.09 here), an independent reproduction of a negative.
- **Volatility validates the harness; a slow-grind control does not.** Realized vol clears 2007/2008/COVID; a 252-day drawdown control built for the other 12 (incl. 2022) clears only 2008 GFC and misses 2022 (|d| = 0.76 vs. floor 0.42), because its own persistence inflates the floor faster than a real decline can separate from it.
- **A known decline is also undetected.** A synthetic ground-truth test (decline only, vol/corr untouched) still gets 0 of 9 channels past FDR: the harness lacks power for slow declines generally, not a defect in one control.
- **Threshold quantified: ~5x real 2022.** A magnitude sweep shows drawdown/`E0` only clear past ≈53% decline (real 2022: −14%), deeper than 2008 GFC. A trend-matched null doesn't change this.
- **A third, non-circular control narrows this to ~3x.** `trailing_return_126d` (bounded memory, unlike drawdown's peak-tracking) matches reduced purity's threshold: real progress, though still short of 2022's actual ≈14%.
- **SLD does not beat classical baselines built to mimic it.** Four channels, from raw-feature classical distances to the exact classical part of quantum Fisher information, all land in the same statistical range as SLD. A synthetic check shows the quantum coherence term can matter in principle, just not here.
- **Drawdown's weaker threshold traces to what it measures, not just autocorrelation.** It tracks the worst point reached in a window, not the net outcome, so ordinary dips that fully recover still count. Trailing return tracks net decline much more tightly (r = 0.79 vs. 0.53 at 2022's window length), giving it a cleaner null.

## Method

Per trading day, a rolling feature vector `x_t ∈ ℝ¹¹` (returns, 5/20-day vol,
momentum, cross-correlation; PCA to 8 components) defines an **error Hamiltonian**
and its ground state:

```
H(x) = ½ Σ_k (A_k − x_k I)²         A_k fixed random Hermitian, seeded
|ψ(x)⟩ = argmin ⟨ψ|H(x)|ψ⟩ ∈ ℂⁿ     via np.linalg.eigh
```

Geometry is read off the **quantum geometric tensor** of `|ψ(x)⟩`: real part =
Fubini–Study metric `g = ¼ F_Q`, imaginary part = Berry curvature. Six geometric
channels, converted to causal past-only z-scores (Algorithm 1):

- **Spectral entropy**: Shannon entropy of the excitation spectrum.
- **Reduced purity**: `tr(ρ_A²)` after a bipartition; low = factor structure breaking.
- **Ground energy `E₀`**: used as a full detector (source treats it as incidental, §6.3).
- **Berry phase rate**: gauge-invariant Wilson-loop plaquette.
- **QFI log-det**: `log det⁺ F_Q`.
- **SLD mixed-state QFI**: the extension (below).

**Reproduction vs. extension.** The first five sit inside Hammond's published
taxonomy, reproductions, implemented independently from the paper's equations
(his source was never read while building them), not claimed as novel. Only the
SLD channel is beyond his framework; the Gaussian HMM is a classical baseline
(Hammond's own). The quantum-metric identity `4g = F_Q` is verified numerically:
corr(g_FD, g_PT) = 1.000000000, max rel. error ~1e-5. All windows are Table G.10
±10 trading days (his §4.1), defined once in `qgmrd/crises.py`. (The prior ad-hoc
convention and its effect on the numbers are in `CHANGELOG.md`.)

### SLD mixed-state QFI, the extension {#sld-orthogonal-not-detected}

A rolling window of embedded states is a mixed state; its QFI w.r.t. time is
computed via the symmetric logarithmic derivative (SLD) of Dingilian, Kurella et
al., *J. Chem. Phys.* (2026), doi:10.1063/5.0316287, Eqs. (8)–(12):

```
ρ_t = (1/w) Σ_s |ψ(x_s)⟩⟨ψ(x_s)|     F_Q = Tr(L²ρ),  ∂ρ = ½(Lρ + ρL)
w = 1 limit ⇒ F_Q → 4 g_aa           tests/test_sld.py::test_pure_state_limit
```

This measures how distinguishable the recent ensemble is from its immediate past,
a detector the pure-state shortcut `4g = F_Q` cannot express once ρ is mixed.
**Novelty source-verified:** grepped Hammond's `qcml_geometry/*.py` for
`sld`/`symmetric_log`/`logarithmic_derivative` → zero matches; his nearest
neighbor `QuantumRelativeEntropyDetector` builds the same mixed state but computes
quantum relative entropy (Kubo–Mori metric, not SLD/Bures, which coincide only
when states commute). His reference state uses an expanding window; ours a fixed
rolling `w`.

**SLD is decorrelated but undetected.** On real SPY/DIA (causal z-scores, full
series) SLD's largest |correlation| with any other channel is **~0.13** (vs.
spectral entropy: ρ = 0.121 Pearson, 0.126 Spearman; largest Pearson/Spearman gap
0.06), roughly half Hammond's geometric-classical benchmark (mean |ρ| ≈ 0.22).
The other six correlate far more (purity vs. `E0` −0.90; Berry vs. QFI log-det
~0.70–0.74; spectral vs. HMM ~0.64–0.71). So the mixed-state generalization is
**empirically distinct** (not a redundant repackaging of the pure-state channels it generalizes), but orthogonality and
detection are independent, and only the first is established: SLD does not clear
its floor on any window tested (below). (Pearson/Spearman diverge >0.1 for two
Berry-phase-rate pairs, a monotonic-nonlinear step-difference effect; doesn't
touch the SLD conclusion.)

## Results

### Offline event study (SPY/DIA, COVID-2020 window, G.10 ±10 vs. rest)

**Offline** = preprocessing fit on all data (crisis included), so this measures
separability, not causal detection. None clears its own noise floor (below); read
the ordering as descriptive.

| method | type | \|d\| |
|---|---|---|
| Gaussian HMM (high-var prob) | baseline | 0.78 |
| Spectral entropy | geometric v0 | 0.76 |
| Reduced purity | geometric v0 | 0.63 |
| Berry phase rate | geometric v2 | 0.58 |
| Ground energy E0 | geometric v2 | 0.43 |
| QFI log-det | geometric v2 | 0.25 |
| SLD mixed-state QFI (w=20) | geometric v3 | 0.08 |

### Causal (past-fit) preprocessing, Task 2 (`python scripts/causal_eval.py`)

Per crisis, `StandardScaler`/`PCA`/HMM are fit only on rows before
`cutoff = crisis_start − 10 business days` (the buffer stops 20-day rolling
features leaking backward), then the full timeline is transformed through them.
Operators are data-independent. Closes **Gap 1** (leaky preprocessing); **Gap 2**
(Cohen's *d* still scores against future days) remains, so this is a causal event
study, not a walk-forward. Point estimates on one realized path; no CIs.

| channel | COVID 2020 | Rate Hikes 2022 | China 2015 |
|---|---|---|---|
| Reduced purity | 0.63 → 0.56 | 0.90 → 0.85 | 0.93 → 0.90 |
| Spectral entropy | 0.76 → 0.79 | 1.18 → 1.23 | 0.70 → 0.48 |
| Ground energy E0 | 0.43 → 0.41 | 1.60 → 1.57 | 0.49 → 0.66 |
| Berry phase rate | 0.58 → 0.57 | 0.96 → 0.88 | 0.22 → 0.77 |
| SLD QFI (w=20) | 0.08 → 0.25 | 0.03 → 0.09 | 0.51 → 0.51 |
| QFI log-det | 0.25 → 0.32 | 1.15 → 1.04 | 0.74 → 0.23 |
| Gaussian HMM | 0.78 → 0.75 | 1.06 → 1.06 | 0.08 → 0.09 |

- **China 2015 is uninterpretable** and not analyzed per-channel: the HMM baseline
  scores 0.08 / 0.09 (blind), and the causal embedding agrees least with offline
  there (row-wise cosine mean 0.93, min 0.43).
- **Preprocessing reaches the embedding but not the channels** (interpretable
  crises): PCA axes stay aligned (min |cos| 0.98 / 0.99 / 0.94); the scaler shift
  is dominated by `xcorr20` (Δscale/scale 0.40 / 0.35 / 0.60), the other ten
  features ≤ 0.09 (median 0.04–0.05, 1/11 above 0.10); daily embedding cosine mean
  0.98 / 0.99 / 0.93 (min 0.83 COVID), yet causal-vs-offline z-series correlate
  0.94–1.00 on COVID/2022 (frame-sensitive Berry and QFI-log-det lowest 0.94–0.96),
  channel robustness, not metric coarseness. (China series-corr drops to 0.79.)
- **Reduced purity: no collapse**: moves ≤ 0.07 (0.56 / 0.85 / 0.90); panel
  median 0.71. The type-correct comparison is Hammond's Table 3 past-fit median
  *d* ≈ 0.83 (his §5.1 fits scaler+PCA+operators pre-crisis). Pipelines differ:
  11 features vs. ~52, 8 PCA vs. 15, fixed operators vs. his refit, 15 crises vs.
  his 17, so "same range" is the honest ceiling, coincidence not excluded. His
  *d* ≈ 0.26 is a *different* protocol (§5.2 frozen holdout / expanding-window,
  which closes Gap 2), out of Task 2's reach. Hammond's own null test: purity real
  median *d* = 0.73 vs. null ~0.53, 95% [0.30, 0.89], **p = 0.18**: leads a
  ranking without clearing noise.
- **2022, geometric channels lead:** `E0` = 1.60 (table-high), spectral 1.18, QFI
  log-det 1.15, above the HMM baseline (1.06). One window; `E0`'s 15-crisis median
  is 0.41, below its own null median.
- **SLD** is blind on 2022 (0.03 / 0.09) and on COVID under G.10 ±10 (0.08 →
  0.25); the COVID move matches QFI log-det's +0.07 in the same window
  (frame-sensitivity, not signal).

### Null-model tests (`python scripts/null_model.py`)

Two per-channel nulls on the causal z-scored series: **(a) random matched-length
windows** (5000 seeded draws, non-overlapping the crisis, are the dates
special?); **(b) circular shift** (all M−1 offsets, mask fixed, is the alignment
real?). Floors are computed per channel, never imported (Hammond's ~0.53); ours
range **0.23 to 0.69**.

**1 of 28 primary-family tests survives BH-FDR at q < 0.05** (7 detectors × 2
crises [COVID, 2022] × 2 nulls; control excluded): `E0` on 2022, |d| = 1.57 vs.
null median 0.31, not one of the 5000 random windows reached it (100th pct; raw
p = 0.0002, q = 0.006). Expected chance survivors at α = 0.05: **1.4**, so one is
barely above the false-positive rate, not a detection. The (b) shift null is
**indeterminate** here: q = 0.277 is floored at 1/(N_eff+1) ≈ 0.020 (N_eff ≈ 50; under 2% of shifts exceeded the real value),
a bound ("at most 0.020"), not a rejection.

`pct` = percentile of the real |d| within its own null (high = strong); `p` =
fraction of null draws at or above it (low = strong).

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

Note the width of those null 95% intervals (up to [0.03, 1.98]), the free lunch a
single short window gets from an autocorrelated, fat-tailed series.

**Does the single-crisis test have a working instrument?** Realized vol, same
downstream, outside the FDR family:

| crisis | channel | \|d\| | (a) floor | (b) floor | p (a) | p (b) | tau | clears? |
|---|---|---|---|---|---|---|---|---|
| COVID | **realized vol (CONTROL)** | 1.99 | 0.44 | 0.45 | **0.027** | **0.034** | 186 | **YES** |
| 2022 | **realized vol (CONTROL)** | 0.53 | 0.39 | 0.36 | 0.310 | 0.275 | 186 | no |
| 2022 | ground energy E0 | 1.57 | 0.31 | 0.30 | 0.0002 | 0.020 | 108 | YES |

On COVID the control clears (single-crisis machinery works on ≥1 window). On 2022
it fails, and the numbers favour **case (b), control poorly suited** over **(a),
statistic invalid**: on 2022 its floor (0.39) is mid-pack (range 0.23–0.55) and
its tau (186) mid-pack (range 26–417), so not an unusually hard bar. Its signal
is just small (|d| = 0.53, 69th pct, six channels exceed it). So `E0`'s 2022
result is **not withdrawn** (case (a) unsupported) and **not confirmed** (no
working control certifies that window; still 1 survivor vs. 1.4 expected).

**SLD, reported as pre-registered** (commitment recorded before the run):
mathematically correct (`test_pure_state_limit`), novel, decorrelated (|ρ| <
0.13), and **not demonstrated to detect above chance**: COVID p ≈ 0.65, 2022
p ≈ 0.76, China p ≈ 0.40 (both nulls), nor on the panel median (p ≈ 0.28–0.30,
q = 0.95). A *stronger* negative because SLD had the **lowest floor** (0.23–0.42)
and by far the **least autocorrelated series** (τ ≈ 26 vs. ≈ 400 for spectral
entropy; N_eff ≈ 205 vs. ≈ 13), the easiest bar and the most power, and it shows
nothing. (On the panel SLD's median 0.46 sits above its own null median 0.39 and
the HMM's 0.34, not a detection, p ≈ 0.28.)

**Reduced purity reproduces Hammond's null result:** 2022 |d| = 0.85 at the 92nd
pct, p ≈ 0.07–0.09, does not clear, matching his real median *d* = 0.73, **p =
0.18** from an independent pipeline (11 features vs. ~52, 8 PCA vs. 15, fixed vs.
refit operators). Panel median is the closest any channel comes (raw p ≈
0.034–0.038) and still fails FDR (q = 0.26).

**HMM demoted to baseline.** It fails its floor on COVID (|d| = 0.75, p ≈ 0.18)
and 2022 (|d| = 1.06, p ≈ 0.11), so this is *not* geometric channels being
uniquely weak. But diagnostics disqualify it as the *control*: 12 of 15 fits
converge on a *negative* final log-likelihood delta (EM oscillating at the
tolerance floor; the warning text is not bitwise reproducible, differing at
~1e-13), and posterior saturation swings with the fit window. The 2007 fit pins
**94.6%** of days near zero and scores |d| = 0.17 where realized vol scores 2.20.
Realized vol (fits nothing) is now the primary control. **This analysis cannot
distinguish "no signal" from "signal too weak to detect with three crises"**. Only the weaker claim is supported.

### Multi-crisis panel (`python scripts/multi_crisis_panel.py`)

**Positive controls license everything below.** (1) Realized vol through the
identical downstream:

| statistic | realized vol | its own null | p |
|---|---|---|---|
| **median \|d\|** (old headline) | 0.32 | 0.45 | **0.93** |
| **count** of crises clearing own 95th-pct floor | **3 / 15** | 0.75 expected | **0.039** |

Per crisis it scores 3.05 (2008 GFC), 2.20 (2007), 1.99 (COVID) and 0.06–0.14 on
nine others (bare G.10 gives median 0.33, same three above |d| = 1). (2)
End-to-end synthetic (spikes injected on the real 15 windows): realized vol median
0.90 (q = 0.008), HMM q = 0.0032, spectral/purity behind (q = 0.058, 0.087);
single-crisis synthetic scores every channel 0.89–4.71. (3) **Pure-drift
synthetic** (vol/corr untouched, isolating a known 2022-scale decline):
**0 of 9 channels survive FDR** (18 tests, 0.9 expected). Reduced purity
closest (q = 0.066); drawdown highest |d| (0.85) but still fails (q = 0.29),
[detail here](#validated-on-volatility-events-only). Supporting checks clean:
masks peak exactly at shift 0 (±60-day sweep); zero NaN crisis days; vectorised
z-score matches production to 7.6e-15; w = 20 near-optimal. **The count replaced
the panel median** (which its own control invalidated, post-mortem in
`CHANGELOG.md`), selected on the control alone.

#### Power ceiling {#power-ceiling}

The count is integer-coarse. Under binomial(15, 0.05):

| crises cleared | p | vs BH rank-1 at m = 14 (0.00357) |
|---|---|---|
| 2 | 0.171 | fail |
| 3 | 0.036 | fail |
| 4 | 0.0055 | fail |
| 5 | 0.00065 | pass |

**A channel needs 5 of 15 to survive correction**, against 0.75 expected by chance,
the ceiling, on arithmetic alone, independent of any detector. Every count
result is **exploratory**; no raw p may be read as if FDR correction were merely
pending (it is unreachable). Realized vol's 3 is a **reference point, not a cap**,
a narrow vol-only detector on a panel with 3 vol events; a channel clearing those
plus two slow-grind crises would hit 5, the profile Task 1's orthogonality
(|ρ| < 0.13) predicts.

#### Result (exploratory)

15 crises, causal past-fit, none skipped (smallest pre-cutoff 609 rows vs. 200
minimum).

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
| **drawdown 252d (CONTROL 2)** | **1** | **0.54** | **1** | **0.65** | 306 | 0.43 (p = 0.82) |

**0 of 14 tests survive BH-FDR**: guaranteed by the statistic's resolution, no
channel information. The informative comparison is the raw count: no geometric
channel clears more than one crisis (0.75 expected) while realized vol clears
three and drawdown, the control built to catch what vol misses, clears only
one: 2008 GFC, a vol crisis realized vol already caught, not a new slow-grind
one ([why it fails](#validated-on-volatility-events-only)):
`E0` → 2022 Rate Hikes; Berry and HMM → 2008 GFC; realized vol → 2007, 2008 GFC,
COVID; drawdown → 2008 GFC. Reduced purity has the best median (0.71) and zero count hits (elevated
across many crises, never spiking past its 95th pct in one). **H1/H2 (`E0`/SLD
clear the panel median) are VOID**: tested against a statistic its control
invalidated, so no verdict either way (post-mortem in `CHANGELOG.md`).
**Persistence asymmetry:** each threshold is calibrated to the channel's own null,
so a high-tau channel faces a structurally harder bar; "cleared / did not clear"
is **not apples-to-apples** across rows (spectral entropy and QFI log-det face
tau ≈ 440–480, SLD only 63); `tau` is printed so the disparity is visible.

**Synthetic-panel finding** (not void, homogeneous alternative, so the median is
valid there): SLD lands at 0.29 (p = 0.77) and QFI log-det at 0.33 (p = 0.85),
both below their own nulls, where spectral (0.81), purity (0.74), Berry (0.66) and
`E0` (0.65) respond. External validity limited to the injected vol/correlation
spike, not a claim about real crises.

**Per-crisis visibility** (descriptive only; not mined, Hammond found no
crisis-type specialization, p = 0.31). The realized-vol column flags **6 of 15
crises at control |d| < 0.2**: 2010 Flash Crash (0.14), 2015 China (0.06), 2018
Volmageddon (0.07), 2018 Q4 (0.06), 2019 Repo (0.06), 2023 SVB (0.09), five below
0.1, so ~40% of the list barely registers in SPY/DIA. The HMM's blind set differs
and overlaps: both quiet on China (0.06 / 0.09), Volmageddon (0.07 / 0.06) and
2019 Repo (0.06 / 0.09); HMM dark on 2007 (0.17) and 2024 Carry (0.16) where vol
screams (2.20) / sees (0.32), while vol is blind on 2018 Q4 (0.06) and SVB (0.09)
which HMM registers (0.83, 0.34). All 15 kept unfiltered. The count calibrates a
floor per crisis, so an invisible crisis contributes null draws, not false
positives.

**Two nulls are not symmetric:** (a) drawn per crisis independently → destroys
cross-crisis dependence → tighter, **anti-conservative**; (b) one common shift →
preserves it, **conservative** but lumpy (high-variance, low-resolution count p).
**Block-bootstrap CIs on the median only** (widths 0.36–0.52; Politis–White block
lengths 144–180, search saturates so lengths are lower bounds; ~11.5 of 15 crises
contribute per replicate, 5th pct 9). No CI on the integer count.

**What this licenses:** not that the channels fail to detect, nor that they detect.
At this resolution (15 crises, control p = 0.039) no confirmatory claim is
supportable, only that no geometric channel clears more than one crisis where
realized vol clears three. Open, unadjudicated explanations: effects genuinely
small; rest-group contamination; SPY/DIA too narrow; or Gap 2 the wrong target
(FAR would behave differently).

### Validated on volatility events only {#validated-on-volatility-events-only}

Realized vol validates the harness only on its own three vol crises (2007,
2008 GFC, COVID). On the other 12, including 2022 (vol |d| = 0.53, p ≈ 0.31,
but `E0` clears), a second control was built: **252-day trailing drawdown**
(`qgmrd/baseline.py::drawdown_series`). **It fails too:**

| crisis | \|d\| | floor | p | tau | clears? |
|---|---|---|---|---|---|
| 2022 Rate Hikes | 0.76 | 0.42 | 0.12 | 306 | no |
| 2020 COVID | 0.61 | 0.49 | 0.21 | 306 | no |
| panel-wide (count) | 1/15 | 0.75 expected | 0.54 | -- | -- |

Its one hit (2008 GFC) is a vol crisis realized vol already caught. **Why:
persistence, not absence of signal.** Drawdown stays "underwater" by
construction until price recovers, so it's more autocorrelated than vol
(τ ≈ 306 vs. 186). Its null 95% interval is [0.04, 3.89], wide enough that
even a real decline doesn't stand out from a random window in a trending
series. Being shaped like a decline isn't enough on its own.

### Is a slow-grind decline even measurable here? {#is-a-slow-grind-decline-even-measurable-here-a-known-ground-truth-test}

A known-ground-truth test (`diagnostics.py` TEST 2b) injects a decline we know
for certain is there (vol/corr untouched) on the real 15-window calendar.
Sanity check: realized vol stays at its null median (0.38), confirming
nothing leaked into variance. **0 of 9 channels survive FDR** (18 tests, 0.9
expected by chance). Reduced purity comes closest (q = 0.066), `E0` next
(q = 0.15); drawdown scores the highest |d| (0.85) but still fails (q = 0.29),
the same persistence problem as before. This reframes the drawdown/2022
result as harness-wide underpower rather than a bad statistic choice. It
doesn't prove declines are unmeasurable in principle, only that this pipeline
can't confirm one even when it's certainly there.

A **trend-matched null** (TEST 2c: compare only against other windows with a
similarly-sized decline, not any random one) doesn't help either. Real 2022's
p stays around 0.14, and the synthetic result barely changes. So raw decline
size isn't what inflates the floor; the channel's own persistence (τ) is.

A **magnitude sweep** (TEST 2d: inject a known decline into 2022's exact
window, scaled from 0.5x to 8x its real size) pins down the ceiling:

| × real 2022 | decline | drawdown \|d\| | `E0` \|d\| | clears? |
|---|---|---|---|---|
| 1× (real) | −14.0% | 0.09 | 0.12 | no |
| 3× | −36.4% | 1.06 | 0.48 | no |
| **5×** | **−53.0%** | **1.91** | **0.74** | **YES** |
| 8× | −70.1% | 2.79 | 0.92 | YES |

**Detecting a slow grind needs a decline roughly 5x the real 2022 move
(≈53%), deeper than 2008 GFC itself.** That's a quantified power ceiling, not
a vague null. It also explains why drawdown clears 2008 GFC: its real
magnitude (≈50%+) happens to sit near this threshold, regardless of its
vol-crisis label.

### A third control that isn't circular {#a-third-control-non-circularly-matching-puritys-threshold}

Purity's edge traces back to its short autocorrelation (τ ≈ 28–83, vs.
drawdown's τ ≈ 160–233). But purity can't be used as a control itself: it's
an output of the embedding pipeline under test, so using it to validate that
pipeline would be circular. So a non-circular version was built instead:
**`trailing_return_series`** (`qgmrd/baseline.py`), a fixed-window cumulative
return whose memory is bounded by the window, unlike drawdown's sliding peak.

Window caveat: 126 days was chosen because it gave the best |d| on real 2022
data in a 5-126 day sweep. The ground-truth synthetic actually preferred a
shorter window (60 days, weakly: |d| = 0.33, p = 0.56). That mismatch is
flagged, not resolved.

| × real 2022 | decline | \|d\| | tau | p | clears? |
|---|---|---|---|---|---|
| 1× | −14.0% | 0.19 | 132 | 0.85 | no |
| 2× | −26.1% | 0.76 | 144 | 0.32 | no |
| **3×** | **−36.4%** | **1.28** | **155** | **0.04** | **YES** |
| 5× | −53.0% | 2.16 | 173 | <0.001 | YES |

**Clears at 3x**, matching purity and beating drawdown's 5x, confirmed by the
trend-matched null too. Its autocorrelation is only modestly lower than
drawdown's (~155–187 vs. ~198–233) even though the threshold nearly halves,
so autocorrelation isn't the whole explanation. It still doesn't clear 2022
itself (3x is about 36%, real 2022 was about 14%): the gap narrows, but
doesn't close.

**Why the gap is bigger than tau predicts.** Across every crisis-length window
tested (62 to 229 days), trailing return's mean value correlates with the
window's own net decline much more tightly than drawdown's does (r = 0.79 vs.
0.53 at 229 days, the 2022 length; r = 0.45 vs. 0.36 at 62 days). Drawdown
tracks the worst point reached, not the net outcome, so an ordinary choppy dip
that fully recovers still registers. Trailing return mostly requires an actual
sustained decline to register. That means far more of ordinary market history
produces a moderately elevated drawdown reading than a large trailing-return
reading, which inflates drawdown's null floor for reasons that have nothing to
do with autocorrelation.

### Classical baselines for SLD: no coherence advantage found {#classical-baselines-for-sld}

SLD is this project's one genuinely novel channel, everything else reproduces
Hammond's taxonomy. Question: does the quantum formalism (embedding, mixed
state, symmetric logarithmic derivative) add real information, or does a
classical statistic on the same inputs do just as well? Four channels test
this, each isolating one more layer:

| channel | isolates | median \|d\| | 95% CI | count |
|---|---|---|---|---|
| `classical_bures_w20` | raw features, Gaussian distance, no embedding | 0.45 | [0.32, 0.83] | 0/15 |
| `classical_mmd_w20` | raw features, nonparametric distance, no embedding | 0.40 | [0.29, 0.85] | 1/15 |
| `frobenius_rho_w20` | same rho_t as SLD, naive Euclidean distance | 0.38 | [0.20, 0.85] | 0/15 |
| `classical_pop_fisher_w20` | same rho_t, classical Fisher info, no coherence | 0.41 | [0.22, 0.60] | 2/15 |
| `sld_qfi_w20` | full quantum Fisher info, with coherence | 0.46 | [0.26, 0.62] | 0/15 |

All five CIs overlap heavily, clustered around median |d| 0.4 to 0.46. SLD
does not separate from any of them, including `classical_pop_fisher_w20`:
QFI decomposes exactly into a classical population term (eigenvalues of rho)
plus a coherence term (eigenbasis rotation), and this channel keeps only the
first. SLD beating it would be the cleanest possible evidence of a real
coherence contribution.

A synthetic sanity check (`qgmrd/classical_baseline.py`, a clean injected
regime shift) shows the coherence term can matter in principle: SLD's peak
response there is 130x its baseline vs. 27x for the classical-population-only
version. So the formalism is not vacuous, that gap just does not appear on
real market data at this sample size.

**Conclusion: SLD should be described as a Bures-rate / Fisher-information
statistic, not as a detector that draws power from quantum coherence.** The
coherence term is real and can matter (shown synthetically), it is not shown
to matter here.

**A bug worth naming.** The first version of `classical_mmd_w20` recomputed
its kernel bandwidth from each pair of adjacent, 95%-overlapping windows.
That inflates the bandwidth exactly when a real shift is present (the local
sample already spans both regimes) and silently suppresses the statistic's
own sensitivity right when it matters. Fixed by fixing the bandwidth once,
globally, from the whole series. Caught by a synthetic sanity check before
the real run, not after.

### Multi-asset panel, Task 3, a null result (`python scripts/multiasset_panel.py`)

Widens the instrument set to **SPY / TLT / UUP / GLD** (DIA dropped, corr > 0.9),
chosen a priori by crisis-type coverage (TLT rate/bond, UUP dollar, GLD haven).
The cross-asset feature is the **absorption ratio** (top-eigenvalue share,
sign-invariant), selected by a pre-registered degeneracy diagnostic on held-out
calm 2014 (k=4 degeneracy refuted: AR/MC correlation ≈ 0.50, median AR 0.464, SPY
loading² 0.20). UUP's 2007-02 inception drops 2007 → **14 crises**. Identical
downstream.

**Widening did not change visibility** (like-for-like on the shared 14, excluding
2007):

| basis | crises visible |
|---|---|
| SPY alone, shared 14 | **2** (2008 GFC, COVID) |
| Union SPY / TLT / UUP / GLD, 14 | **2** (2008 GFC, COVID) |
| Added by TLT / UUP / GLD | **0** |

TLT clears {2008, COVID}, UUP and GLD only {2008}, all global vol spikes SPY
already caught. Near-misses (elevated but sub-floor):

| crisis | TLT \|d\| / floor | SPY \|d\| (same window) |
|---|---|---|
| 2022 Rate Hikes | 1.03 / 1.09 | 0.83 |
| 2011 Euro Crisis | 1.49 / 1.55 | 0.49 |
| 2018 Q4 Selloff | 1.08 / 1.79 | 0.22 |

TLT 2022 misses its floor by 0.06. **What the panel establishes:** the realized-
vol control's persistent floor (~1.1–2.2), not the asset set, caps visibility
*under this control*. Whether the instrument is *also* limiting (would a better-
matched control clear those crises, or is the signal absent?) is **untested**; the
two explanations predict the same output here. Geometric channels: 0 of 14 survive
FDR (ceiling 5 of 14), each clearing ~one out-of-control crisis (QFI log-det →
2019 Repo, spectral → 2022, SLD → 2013 Taper Tantrum) = the ~0.75 chance rate.
This elevated the slow-grind [second control](#validated-on-volatility-events-only)
to the critical next step, as the one instrument able to separate the two
explanations. Built and run, plus a third non-circular variant afterward:
both fail to clear 2022, so the two explanations remain **undistinguished**
and the gap stays open.

### False-alarm-rate evaluation, Task 4, inconclusive by infeasibility (`python scripts/far_eval.py`)

A FAR test asks the deployed-operator question: fix one threshold on past calm
data, freeze it, run forward, count false alarms/yr on calm days vs. detection
during crises. The decision at day *t* uses only data ≤ *t*, the first genuinely
real-time metric here. Pre-registered before the run: one
**deploy-once** detector per channel (scaler/PCA/HMM + threshold frozen on the calm
2005-02-01 → 2007-07-03 block, zero crisis days), τ = 1 alarm/yr, alarms as
upcrossings, chance floor from each channel's circular-shift null, BH-FDR over 7
detectors.

**Inconclusive on detection, not a null.** Two of this repo's metrics returned
verdicts (both null: offline Cohen's *d*, panel count-nulls); FAR **could not be
evaluated at all**: no frozen threshold hitting the target rate transfers to the
deployed period. τ calibrates correctly in-sample (all channels 0.92/yr on the
block, 2 events over 2.18 years) but the block's causal-z **dynamic range is
compressed**, so a block-set threshold sits at the bottom of the forward
distribution:

| channel | block z-max (2.4 calm yr) | forward z-max | forward FAR @ block-τ (target 1.0) |
|---|---|---|---|
| qfi_logdet | 1.13 | 4.96 | **3.37** |
| berry_phase_rate | 1.41 | 8.84 | **2.25** |
| reduced_purity | 1.93 | 2.27 | 0.52 |
| ground_energy E0 | 1.56 | 1.64 | 0.45 |
| sld_qfi_w20 | 8.86 | 4.93 | 0.22 |
| Gaussian HMM | 320.8 | 10.88 | 0.22 |
| **realized vol (CONTROL)** | 3.44 | 7.71 | 0.15 |
| spectral_entropy | 2.95 | 2.89 | 0.07 |

Realized forward FAR spans **0.07–3.37/yr** against 1.0. A 1/yr threshold exists
for each channel but only by reference to the (unavailable) forward distribution.
The pre-registered **drift** caveat dominated (qfi's calm FAR 2.5–3.8/yr across
2005–2019, peaking **5.26/yr in 2020–2024**), so raw forward FAR measures drift and
range-mismatch, not detection; no detection count is read off the run (chatty
channels fire ~20% of *all* days, qfi 22.0% of crisis days vs. 23.9% of calm).

**The clean residue, fixed-line separation** (in-crisis vs. calm exceedance ratio
at the operating points):

| channel | crisis exc. | calm exc. | ratio |
|---|---|---|---|
| **realized vol (CONTROL)** | 7.9% | 0.4% | **17.8×** |
| Gaussian HMM | 6.3% | 0.8% | **8.2×** |
| sld_qfi_w20 | 2.3% | 0.8% | 2.9× (few events) |
| qfi_logdet | 22.0% | 23.9% | **0.92×** |
| berry_phase_rate | 8.9% | 12.9% | **0.69×** |
| E0 / spectral / purity | ≤ 0.4% | ≤ 2.7% | too few forward exceedances to test |

The two geometric channels that fire enough to test (qfi, berry) fire no more,
berry *less*, during crises than calm, while the control fires 18× more and the
HMM 8× more. **Data-not-code:** the control also misses the target rate (0.15/yr
forward) yet separates crisis/calm 18× and flags the three vol events. The
pipeline detects where fixed-line signal exists; the geometric channels don't
supply it. A rolling / periodically-recalibrated-threshold FAR would plausibly fix
the non-transfer but is a **separate experiment needing its own pre-registration**.

## Honest scope

- Operators are **fixed random Hermitian** (paper-endorsed for detection), not
  gradient-learned, zero look-ahead by construction.
- Preprocessing is **causal / past-only** in every result carrying a claim;
  "offline" columns are the labelled leaky comparison. Closes Gap 1, not Gap 2:
  a causal event study, **not a walk-forward** (the word is reserved for the
  unbuilt monthly-refit protocol; the distinction is Hammond's Table 3 vs. Table 5).
- Synthetic-data *d* is not comparable to real-crisis numbers; it only proves the
  pipeline produces separable signal.
- **No channel has been shown to detect above chance, and the tests lack the
  resolution to show it either way** (control tops out at p = 0.039). Read effect
  sizes as un-cleared and untested at useful power, not as absence.

### Methodology caveats

- **The floor is conservative, not clean:** null windows can land on 2008/2011/2018
  (real crises), inflating the floor, the safe direction (harder to clear).
- **Circular-shift p floored at 1/(N_eff+1):** for spectral entropy and QFI
  log-det, τ ≈ 400–442 and N_eff ≈ 12–13, so shift p cannot resolve below ≈ 0.07.
- **The "rest" group is contaminated:** the 15 windows cover **1412 of 5386 trading
  days (26.2%)**, so each crisis is scored against a group ~¼ composed of other
  crises. Every absolute |d| is biased downward. Matters for comparing to
  Hammond; cancels in the null tests (same contaminated rest group). Not corrected
  (deciding which days are "clean" is a forking path).
- **|d| is folded** (discards direction). A wrong-way channel still registers; the
  `dir` column shows each real effect's sign.
- **China 2015 is excluded from the FDR family** (blind HMM control, d = 0.08 /
  0.09); best there is reduced purity, p ≈ 0.14–0.15.
- **Multiple comparisons pre-specified:** primary family = COVID + 2022 (28 tests),
  China held out, raw p and BH q both reported; the control is reported but
  excluded from the family.
- **2007 short-normalization-history artifact:** `causal_zscore` normalizes against
  all prior smoothed values, so the earliest crises get a short, calm denominator:
  the expanding SD at the 2007 midpoint is **0.0019 vs. ~0.009 for every later
  crisis (~5× smaller)**; realized vol scores |d| = **2.20** on the 2007 Quant
  Meltdown, implausibly high on its face. Flagged, not corrected. No general time
  trend (strongest per-channel Spearman is purity ρ = −0.49, p = 0.066 over 15
  points; expanding SD drifts down after 2008), a 2007-specific artifact.
- **Two-null cross-check:** the two structurally different nulls agree to ~0.02 on
  every channel (e.g. COVID purity 0.416 vs. 0.411).

## Reproducibility

Every real-data number comes from a **pinned price snapshot**
(`data/spy_dia_close.csv`, SPY/DIA close, 2005-01-03 → 2026-06-30, `end`
exclusive), not a pinned end date, because `yfinance` returns split/dividend-
adjusted closes so every new distribution retroactively rescales the whole history
(observed drift before pinning ≤ 0.0002 in p-values). `auto_adjust=True` is pinned
explicitly; the package must be installed **editable** (a stale non-editable copy
shadows the repo). A later fetch **will not match** (a fetch in 2027 returns a differently-adjusted 2005–2026 series and a different checksum), and that is the point; the
checksum then tells you whether your fetch reproduces this data.

```bash
uv venv && source .venv/bin/activate       # or: python -m venv .venv
uv pip install -e ".[dev]"                 # numpy/scipy/pandas/sklearn/hmmlearn/pytest
pytest -q                                  # smoke tests
python scripts/channel_correlations.py     # correlation matrix
python scripts/causal_eval.py              # offline vs. causal |d|
python scripts/null_model.py               # per-channel floors + control
python scripts/multi_crisis_panel.py       # 15-crisis panel (headline)
python scripts/diagnostics.py              # positive controls + bug checks
python scripts/far_eval.py                 # FAR (Task 4)
```

`scripts/run_demo.py` runs on synthetic data (inflated d-values, smoke test only):
example output HMM 2.14 / spectral 1.71 / purity 1.49.

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

### Snapshot provenance, multi-asset panel (Task 3)

| | |
|---|---|
| file | `data/multi_asset_close.csv` |
| SHA-256 | `0586e5b64810f87074cef33509d49f90971c74bb7701645193ba20f712958dd8` |
| rows | 4864 |
| columns | `SPY`, `TLT`, `UUP`, `GLD` (close) |
| first / last | 2007-03-01 / 2026-06-30 |
| source | `yfinance`, `auto_adjust=True` (pinned explicitly) |

```bash
shasum -a 256 data/multi_asset_close.csv   # must match the value above
```

The frame begins 2007-03-01 because UUP has no earlier history, why the
multi-asset panel runs 14 crises, not 15.

## Roadmap

- **v1–v5.5, DONE.** Embedding + 7 channels + SLD; offline evaluation; channel
  correlation (SLD decorrelated, not redundant); causal past-fit preprocessing (Gap 1); null-model
  tests (1 of 28 survives FDR vs. ~1.4 expected); 15-crisis panel (median → count,
  a **power** finding, the evaluation tops out at p = 0.039; H1/H2 VOID); harness
  diagnostics (code correct, statistic was the problem; HMM demoted; 2007 artifact
  flagged); multi-asset panel (null); FAR (inconclusive by infeasibility).
- **v5.6–v5.9, DONE.** Slow-grind control investigation: drawdown_252d (second
  control) fails to validate (1/15 panel crises; misses 2022 at |d| = 0.76 vs.
  floor 0.42, τ ≈ 306 vs. vol's 186); a known-ground-truth synthetic confirms
  0/9 channels survive FDR (harness-wide underpower, not a bad statistic); a
  trend-matched null doesn't help; a magnitude sweep quantifies the ceiling at
  ≈5x real 2022 (≈53%, deeper than 2008 GFC); a third, non-circular control
  (`trailing_return_126d`, following reduced purity's short-memory lead)
  narrows this to 3x, still short of 2022's actual ≈14%. See
  ["Validated on volatility events only"](#validated-on-volatility-events-only)
  onward.
- **v5.10, DONE.** Bures/MMD baseline for SLD (Task 3.5), the decisive test
  for whether it is more than a relabeled classical statistic. Four channels,
  each isolating one more layer (raw features, same rho_t, with and without
  coherence); SLD does not separate from any of them. A synthetic check shows
  the coherence term can matter in principle (130x vs. 27x peak response on
  an injected shift), just not on real data at this sample size. See
  ["Classical baselines for SLD"](#classical-baselines-for-sld).
- **v5.11, DONE.** Why trailing return beats drawdown beyond τ: across every
  crisis-length window (62 to 229 days), trailing return's mean value tracks
  the window's own net decline much more tightly than drawdown's does
  (r = 0.79 vs. 0.53 at 229 days). Drawdown responds to the worst point
  reached, not the net outcome, so ordinary choppy dips that fully recover
  still inflate its null floor. See ["A third control that isn't
  circular"](#a-third-control-non-circularly-matching-puritys-threshold).
- **v6, next, power still the binding constraint** (more crises won't help; the
  count saturates at 15): (i) **multi-asset widening**: raises how many crises
  *exist* to detect; (ii) **FAR / expanding-window**, closes Gap 2. Pre-registration
  is deferred to whichever is built.

## Layout

```
qgmrd/
  operators.py    random Hermitian operators {A_k}
  embedding.py    H(x) and ground state via eigh
  observables.py  spectral entropy, reduced purity
  geometry.py     metric (2 paths), QFI, QCRB, Berry plaquette
  channels.py     Berry rate / QFI log-det / E0 time series
  sld.py          SLD, mixed-state QFI (the extension) + classical baselines
                  on the same rho_t (Frobenius, population Fisher info)
  classical_baseline.py  raw-feature classical baselines for SLD (Bures, MMD)
  zscore.py       causal expanding-window z-score (Algorithm 1)
  features.py     returns / vol / momentum / cross-corr
  crises.py       THE crisis window registry (G.10 +/-10 trading days)
  pipeline.py     embed_series + Cohen's d
  baseline.py     Gaussian HMM (baseline) + realized_vol_series / drawdown_series / trailing_return_series (CONTROLS)
  data.py         synthetic generator + pinned snapshot loader
scripts/
  run_demo.py             synthetic smoke run
  channel_correlations.py Task 1, redundancy matrix
  causal_eval.py          Task 2, offline vs. causal past-fit |d|
  null_model.py           per-channel noise floors, single crises + control
  multi_crisis_panel.py   15-crisis panel, count statistic, ceiling
  diagnostics.py          positive controls + bug checks; statistic provenance
  multiasset_panel.py     Task 3, multi-asset visibility panel
  far_eval.py             Task 4, deploy-once false-alarm-rate evaluation
tests/
  test_smoke.py  test_geometry.py  test_sld.py
```

`qgmrd/crises.py` is the single source of truth for crisis windows. Every script
imports it, so the window convention cannot drift between them.
