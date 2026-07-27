# Pre-registration — Causal False-Alarm-Rate (FAR) evaluation

**Status: pre-registration. Recorded BEFORE any FAR result is computed.**
This document fixes every knob of the FAR evaluation — architecture, calibration
block, threshold rule, metrics, expectation, and caveats — so that none of them
can be chosen after seeing which choice flatters a channel. It is the FAR analog
of the discipline the rest of the repo already applies to feature selection and
to the panel's count statistic (see README). Nothing
below is to be revised in response to results; if a knob genuinely must change,
that is a new pre-registration with the reason recorded, not an edit here.

Naming discipline (causal vs. walk-forward, see README): this protocol is a **causal frozen-threshold
FAR event study**. It is **not** "walk-forward" — there are no monthly refits.
The term walk-forward remains reserved for roadmap item 5.

---

## 1. What FAR measures and why it is worth running

Cohen's *d* and the panel count both ask an **offline, whole-sample** question:
"is the score distribution during the crisis different from the rest of the
series?" That compares the crisis against days that had not happened yet.

FAR asks a **deployed operator's** question. Fix a threshold τ, run the detector
forward day by day, and measure two things:

1. **On calm days, how often does it cry wolf?** — false alarms per year.
2. **When a real crisis hits, does it fire, and how fast?** — detection rate and
   detection delay.

The decision at day *t* uses only data through day *t*. That is why FAR is a
genuinely real-time metric and Cohen's *d* is not, and why FAR **substantially
addresses the core defect of Gap 2** (scoring the crisis window against future
days) — though it is **not** the full walk-forward protocol (Sec. 8 above).

FAR can also see structure Cohen's *d* averages away: a channel with mediocre
*d* can still spike cleanly at the right moments with few alarms between, and
threshold-and-count exposes exactly that separation. This re-measures the
**existing** channels with a better ruler — no new data, no new overfitting
surface **except the threshold, which is the entire risk** (Sec. 4).

---

## 2. Pre-registered expectation (recorded before results)

Every channel was at chance under Cohen's *d* (see README) and under the
panel count (Open Question 4). The base rate is therefore that **FAR also shows
little detection beyond the chance floor** (Sec. 6). FAR is run because it asks a
sharper question the blunt metric could miss — **not because a positive is
expected.**

- If a channel shows **high detection with low FAR** where its *d* was
  unremarkable, that is a genuine finding and gets reported in full.
- If nothing improves, that is a **stronger negative** — two independent rulers
  (Cohen's *d* / count, and FAR) agreeing — and gets reported too.

τ will not be tuned toward the positive. The honesty architecture cuts both ways:
"no signal" is as much an overclaim as "signal" when the honest statement is
"underpowered to tell."

---

## 3. Data, instruments, and the deploy-once architecture

**Data.** The pinned SPY/DIA snapshot `data/spy_dia_close.csv`
(SHA-256 `c8fe62c2…`, 2005-01-03 → 2026-06-30). After `build_features` the
feature frame runs **2005-02-01 → 2026-06-30, 5386 trading days**. The multi-asset
panel (`data/multi_asset_close.csv`) is a cheap follow-on run under the identical
protocol if the SPY/DIA run completes; SPY/DIA is the primary and the minimum.

**Detectors.** The six geometric channels plus the classical baseline, exactly as
`scripts/causal_eval.py::raw_channels` produces them:

| key | role |
|---|---|
| `spectral_entropy` | geometric (reproduction) |
| `reduced_purity` | geometric (reproduction) |
| `ground_energy_E0` | geometric (reproduction) |
| `berry_phase_rate` | geometric (reproduction) |
| `qfi_logdet` | geometric (reproduction) |
| `sld_qfi_w20` | geometric (**extension**) |
| `hmm_high_var_prob` | classical **baseline** |
| `realized_vol_20d` | positive **control** — reported, never in any test family |

**The deploy-once architecture.** FAR is the alarm rate of a **single deployed
detector**, so there is exactly one detector per channel:

1. Fit `StandardScaler` and `PCA` **once**, on the calibration block only
   (Sec. 3.1). Freeze them.
2. Transform the **full** timeline through those frozen objects; compute each raw
   channel; apply the causal expanding-window z-score (`qgmrd/zscore.py`,
   Algorithm 1, past-only) → **one forward score series per channel** spanning
   2005-02-01 → 2026-06-30.
3. Calibrate one threshold τ per channel on the calibration block (Sec. 4),
   freeze it, and run forward.

The HMM baseline is fit on the calibration block's returns only (matching the
scaler/PCA), so the baseline column is causal end-to-end. The control needs no
fitting.

This is deliberately **not** the per-crisis past-fit architecture of the panel.
Fifteen per-crisis thresholds would be fifteen different detectors, and "false
alarms per year of *which* detector" is then incoherent. One frozen detector is
what "deploy" means and is what makes the number comparable to Hammond's
~1 alarm/yr.

### 3.1 The calibration block — exact boundaries, zero crisis days

The block must contain **zero crisis days**, because the target is "1 alarm/year
on **non-crisis** days." The first crisis in the registry is the 2007 Quant
Meltdown (Table G.10 `2007-08 → 2007-09`, extended ±10 trading days →
window start **2007-07-18**, causal cutoff **2007-07-04**). The block is every row
strictly before that cutoff:

| | |
|---|---|
| **calibration block** | **2005-02-01 → 2007-07-03** |
| rows | **609** (≈ 2.42 years) |
| crisis days in block | **0** (verified: first crisis day in the series is 2007-07-18) |
| boundary rule | `index < context("2007 Quant Meltdown").cutoff`, i.e. `< 2007-07-04` |

The block ends **before** the first extended crisis window, with the standard
10-business-day cutoff buffer (2007-07-05 … 2007-07-17) left as an unused
no-man's-land so 20-day rolling features cannot leak the 2007 event backward into
the fit. **Exclusion rule, pre-registered even though it currently removes
nothing:** should the block boundary ever shift, any day falling inside any
G.10 ±10 window is dropped from both the embedding fit and the τ calibration.

After the z-score warm-up (`m = 60`), roughly **549** usable calm days remain for
calibration.

---

## 4. Threshold calibration — the whole game

τ is the only knob, and a τ nudged until crises light up is the +0.415 optimism
trap (see README) in a new costume. Rules:

**Per-channel τ, common target rate.** Channels differ enormously in persistence
(panel τ_autocorr 63 → 483), so a shared `z > 2` cut yields wildly different
FARs. Each channel is instead calibrated to the **same target FAR**, which is the
FAR analog of computing per-channel null floors rather than importing Hammond's
0.53. τ is reported per channel.

**Target FAR = 1 alarm / year**, Hammond's deployment figure, **fixed**. Targets
of **0.5/yr** and **2/yr** are run only as a descriptive sensitivity sweep (an
ROC-like curve of detection vs. FAR); the 1/yr result is the headline and the
sweep is never mined for the target that flatters a channel.

**An alarm is an upcrossing, not a day.** An alarm fires when the score crosses
from below τ to above τ. A detector sitting above τ for 30 consecutive days is
**one** alarm; it must drop back below τ to re-arm before it can fire again.
Event-FAR (alarms/yr) is primary; day-FAR (fraction of days above τ, annualized)
is reported as a secondary descriptive figure.

**Calibration procedure.** By bisection on τ, find the value whose alarm-**event**
rate over the calibration block's calm z-scored days equals 1/yr, then freeze it.

**Coarseness caveat, pre-registered.** A zero-crisis pre-everything block tops out
at ~2.4 years (the first crisis is Aug 2007), so at 1/yr the target is only ~2
expected alarm events — τ is therefore **coarsely determined** (event granularity
≈ 0.4/yr steps). This is inherent to deploy-once and is not a bug; the achieved
in-sample rate is reported (Sec. 7-ii) up to that granularity, the sensitivity
sweep brackets it, and the by-era forward FAR (Sec. 7-iii) shows how it landed.

**Leakage rules (assert in code, not comment):**
- τ and the frozen scaler/PCA/HMM depend **only** on calibration-block rows.
- No crisis day and no forward (≥ 2007-07-04) day enters any fit or the τ search.
- The forward z-score at each day *t* still normalizes against past-only history
  (expanding), so it stays deployable; τ is expressed in those evolving z-units.

---

## 5. Forward evaluation — metrics reported per channel

Run the frozen detector over 2007-07-04 → 2026-06-30. **Calm days** = all days
outside every G.10 ±10 window.

- **Detection rate** — fraction of the **15** crises with ≥ 1 in-window alarm
  (upcrossing inside, or a pre-window alarm still active at window start).
- **Median detection delay** — trading days from window start to the first
  in-window above-τ day, over **detected crises only**. Alarms whose upcrossing
  occurs in the shoulder just before a window and are still active at window start
  are flagged as **early detections** (delay ≤ 0), not counted as false alarms.
- **Realized calm FAR/year** — alarm events on calm days ÷ calm-day-years,
  reported **overall and by 5-year era** (Sec. 7-iii).
- **τ** — the frozen threshold in z-units (exposes the persistence asymmetry
  across channels, exactly as the panel prints τ_autocorr).

Control-first discipline (as in the panel): the realized-vol control must post a
good FAR profile — detecting the 3 vol crises (2007, 2008, COVID) with low delay
and near-target calm FAR. If the **control** has a broken FAR profile, the harness
is suspect and no channel result is read, the same logic that demoted the median.

---

## 6. The chance-detection ceiling — FAR does not escape the power problem

At 1 alarm/yr ≈ 1/252 per day, a crisis window of *L* trading days has a
**chance** detection probability ≈ 1 − e^(−L/252) even for a pure-noise detector.
For COVID (~82 d) that is ~28%; for the long 2008 GFC window (~150 d) ~45%; for a
40-day window ~15%. **Summed over the 15 real window lengths, a pure-noise
detector is expected to "detect" roughly 3–4 of 15 crises for free.**

Therefore detection count is judged **against a pre-registered chance floor**
computed from the real window lengths (reported before the channel numbers), not
against zero — otherwise FAR merely relabels the same non-result. This is the
direct analog of the panel's count ceiling, and it very likely recurs: I expect
the chance floor to sit near the control's achievable detection count, leaving
little headroom. The chance floor is computed by simulation under each channel's
own null (circular-shift of its forward series, masks fixed), so it inherits that
channel's persistence — a highly persistent channel gets a higher chance floor,
the same per-channel fairness the panel enforces.

**The FDR family is fixed now and applied unconditionally.** The correction is
pre-registered here and run **regardless of what the numbers show** — never
"added if a channel approaches the line," because conditioning the machinery on
the result is a soft forking path, and it is exactly when the prior is null and
the stakes feel low that the discipline has to hold. The family is the **7
detectors** (six geometric channels + the HMM baseline) × the crisis-detection
question (does the channel's forward detection count exceed its own circular-shift
chance floor), one raw p-value per detector from the null-count distribution.
Benjamini–Hochberg is applied over all 7 and the q-values reported. The
realized-vol control is **outside** the family (instrument check, not a
hypothesis), as everywhere else in the repo. No confirmatory claim is made for
any channel that does not survive BH; if nothing approaches the line the
correction costs nothing, and if something does, it was corrected from the start
rather than retrofitted.

**Persistence asymmetry — carried over explicitly, as the panel does.** Each
channel's chance floor comes from its own circular-shift null, which inherits its
own autocorrelation, so a **highly persistent channel faces a structurally higher
detection-count floor** (its alarms clump, so a shifted null clears more windows
by chance). This is correct per-channel behaviour — it is what makes each p-value
valid, and the reason no single floor is imported across channels — but it means
**"cleared / did not clear" is NOT apples-to-apples across channels**, and the FAR
count inherits that asymmetry directly. Each channel's τ_autocorr is printed
beside its count so a reader can see who faced the harder bar; a reader must not
assume the floors are comparable across rows. This is the same mechanism the panel
invokes both ways (against SLD, where its low persistence made the bar easy and it
still showed nothing; and in favour of high-τ channels, whose bar is harder).

---

## 7. Verification that the calibration executed correctly

Three checks, all pre-registered:

- **(i) Causality audit.** Assert that the set of row indices entering the
  embedding fit and the τ search is a subset of the calibration block, and that it
  intersects neither any crisis mask nor any forward day. A hard assertion in the
  code.
- **(ii) In-sample calibration-block FAR ≈ target (executes-correctly check).**
  After freezing τ, the realized alarm rate **on the calibration block** must
  match 1/yr **by construction**, up to the integer-event granularity of a
  2.4-year block (Sec. 4). The achieved value is reported per channel; a large
  deviation means the bisection did not converge and is a bug, not a finding.
- **(iii) Out-of-sample forward calm FAR, by era (the real transfer test).**
  Realized calm FAR on forward days, bucketed into fixed 5-year calendar eras.
  **The bin edges are pre-registered here — 2005–2009, 2010–2014, 2015–2019,
  2020–2024, 2025–2026 (partial) — fixed a priori on the decade grid, NOT placed
  after seeing where FAR jumps.** They are calendar boundaries alone; no bin edge
  is chosen with reference to any realized-FAR value, so the drift diagnostic
  cannot be gerrymandered:

  | era | trading days | crisis | calm |
  |---|---|---|---|
  | 2005–2009 | 1239 | 228 | 1011 |
  | 2010–2014 | 1258 | 251 | 1007 |
  | 2015–2019 | 1258 | 394 | 864 |
  | 2020–2024 | 1258 | 539 | 719 |
  | 2025–2026 (partial) | 373 | 0 | 373 |

  The calibration block is carved out of the 2005–2009 bucket, so forward eras
  begin at the 2007 cutoff. If forward FAR ≈ 1/yr, the frozen τ transfers and
  calibration worked. If forward FAR ≫ target, see the deploy-once caveat next.

---

## 8. Pre-registered caveats (not footnotes — read before the numbers)

- **Deploy-once threshold drift — the central limitation.** τ is frozen on an
  early, placid block (2005–mid-2007) and **never updated**. If volatility regimes
  drifted over the following ~20 years, later calm-period alarms may reflect a
  **rising baseline rather than detection.** Realized calm FAR may therefore
  exceed the 1/yr target in later eras for reasons unrelated to signal. This is
  why calm FAR is reported **by 5-year era** (Sec. 7-iii): a reader can see
  whether the frozen threshold drifted. **A rising late-period FAR is not read as
  detection.** This caveat is registered as a primary result of the design, not an
  aside.
- **Coarse calibration.** ~2.4 usable years and ~2 expected events make τ coarsely
  determined (Sec. 4); the sensitivity sweep and by-era FAR exist to expose this.
- **Metric-conditional.** Every FAR number is conditional on the target rate
  (1/yr) and the calibration block (2005-02-01 → 2007-07-03). Reported as such.
- **Delay is defined only for detected crises.**
- **Frozen placid frame.** The scaler/PCA basis is fit on the low-vol 2005–07
  regime and never refits — a more severe version of the panel's PCA-frame-rotation
  caveat. Frame-sensitive channels (Berry, QFI-logdet, purity) are most exposed.
- **No working control on 12 of 15 crises.** The realized-vol control validates
  the harness on vol spikes only (2007, 2008, COVID). A FAR "detection" on any of
  the other 12 crises — including 2022 — is **unvalidated**, carried forward from
  the panel, not certified by FAR. The slow-grind second control (roadmap item 7)
  is what would close this.
- **2007 short-normalization-history artifact.** Early z-scores are inflated
  (expanding SD at the 2007 midpoint is ~5× smaller than later); 2007 detection
  and any near-2007 calm FAR are suspect on their face. Flagged, not corrected.
- **Not walk-forward.** Single frozen fit, no monthly refits (roadmap item 5
  remains distinct and unbuilt).

---

## 9. Reproducibility

Deterministic: one seed (`SEED = 42`, matching the repo), the pinned snapshot, and
a fixed calibration block. The script will print the calibration block bounds, the
per-channel τ, the causality-audit result, the in-sample check, the by-era FAR
table, the chance floor, and the per-channel detection/delay/FAR — so every number
maps to a command a reader can run and match.

    python scripts/far_eval.py   # deterministic; reproduces every number below

---

## 10. Realized outcome (recorded after the run — everything above is the pre-commitment)

**Sections 1–9 are the pre-registration and are left unedited.** This section
records what happened. Ran `scripts/far_eval.py` on the pinned SPY/DIA snapshot;
the causality audit passed (fit rows 0–608 ⊆ block; 0 crisis days and 0 forward
days in the fit set) and the run is deterministic. Numbers below are from that
run, not from recollection (the "live re-run is authoritative" rule).

**Verdict: FAR is INCONCLUSIVE as a detection test on this data — the deploy-once
operating point cannot be set causally. This is not a null.** Same discipline as
the void H1/H2 and the floored E0 shift-p: a test whose instrument was never
validly calibrated delivers no verdict on detection, in either direction.

**Why the operating point can't be set (the infeasibility, precisely).** τ is
calibrated to ~1/yr on the calm 2005–07 block, and the in-sample check passes
(all channels 0.92/yr = 2 events / 2.18 yr, the pre-registered granularity). But
the block's causal-z **dynamic range is compressed** relative to the deployed
period, so a block-set threshold does not transfer:

| channel | block z-max (2.4 calm yr) | forward z-max | forward FAR @ block-τ (target 1.0) |
|---|---|---|---|
| qfi_logdet | 1.13 | 4.96 | **3.37** |
| berry_phase_rate | 1.41 | 8.84 | **2.25** |
| reduced_purity | 1.93 | 2.27 | 0.52 |
| ground_energy_E0 | 1.56 | 1.64 | 0.45 |
| sld_qfi_w20 | 8.86 | 4.93 | 0.22 |
| hmm_high_var_prob | 320.8 | 10.88 | 0.22 |
| realized_vol (CTRL) | 3.44 | 7.71 | 0.15 |
| spectral_entropy | 2.95 | 2.89 | 0.07 |

Realized forward FAR spans **0.07–3.37/yr against the 1.0 target — off in both
directions and uncontrolled.** For qfi/berry the compressed block range puts τ at
the bottom of the forward distribution (over-fires); for the rest τ over-fires
nothing (under-fires). A 1/yr threshold *does* exist for every channel — but only
by reference to the forward distribution, which is not causally available, so it
is not findable under deploy-once. **The pre-registered coarseness (§4) and drift
(§8) caveats compounded into full non-transfer; the drift caveat was the dominant
effect and is vindicated as a primary caveat, not an aside** (qfi's calm FAR sits
at 2.5–3.8/yr across the 2005–2019 eras and peaks at 5.26/yr in 2020–2024, from
the by-era table the script prints). **Raw forward FAR therefore measures
range-mismatch and drift, not detection, and is labelled as such.**

**Consequence for the detection test — no detection count is reported from this
run.** At an ill-set operating point an in-window "hit" only means the detector
happened to be firing, and the chatty channels fire ~20% of *all* days regardless
of crisis (qfi 22.0% of crisis days vs 23.9% of calm), so a raw count reflects
firing rate, not alignment to crises. The count-vs-chance-floor machinery still
runs in the script (and the chance floor correctly discounts the chatty channels),
but its output is **not** read as a detection verdict here — a count from a
detector firing several to many times a year on calm days carries no detection
meaning. The BH-FDR line the script prints is therefore **not** "the null
confirmed"; it is "the calibrated-rate test could not be run at a valid operating
point." Do not quote a FAR detection count as a result.

**The informative residue — fixed-line separation.** What *is* clean is whether a
channel crosses its threshold preferentially during crises. In-crisis vs calm
exceedance ratio at the deploy-once operating points:

| channel | crisis exc. | calm exc. | ratio |
|---|---|---|---|
| realized_vol (CTRL) | 7.9% | 0.4% | **17.8×** |
| hmm_high_var_prob | 6.3% | 0.8% | **8.2×** |
| sld_qfi_w20 | 2.3% | 0.8% | 2.9× (few events) |
| qfi_logdet | 22.0% | 23.9% | **0.92×** |
| berry_phase_rate | 8.9% | 12.9% | **0.69×** |
| E0 / spectral / purity | ≤0.4% | ≤2.7% | too few forward exceedances to test |

The two geometric channels that fire often enough to test (qfi, berry) fire **no
more — berry *less* — during crises than calm**; the control fires 18× more and
the HMM 8× more. So no geometric channel crosses a fixed line preferentially in a
crisis. This **corroborates the offline weak-separation picture through a
different lens** (it is not an independent clean verdict, since the operating
points themselves are ill-set), and it is the takeaway.

**Data-not-code confirmation.** The control also misses the target rate
(0.15/yr forward at its block-set τ), so non-transfer is a property of
protocol-meets-data, not a bug — yet the control still separates crisis/calm 18×
and flags the three vol events (2007, 2008, COVID). The pipeline detects where
fixed-line signal exists; the geometric channels do not supply it.

**No protocol switch.** A rolling / periodically-recalibrated-threshold FAR would
plausibly fix the non-transfer, but changing the protocol after seeing deploy-once
fail is a forking path. It is a **separate experiment requiring its own
pre-registration before it runs**, not a patch to this one. Deploy-once is
recorded here as: **infeasible operating point, inconclusive on detection,
informative on (absence of) fixed-line separation.**
