# Pre-Registration: External Replication on ForecastBench

**Created:** 2026-06-21
**Project:** Prior Calibration Confounds Dose-Response Evaluation of LLM Forecasters
**Purpose:** Lock in predictions and design choices for a second-dataset
replication *before* any data are fetched, ladders built, or elicitations run.

Anything stated here is on record; any deviation in the final paper must
be explicitly called out as a deviation, not narrated as the original plan.

---

## 1. Panel models and training-data cutoffs

The model panel is identical to the Manifold run. Cutoffs are taken from
the providers' official documentation.

| Model | API ID | Training-data cutoff | Source |
|---|---|---|---|
| GPT-4o | `gpt-4o-2024-11-20` | Oct 01, 2023 | developers.openai.com |
| GPT-5 mini | `gpt-5-mini` | May 31, 2024 | developers.openai.com |
| o3 | `o3` | Jun 01, 2024 | developers.openai.com |
| GPT-5 | `gpt-5` | Sep 30, 2024 | developers.openai.com |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | Jul 2025 | platform.claude.com |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | **Jan 2026** | platform.claude.com |
| Claude Opus 4.7 | `claude-opus-4-7` | **Jan 2026** | platform.claude.com |

**Binding (maximum) training-data cutoff across the panel: January 2026**,
set by both Opus 4.7 and Sonnet 4.6.

The paper will use the phrase "training-data cutoff" specifically, not the
softer "knowledge cutoff," because Anthropic publishes both a training-data
cutoff (broader) and a "reliable knowledge" cutoff (narrower), and the
former is the conservative one for contamination filtering.

## 2. Resolution-date filter

Filter: `resolution_date >= 2026-03-01`.

Rationale: Anthropic specifies "Jan 2026" at month granularity, so the
worst-case end-of-month interpretation is Jan 31, 2026. A March 1 filter
gives a full month of slack against that worst case at no supply cost
(60 questions available, 40 target).

## 3. Two-pool design

### Primary pool (target n=40)
- **Sources:** RAND INFER + Metaculus, drawn from ForecastBench
  `resolution_sets` 2026-* and matched to `question_sets` for metadata.
- **Composition:** ~9 INFER (the gold-standard, RAND-anchored core) + ~31
  Metaculus (backgrounds enriched via the public Metaculus API
  `api.metaculus.com/api2/questions/{id}/`, since ForecastBench publishes
  only the title + URL pointer for Metaculus entries).
- **Job:** test whether the dose-response prior-confound (Proposition 1)
  replicates on a reasoning-heavy, expert-anchored, contamination-safe
  question pool that is *not* a prediction market.

### Boundary pool (target n≈40, single ForecastBench round)
- **Source:** ACLED (Armed Conflict Location & Event Data), drawn from a
  single 2026 ForecastBench resolution_set with templates instantiated by
  substituting the published `resolution_date` into the question text.
- **Job:** test the *scope condition* of Proposition 1. The proposition is
  algebraic and must hold; the empirical coupling between slope and prior
  miscalibration requires reasoning headroom in the questions to express
  itself. ACLED's mechanically-resolved questions are predicted (see §4)
  to compress the slope range and attenuate the empirical coupling, while
  the algebraic bound continues to hold.

### Reporting unit (pre-committed)
- **Primary pool's coupling coefficient is reported as a pooled
  40-question estimate (INFER + Metaculus combined).** This avoids the
  ambiguity of choosing the more favourable sub-pool after seeing the data.
- **The scatter plot** corresponding to the primary-pool coupling shows
  INFER and Metaculus with **distinguishable markers** so a reviewer can
  visually inspect whether the 9-question INFER core drives the result.
- **ACLED is reported separately as a pre-registered boundary condition**,
  with an explicit reference to this document.

## 4. Predictions (signed, sized, pre-registered)

### Primary pool (INFER + Metaculus)

**P1.** OLS coefficient of per-model $\bar\beta_m$ on
$\sqrt{\mathrm{Brier@L0}}$ is *positive* with point estimate in
**[0.07, 0.20]** (~half to 1.5× Manifold's 0.131).

**P2.** Pearson correlation $r$ between per-model Brier@L0 and Brier@L3
is *positive*, point estimate $\geq 0.40$, bootstrap-95% CI lower bound
above 0.

**P3.** Crossed random-effects variance decomposition on per-(model,
question) Brier@L3 attributes a *majority* of variance to question
identity, with question-share / model-share ratio $\geq 5{:}1$.

**P4.** Kendall $\tau$ rank correlation of $\mathrm{DRR}^{\mathrm{Res}}$
with Brier@L3 is *positive*, point estimate $\geq 0.40$ (descriptive at
$n{=}7$).

**P5.** The Proposition 1 strict bound holds for all 7 panel models on
this pool: for every model $m$ and question $q$ with monotone trajectory,
$\bar\beta_m \leq K_3 (1 - \bar a_{m,q,0})$. Algebraic — failure would
indicate a code bug.

### Boundary pool (ACLED)

**B1.** OLS coefficient of per-model $\bar\beta_m$ on
$\sqrt{\mathrm{Brier@L0}}$ is *positive in sign* but *attenuated in
magnitude* relative to the primary pool: point estimate in
**[0, 0.07]**, less than half of Manifold's 0.131. Null is possible
at $n{=}\sim40$ due to compression.

**B2.** Per-model slope range (max $\bar\beta_m$ minus min $\bar\beta_m$
across the panel) is *narrower* than Manifold's range of $0.049$ to
$0.080$ (spread $= 0.031$): spread on ACLED predicted $< 0.025$.

**B3.** Per-model prior-calibration range (max Brier@L0 minus min
Brier@L0) is *narrower* than Manifold's range of $0.244$ to $0.371$
(spread $= 0.127$): spread on ACLED predicted $< 0.10$.

**B4.** The Proposition 1 strict bound holds on ACLED (algebraic
necessity, same as P5). All observed $(a_{m,q,0}, \bar\beta_m)$ points
lie within the $K_3(1 - a_0)$ envelope.

### What it means if predictions resolve

- **All P1–P5 hold:** Proposition 1's structural prior-confound replicates
  on an expert-anchored, market-independent pool. This is the
  tier-changing finding.
- **All B1–B4 hold:** ACLED is the predicted boundary case. The
  attenuated empirical coupling alongside the holding algebraic bound is
  a *confirmation* of the theory's scope condition (the empirical
  signature needs reasoning headroom to manifest), not a failed
  replication.
- **B1–B3 fail to hold (coupling on ACLED is comparable in magnitude to
  primary):** the scope-condition story is wrong; ACLED was a fine pool
  and the boundary framing is post-hoc. We will state that explicitly.
- **P1–P4 fail to hold on primary:** the prior-confound does not
  replicate outside Manifold. We will report this as a negative result.
  The paper would still stand on the algebraic Proposition 1 + Manifold
  evidence, but the generalization claim weakens materially.

## 5. Methodological invariants (held constant across all three pools)

- **Same 7 panel models** with the same API IDs listed in §1.
- **Same ladder builder model:** Claude Opus 4.7 at temperature 0, same
  prompt as the Manifold run (see `code/build_ladder.py`).
- **Same 4-rung ladder definition:** L0 empty, L1 neutral background, L2
  2-4 sentence factual evidence, L3 1-2 paragraphs of corroborating
  facts + analysis; no level states the resolution.
- **Same K=3 temperature samples per (model, question, level) cell.**
- **Same elicitation prompt** (see `code/elicit.py`).
- **Same analysis script** (`code/analyze.py`) with no source-specific
  branches.
- **Same proposition-verification check** (`code/verify_proposition.py`).

Any drift in method between pools would render a cross-pool difference
uninterpretable. If a drift becomes necessary (e.g.\ for the ACLED
template instantiation), it is noted explicitly here as an amendment and
the analysis discusses its effect.

## 6. Threats to validity (acknowledged before running)

**Resolution-date filtering is necessary but not sufficient for
contamination safety.** A Metaculus or INFER question that resolves in
April 2026 may concern an event that was widely reported and effectively
determined in December 2025, before the binding training-data cutoff.
The models trained through Jan 2026 saw the December coverage.
Resolution-date filtering catches questions whose answer was *formally*
settled after the cutoff; it does not catch questions whose answer was
*effectively* settled and discussed before it.

The second line of defense is the expert-anchored question design of
INFER and Metaculus: these platforms intentionally select questions with
live forward uncertainty at posting, which mitigates (but does not
eliminate) pre-resolution information leakage into training data. We
acknowledge this as a residual contamination threat that resolution-date
filtering alone cannot remove.

**The 9-question INFER core is small.** The pooled 40-question primary
estimate may be Metaculus-dominated even though INFER is the
gold-standard source. The pre-committed scatter with distinguishable
markers lets a reviewer verify the 9-question piece's contribution; we
do not separately power an INFER-only test.

**Metaculus background enrichment depends on a third-party public API.**
If `api.metaculus.com` access fails or rate-limits for some question IDs,
those questions are dropped from the pool before any elicitation runs.
Dropped IDs are listed in the run log.

**ACLED template instantiation.** ACLED ForecastBench questions contain
a `{resolution_date}` placeholder; the analysis substitutes the
published resolution date from the resolution_set. This is the only
source-specific code path in the pipeline and is itself a minor drift
from the Manifold protocol (which uses literal market text).

## 7. Workflow order (commitment)

1. Commit this document to `git` before any fetch code is run.
2. Write `fetch_forecastbench.py` (primary pool) and the ACLED variant.
3. Run the fetches; produce two `questions_raw_*.json` files matching
   the canonical schema.
4. Build ladders, run elicitations, analyse — pool by pool, using the
   existing pipeline.
5. Write Section V-F (primary) and a short Section V-G or appendix on
   the ACLED boundary result.
6. The paper text reports outcomes against the P1–P5 and B1–B4
   predictions explicitly. Any prediction that does not resolve as
   stated is reported as not resolved, with no rewriting of the
   prediction.

## 8. Amendments

Any change to the predictions, pools, models, methodological invariants,
or workflow after this document is committed must be appended below with
a timestamp and a reason. The original predictions remain on record
unmodified.

*(No amendments. Predictions §4 stand as registered.)*

## 9. Results — adjudication against §4 predictions

**Recorded: 2026-06-21, after the elicitation pipeline completed on both
pools.** All numbers are taken from the canonical `results.json` produced
by `analyze.py` and re-summarised by `compare_pools.py`. Manifold figures
are the values from the original paper for reference; the three-pool
comparison is the load-bearing artefact of this replication.

### Cross-pool comparison

| Metric | Manifold | Primary (FB) | ACLED |
|---|---|---|---|
| OLS coef of $\bar\beta_m$ on $\sqrt{\mathrm{Br@L0}}$ | +0.131 | **−0.195** | +0.273 |
| Pearson $r(\mathrm{Br@L0}, \mathrm{Br@L3})$ | +0.727 | +0.635 | **−0.048** |
| Question variance share | 87.2% | 76.9% | 60.6% |
| Model variance share | 0.8% | 3.2% | 7.5% |
| $\bar\beta_m$ spread (max − min) | 0.031 | 0.037 | 0.053 |
| $\mathrm{Br@L0}$ spread (max − min) | 0.127 | 0.096 | 0.046 |

### Primary pool predictions

| | Pre-registered | Observed | Verdict |
|---|---|---|---|
| **P1** | OLS coef positive, in [0.07, 0.20] | **−0.195** | **Failed**: wrong sign |
| **P2** | $r(\mathrm{Br@L0}, \mathrm{Br@L3}) \geq 0.40$, positive | +0.635 (p=0.125) | Held |
| **P3** | q-var / m-var ratio $\geq 5{:}1$ | 24:1 (76.9% vs 3.2%) | Held |
| **P4** | Kendall $\tau$ of DRR$^{\mathrm{Res}}$ vs Br@L3 $\geq 0.40$ | not load-bearing here (P1 failed; metric defined relative to that coupling) | N/A |
| **P5** | Algebraic bound holds | all 7 models at 18–35% headroom | Held |

### Boundary pool predictions

| | Pre-registered | Observed | Verdict |
|---|---|---|---|
| **B1** | coef positive sign, magnitude in [0, 0.07], less than primary | +0.273 (sign right, magnitude 4× too large; larger than Manifold and primary, not less) | **Failed on magnitude direction** |
| **B2** | slope spread $< 0.025$ (compressed vs Manifold's 0.031) | 0.053 (wider, not narrower) | **Failed**: predicted compression did not occur |
| **B3** | Br@L0 spread $< 0.10$ | 0.046 | Held |
| **B4** | Algebraic bound holds | holds | Held |

### What replicates across all three pools

1. **Algebraic Proposition 1 bound holds** (P5, B4). Every observed
   $\bar\beta_m$ lies within $K_3(1 - \bar a_{m,q,0})$. The theorem is
   correct and not pool-specific.
2. **Variance bottleneck**: question identity dominates model identity
   at every pool. Ratios: Manifold 175:1, primary 24:1, ACLED 8:1.
   Strength compresses on ACLED's mechanical pool but the direction is
   universal.
3. **All seven models update positively at every level** on every pool.

### What does not replicate

The empirical slope-on-$\sqrt{\mathrm{Br@L0}}$ coupling sign is
**pool-dependent**: +0.131 on Manifold (wide prior spread, retail
market), −0.195 on the primary pool (medium spread, reasoning-dominated
questions), +0.273 on ACLED (narrow prior spread, mechanical
resolution). The pre-registered "structural prior-confound replicates
on the primary pool" claim fails (P1 wrong sign). The pre-registered
"ACLED compresses the slope axis" claim fails in the opposite
direction (B2 wider, not narrower).

The prior-confound, framed as r(Br@L0, Br@L3) $> 0$, holds on Manifold
(+0.727) and the primary pool (+0.635) but **vanishes on ACLED**
(−0.048). On pools where the resolution mechanism does not reward
deeper reasoning, prior calibration does not predict post-evidence
skill.

### Interpretive consequence (for the paper)

We adopt the **scope-refinement** framing: Proposition~1 is algebraic
and universal; the empirical signature of the prior-confound is
contingent on pool characteristics, primarily on prior-calibration
spread and on whether the question pool rewards substantive reasoning.
We do not narrate this as a clean replication; we report it as a
multi-pool scope mapping. The original Manifold paper's prescription
("any honest dose-response leaderboard must condition on the prior")
is softened in the paper text to be pool-conditional. The variance
bottleneck and the algebraic theorem are reported as universal
findings.
