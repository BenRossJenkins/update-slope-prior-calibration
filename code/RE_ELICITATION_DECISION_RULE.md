# Pre-committed re-elicitation decision rule

**Committed at:** 2026-06-25, before reading the blinded Manifold ladder output.

**Pool considered for re-elicitation:** Manifold only (~$70 API spend). The other
two pools (ForecastBench primary, ACLED) are scope-mapping rather than
load-bearing and are explicitly out of scope for this decision.

**Rationale for binding the rule pre-hoc:** The V-A confirmatory test's
$\beta^{02}$ uses $a_{m,q,2}$, so the leakage threat (named in Sec. VI
Limitations) operates specifically through the L2 ladder rung. Deciding the
re-elicitation cutoff after seeing the comparison invites motivated reasoning
toward whichever answer is cheaper.

## Comparison protocol

For each of the 40 Manifold questions, score the original L2 and the blinded L2
on three leakage-relevant dimensions:

1. **Hindsight-marker phrases.** Presence of phrases that only a post-resolution
   author would write: "as it turned out," "ultimately," "in the event," past
   tense for outcomes that were future at the question's open date, references
   to events post-dating the question's resolution.
2. **Date-of-statement leakage.** Mention of any date that falls between the
   question's open date and resolution date, where the date is the date of the
   outcome itself or a precursor that strongly implied it.
3. **Evidence-selection asymmetry (YES vs NO).** Whether the L2 evidence
   selection is asymmetric toward the realized resolution direction --
   measured by whether a human rater, blinded to the resolution, would predict
   the realized direction at >70% confidence after reading only L2.

For each dimension, count the questions where the original L2 shows that
leakage marker AND the blinded L2 does not.

## Decision thresholds (committed)

Let $d_i \in \{0, 1\}$ for question $i$ and leakage dimension $\in \{1, 2, 3\}$
indicate "leakage present in original, absent in blinded."

* **Re-elicit Manifold** if EITHER of the following holds on a 20-question
  sampled subset:
  * Aggregate leakage incidence $\frac{1}{60}\sum_i \sum_d d_i \ge 0.20$
    (i.e. average of 4+ leakage markers across the 60 question--dimension
    cells), OR
  * Evidence-selection asymmetry alone $\frac{1}{20}\sum_i d_{i,3} \ge 0.30$
    (i.e. 6+ of 20 questions show resolution-direction bias removed by
    blinding).

* **Do not re-elicit; claim minimal leakage** if both aggregates are below
  those thresholds. In that case the paper text adds one sentence:
  "We compared the original ladder to a blinded ladder regenerated without
  showing the builder the resolution outcome (see Appendix X); the blinded
  ladders differ from the originals on $k$ of the 60 question--dimension
  cells we audited, below the 12-cell threshold we pre-committed for
  re-elicitation, so we report the original-ladder results as our main
  finding and the blinded comparison as a robustness check."

## What re-elicitation entails if triggered

* Re-elicit forecasts from all 7 models on the 40 blinded Manifold ladders
  (~3360 calls; ~$70 API spend; ~1 hour of clock time).
* Re-fit V-A hierarchical regression on the blinded $a_2$ values; compare
  $\hat\gamma_1^{\text{blind}}$ against $\hat\gamma_1^{\text{orig}} = -1.78$
  to determine whether the controlled coefficient survives the L2 leakage
  audit.
* If the controlled coefficient on blinded data is within the original CI
  $[-1.95, -1.61]$, claim the V-A result is robust to L2 leakage.
* If it moves outside the original CI but stays significantly negative,
  report both numbers and revise the limitations.
* If it goes to zero or positive, retract the V-A confirmatory claim and
  reposition the paper as theorem-only with cross-pool scope mapping but no
  empirical confirmation on Manifold.
