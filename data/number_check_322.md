# Number check — paper.tex (camera-ready 322), 2026-09-06

Paper: `/Users/benjenkins/Downloads/IEEE-conference-template-062824/paper.tex`
Macros: `/Users/benjenkins/Downloads/IEEE-conference-template-062824/figures/data.tex`
Ground truth: `camera_ready.json`, `pool_comparison.json`, `results.json` (+ per-pool `results.json`, `blinded_comparison.json`)

## 1. Hardcoded numbers in paper.tex

1. **K3 = 0.4 — OK.** Formula check: L=3, k=2, 6·2·2/(3·4·5)=0.4. LP verification confirms the bound is tight at eps=0 (max slope exactly K3(1−a0)).
2. **C3 = 0.2 — OK (as an upper bound).** LP over trajectories with a_{l+1} ≥ a_l − eps confirms max slope ≤ 0.4(1−a0) + 0.2·eps at all tested (a0, eps); the eps term is valid (not tight at tested points, but the claim is only an upper bound).
3. **"+0.166" aggregate movement (orig L3−L0) — OK.** 0.6870119 − 0.5212332 = 0.16578.
4. **"+0.047" aggregate movement (blind L3−L0) — OK.** 0.5659762 − 0.5190714 = 0.04690.
5. **"roughly 72% of the apparent evidence effect disappears" — OK.** 1 − 0.04690/0.16578 = 71.7%.
6. **"roughly fourfold" utilization inflation (abstract, contributions, discussion) — OK.** u_mean ratio 0.33810/0.08452 = 4.00; mean-slope ratio 3.91.
7. **Figure caption "roughly four times the aggregate movement" — MINOR IMPRECISION.** The *aggregate movement* ratio is 0.16578/0.04690 = **3.53×**, not ~4×. The ~4× figure is the *utilization* (or per-model slope) ratio. If precision matters, change the caption in `paper.tex` line 162 from "roughly four times the aggregate movement" to "roughly 3.5 times the aggregate movement" (or "roughly four times the panel utilization"). Everywhere else "fourfold" is tied to utilization/slopes and is correct.
8. **"1.67% aggregate incidence" audit — OK.** `blinded_comparison.json` decision block: `aggregate_leakage_rate = 0.0167` (2 markers removed of 120 question–dimension cells), below the 0.20 threshold, decision DO_NOT_RE_ELICIT.
9. **"maximum violation 0.029" — OK.** Full-precision per-model curves (`results.json` per_model_curves): Sonnet 4.6 L0→L1 dip = 0.52262 − 0.49386 = 0.0288 → 0.029. (Other dips: Haiku 0.0242, Opus 0.0040.)
10. **"four of seven models" monotone — OK.** Monotone: GPT-5 mini, GPT-4o, GPT-5, o3. Non-monotone: Opus 4.7, Sonnet 4.6, Haiku 4.5.
11. **"roughly threefold increase" (both datasets) — OK.** Orig: −1.7814/−0.5312 = 3.35×; blind: −1.6553/−0.5059 = 3.27×. Both "roughly threefold."
12. **"utilization $0.27$--$0.41$" (Sec. mech) — OK, with a rounding note.** uMin = 0.2708 → 0.27; uMax = 0.4122 → 0.41 at 2 dp (correct rounding). Note the adjacent macro-based range prints "[0.271, 0.412]" (\uMin/\uMax at 3 dp); "0.41" vs "0.412" is a display-precision difference only, not an error. Acceptable; change to "0.27–0.412" or "$[\uMin,\uMax]$" only if you want strict consistency.
13. **"sd $\uSd$, CV $0.16$" — OK.** 0.055291/0.338101 = 0.1635 → 0.16 (macro uSd 0.055 matches).
14. **"CV $0.47$" (blind) — OK.** u_cv_blind = 0.4716.
15. **"roughly $62\%$" Murphy claim — CANNOT VERIFY FROM LOCAL DATA (external citation).** It is a claim about Murphy (2026)'s own ForecastBench analysis, not about this paper's data. It matches the identical claim in `backup_322_submitted.tex` (line 242), so it is at least internally consistent across versions. Flagged per instructions; verify against the cited paper if needed.
16. **"280 model–question cells" — OK.** 7 models × 40 questions; `camera_ready.json` cells.total = 280.
17. **"six each for Sonnet 4.6 and Haiku 4.5" — OK.** incomplete_cells: exactly 6 Haiku 4.5 + 6 Sonnet 4.6 = 12 (\nCellsIncomplete = 12). "Almost all at the no-context L0 or L1 rungs" — in fact *all* 12 missing levels are L0 or L1 (one cell also has only 1 valid L2 sample, which still counts as valid), so the statement is conservative and correct.
18. **"268 cells common to both runs" (hardcoded twice) — OK.** cells.common = 268 = orig_complete.
19. **"Three of nine pre-registered predictions failed" — OK.** `backup_322_submitted.tex` prereg table (tab:prereg, lines 977–1020): nine predictions; six hold; three fail (P1 sign-flip on primary, B1 magnitude on ACLED, B2 slope-spread on ACLED), all in the coupling family — matching the paper's "all in this coupling family."
20. **Panel composition — OK.** Data reasoning flags: gpt-5, o3, claude-opus-4-7, gpt-5-mini = reasoning (4); gpt-4o-2024-11-20, claude-sonnet-4-6, claude-haiku-4-5-20251001 = non-reasoning (3). Matches "four reasoning (GPT-5, o3, Claude Opus 4.7, GPT-5 mini) and three non-reasoning (GPT-4o, Claude Sonnet 4.6, Claude Haiku 4.5)."
21. **"$n_q{=}40$ per pool" (cross-pool table caption) — OK.** All 7 models have n_questions = 40 in forecastbench and acled results.json.
22. **"$K=3$ probability forecasts" — OK.** valid_samples cap at 3 throughout.

## 2. Macro resolution and values

23. **All macros resolve — OK.** Every macro used in paper.tex (77 data macros) is \def'd in figures/data.tex; only \cref and \frac are external (cleveref/amsmath). `paper.log` has zero "Undefined control sequence" errors.
24. **Spot-checked macro values vs JSON — all OK:**
    - crOrigNaiveBeta −0.531 (−0.53124), CI [−0.760, −0.302] ([−0.76032, −0.30216])
    - crOrigCtrlBeta −1.781 (−1.78139), CI [−1.942, −1.620]
    - crBlindNaiveBeta −0.506 (−0.50592), CI [−0.785, −0.227]
    - crBlindCtrlBeta −1.655 (−1.65532), CI [−1.801, −1.510]
    - crBlindCommonBeta −1.654 (−1.65371), CI [−1.820, −1.487], n=268
    - crOrigSlopeTwelveBeta −0.638, CI [−0.681, −0.595]; crBlindSlopeTwelveBeta −0.601, CI [−0.694, −0.509]
    - uMean 0.338 (0.33810); uSd 0.055; uMin 0.271; uMax 0.412; KthreeUMean 0.135 (0.13524)
    - uMeanBlind 0.085 (0.08452); uMinBlind 0.018; uMaxBlind 0.126; KthreeUMeanBlind 0.034
    - corrUH +0.27 (0.26709); corrUHBlind +0.87 (0.87251)
    - couplingSqrtBrier +0.131 (0.13150); couplingH +0.227 (0.22721); couplingSqrtBrierBlind +0.180; couplingHBlind +0.259
    - corrChangeBaseline −0.55 (−0.54900); corrChangeOldham −0.02 (−0.02300)
    - blomqvistObs −0.582 (−0.58170); blomqvistCorr −0.564 (−0.56400); sigmaRatioPct 4.1 (4.06%)
    - cellEpsMedian 0.023 (0.02333); cellEpsPninety 0.192; fracMonoCells 34 (33.6%)
    - nCellsOrig 268; nCellsBlind 280; nNullRecords 181; nCellsIncomplete 12
    - aggOrigLZero 0.521 / aggOrigLThree 0.687 / aggBlindLZero 0.519 / aggBlindLThree 0.566; aggCoordsOrig/Blind match aggregate_aligned to 4 dp
    - rankTauHake +0.333, p 0.381 (recomputed: tau=0.333, p=0.381); rankTauMurphyRes +0.619, p 0.069 (recomputed: 0.619/0.069); rankTauBrierlThreeVsslope 0.143, p 0.773 (recomputed exactly)
    - brierLThreeGptFourOFull 0.140553 (0.14055278); brierLThreeHaikuFull 0.140631 (0.14063056)
    - Cross-pool macros vs pool_comparison.json: manifold +0.131/+0.73/87%/0.8%; ForecastBench −0.195/+0.64/77%/3.2%; ACLED +0.273/−0.05/61%/7.5% — all match (−0.19544, 0.63515, 76.94, 3.19; 0.27260, −0.04829, 60.60, 7.47).

## 3. Internal consistency

25. **Abstract vs body — OK.** Abstract numbers are all macro-driven (uMean, uMeanBlind, cr*Beta) or match body hardcodes (K3=0.4, "roughly fourfold", "panel size seven"); no discrepancies.
26. **"utilization 0.27–0.41" vs [\uMin,\uMax]=[0.271,0.412] — OK (flagged).** See item 12: 0.412→0.41 is correct 2-dp rounding; the two spots display different precision. Cosmetic only.
27. **CV arithmetic — OK.** uSd/uMean = 0.055291/0.338101 = 0.1635 → "CV 0.16"; blind CV 0.4716 → "0.47".
28. **\uTableRowsBoth vs u_table / u_table_blind — OK.** All 7 rows match: a0, slope (4 dp), u (3 dp) for outcome-aware; slope, u for blind (e.g., Sonnet 0.523/0.0787/0.412 and 0.0218/0.108; Haiku 0.577/0.0489/0.289 and 0.0030/0.018). **One presentational note:** the table's single $\bar a_{m,0}$ column shows the *outcome-aware* a0, while the blind u_m values are computed from the *blind* a0 (which differs, e.g. Sonnet 0.523 vs 0.494, GPT-5 mini 0.502 vs 0.517). A reader recomputing blind u from the displayed a0 gets 0.114 instead of 0.108 for Sonnet. Not a data error (values match JSON), but consider a caption note if space allows.
29. **Text "At L0 the two runs coincide (0.521 vs 0.519)" — OK** at the aggregate level (per-model L0 differs by up to 0.029 for Sonnet, but the claim is about the aggregate and is fair).

## 4. Bound verification ("holds at every per-model observation")

30. **OK — verified computationally on all available pools.** mean slope ≤ K3·(1−mean a0), i.e. u ≤ 1, for every model:
    - Manifold outcome-aware: max u = 0.412 (all 7 ≤ 1)
    - Manifold blinded: max u = 0.126 (all 7 ≤ 1)
    - ForecastBench (`data/forecastbench/results.json`): u ∈ [0.225, 0.416], all ≤ 1
    - ACLED (`data/acled/results.json`): u ∈ [0.205, 0.524], all ≤ 1
    - (fb_pilot, not in the paper's table: u ∈ [0.076, 0.571], also all ≤ 1)

## 5. Items not locatable in data

31. **"62% Murphy" (item 15)** — external-citation claim; not verifiable from local files, consistent with the submitted version.
32. Everything else in the checklist was located and verified against the ground-truth files.

## Summary of action items

- **Only candidate edit:** figure caption, paper.tex line 162 — "roughly four times the aggregate movement" is 3.53× for aggregate movement; either change to "roughly 3.5 times" or re-anchor to utilization. All other checked numbers are correct.
- Optional cosmetic: "0.27–0.41" vs "[0.271, 0.412]" precision (item 12/26) and the shared-a0-column note in tab:utilization (item 28).
