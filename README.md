# LLMs Under-Update — ICML 2026 AI Forecasting Workshop submission

End-to-end pipeline for the dose-response forecasting study.

## What's here

```
paper.tex                  # main 4-page submission (compiles to paper.pdf)
paper.bib                  # references
icml2026.{sty,bst}         # workshop template (do not modify)
code/
  config.py                # central configuration
  fetch_questions.py       # pull resolved binary Qs from Metaculus
  build_ladder.py          # construct 4-level evidence ladders (uses Claude)
  elicit.py                # query OpenAI + Anthropic panel, K samples per cell
  analyze.py               # compute slopes, Brier, correlations, p-values
  emit_data_tex.py         # write figures/data.tex from results.json
  run.py                   # orchestrator
data/                      # populated by the pipeline (json + jsonl)
figures/data.tex           # macros consumed by paper.tex (overwritten by emit_data_tex)
```

## Quickstart

1. **Environment**

   ```bash
   cd code
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   export OPENAI_API_KEY=sk-...
   export ANTHROPIC_API_KEY=sk-ant-...
   ```

2. **Run the full pipeline**

   ```bash
   python run.py
   ```

   This executes, in order: `fetch_questions` → `build_ladder` →
   `elicit` → `analyze` → `plot`. Each stage writes to `data/`. Re-running is
   incremental: `elicit` skips cells already present in
   `data/elicitations.jsonl`.

3. **Compile the paper**

   ```bash
   cd ..
   pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
   ```

   This produces `paper.pdf`. Replace the `[TBD]` markers in `paper.tex`
   with the numbers printed by `analyze.py` (also stored in
   `data/results.json`).

## Per-stage detail

### `fetch_questions.py`
Pulls resolved binary questions from the Metaculus public API filtered to
`resolve_time >= 2025-01-01`. Configurable via
`TARGET_N_QUESTIONS` and `RESOLUTION_AFTER` in `config.py`.

### `build_ladder.py`
Constructs the four-level evidence ladder per question. Uses Claude
Opus 4.7 at temperature 0; **never** queries any model in the elicitation
panel for ladder construction (avoids the model grading evidence it itself
authored). Costs ~1-2 cents per question.

### `elicit.py`
Queries the model panel concurrently. Default panel:

| API ID | Display | Reasoning |
|---|---|---|
| `gpt-5` | GPT-5 | yes |
| `gpt-5-mini` | GPT-5 mini | yes |
| `gpt-4o-2024-11-20` | GPT-4o | no |
| `o3` | o3 | yes |
| `claude-opus-4-7` | Claude Opus 4.7 | yes |
| `claude-sonnet-4-6` | Claude Sonnet 4.6 | no |
| `claude-haiku-4-5-20251001` | Claude Haiku 4.5 | no |

Edit `MODELS` in `config.py` to drop or add models. Cost scales as
`N_questions × N_models × 4_levels × SAMPLES_PER_CELL`. With defaults
(40 × 7 × 4 × 3 = 3,360 calls) expect on the order of $200–$500 depending
on reasoning model usage.

### `analyze.py`
Computes per-(model, question) update slopes, per-model summaries,
reasoning-vs-non Mann–Whitney U test, slope-vs-Brier Pearson correlation,
and writes `data/results.json`.

### `emit_data_tex.py`
Writes `figures/data.tex` — a single LaTeX file of `\def`-macros that
`paper.tex` imports via `\input{figures/data.tex}`. All four paper
figures are authored inline as pgfplots blocks in `paper.tex`; this
script supplies the data macros (per-level means, per-model curves,
slope/Brier scatter coords, box-plot stats, top-line numbers,
per-model table rows). Re-running this script after updating
`results.json` is enough to refresh every figure and every numerical
claim in the paper.

## How the paper picks up live data

Every numerical claim and figure in `paper.tex` reads from a macro
defined in `figures/data.tex` (e.g. `\pearsonR`, `\mannWhitneyP`,
`\dosesAggCoords`, `\perModelPlots`, `\perModelTableRows`).

After `python run.py`, `emit_data_tex.py` overwrites `figures/data.tex`
with the fresh values, and re-running `pdflatex` is the only thing
needed to refresh figures, in-text numbers, and the per-model table.

The repo ships `figures/data.tex` pre-populated with **plausible draft
numbers** so the paper compiles end-to-end before any experiments are
run. Those drafts are clearly marked at the top of the file.

## Cost-control flags

- Reduce `SAMPLES_PER_CELL` from 3 to 1 for a single-shot dry run.
- Reduce `TARGET_N_QUESTIONS` (in `config.py`) to 10 to validate the
  pipeline end-to-end before paying for the full panel.
- Drop expensive reasoning models from `MODELS` for a non-reasoning-only
  baseline.

## Submission checklist

- [ ] Run `python code/run.py` end-to-end (populates `figures/data.tex`)
- [ ] Recompile: `pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper`
- [ ] Verify `paper.pdf` main body is ≤ 4 pages (excl. references and appendix)
- [ ] Spot-check that no draft `figures/data.tex` values remain
- [ ] Confirm anonymization (no author info, no identifying repo links)
- [ ] Register abstract on OpenReview by May 11, 2026 23:59 UTC
- [ ] Upload PDF to OpenReview by May 13, 2026 23:59 UTC
