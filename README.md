# Update Slope Is Bounded by Prior Calibration

Code, data, and analysis for:

> **Update Slope Is Bounded by Prior Calibration: A Structural Bound and Its Contingent Empirical Signature**
> Ben Jenkins and Mihaela Cardei
> *38th IEEE International Conference on Tools with Artificial Intelligence (ICTAI), 2026 (short paper).*

The paper studies the *update slope* of LLM forecasters on four-rung evidence ladders and shows (i) outcome-aware evidence construction inflates measured responsiveness roughly fourfold relative to resolution-blinded ladders, and (ii) raw slope carries a prior-dependent structural ceiling, so cross-model comparisons require conditioning on prior calibration.

## What's here

```
paper.tex / paper.bib      # ICTAI 2026 camera-ready source
figures/data.tex           # numeric macros consumed by paper.tex (fully regenerable)
PREREGISTRATION.md         # pre-registered predictions (cross-pool study)
code/
  config.py                # central configuration (models, pools, env vars)
  fetch_questions.py       # pull resolved binary questions (Manifold)
  fetch_forecastbench.py   # ForecastBench pools
  build_ladder.py          # outcome-aware evidence ladders (sensitivity arm)
  build_ladder_blinded.py  # resolution-blinded ladders (PRIMARY construction)
  elicit.py / elicit_blinded.py    # panel elicitation, K=3 samples per cell
  analyze.py               # slopes, Brier, correlations (outcome-aware run)
  hierarchical_clean.py    # crossed mixed-effects fits (sub-slope spec)
  camera_ready_analysis.py # ALL camera-ready quantities -> data/camera_ready.json
  compare_pools.py         # cross-pool comparison -> data/pool_comparison.json
  compare_blinded_ladders.py + RE_ELICITATION_DECISION_RULE.md  # pre-registered leakage audit
  emit_data_tex.py         # regenerates figures/data.tex from the JSON results
  run.py                   # orchestrator for the elicitation pipeline
data/
  elicitations.jsonl / elicitations_blinded.jsonl   # raw per-sample forecasts (both runs)
  questions_ladder*.json   # both ladder texts
  results.json / camera_ready.json / pool_comparison.json / blinded_comparison.json
  forecastbench/ acled/    # cross-pool inputs and results
```

Note: `code/hierarchical_blinded.py` is retained for provenance but superseded by
`camera_ready_analysis.py`. It regressed on the full L0-to-L3 slope, which shares
the endpoint with Brier@L3 and is not comparable to the paper's L0-to-L2
sub-slope specification; all hierarchical numbers in the camera-ready come from
`camera_ready_analysis.py`.

## Reproducing the paper (no API keys needed)

All raw elicitation data is committed, so every number in the paper regenerates
without querying any model API:

```bash
python -m venv .venv-arm && .venv-arm/bin/pip install -r requirements-exact.txt
.venv-arm/bin/python code/camera_ready_analysis.py   # -> data/camera_ready.json
cd code && ../.venv-arm/bin/python emit_data_tex.py  # -> figures/data.tex
```

The regenerated `figures/data.tex` matches the file used to compile the
camera-ready byte-for-byte. Tested with Python 3.12 on Apple silicon
(`requirements-exact.txt` is the frozen environment; `code/requirements.txt`
gives loose ranges). If a `code/.venv` built on x86_64 is present locally, do
not use it on arm64.

## Re-running the full pipeline (API keys required)

`code/run.py` re-fetches questions, builds ladders, and re-elicits the panel
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY` via environment). Model list and pools
are in `code/config.py`. Elicitation is stochastic (temperature sampling,
K=3), so a fresh run reproduces the paper's findings in distribution, not
byte-for-byte.

## Citation

```bibtex
@inproceedings{jenkins2026slope,
  author    = {Jenkins, Ben and Cardei, Mihaela},
  title     = {Update Slope Is Bounded by Prior Calibration: A Structural Bound and Its Contingent Empirical Signature},
  booktitle = {Proceedings of the 38th IEEE International Conference on Tools with Artificial Intelligence (ICTAI)},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`). Question data derive from public sources (Manifold
Markets, ForecastBench, ACLED) and retain their original terms.
