"""Compute the dose-response statistics from elicitations.

For each (question, model) pair we have up to len(LEVELS) * SAMPLES_PER_CELL
probability forecasts. We compute:

  * Mean forecast at each level (averaged across samples).
  * Truth-aligned forecast: p_aligned = p if YES else (1 - p).
  * Update slope: linear regression coefficient of p_aligned on level index
    (0..3), per question. This is the per-question dose-response slope.
  * Per-model summary: mean slope (across questions), median slope, fraction
    of questions where slope > 0, Brier at each level, and Brier reduction
    L0->L3.
  * Reasoning vs non-reasoning subgroup comparison (paired t-test on per-
    question slope differences for matched models, plus an overall Mann-
    Whitney U on per-(model, question) slopes).
  * Correlation between per-model mean slope and per-model Brier at L3.

Writes data/results.json.
"""
from __future__ import annotations
import json
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
from scipy import stats

from config import ELICITATIONS, RESULTS, QUESTIONS_LADDER, LEVELS, MODELS

LEVEL_INDEX = {l: i for i, l in enumerate(LEVELS)}


def _load() -> List[Dict[str, Any]]:
    rows = []
    with ELICITATIONS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _truth_lookup() -> Dict[str, str]:
    qs = json.loads(QUESTIONS_LADDER.read_text())
    return {q["id"]: q["binary_resolution"] for q in qs}


def main():
    rows = _load()
    rows = [r for r in rows if r.get("prob") is not None]
    truth = _truth_lookup()

    # cell_means[(qid, model)][level] = mean prob
    cell_probs: Dict[tuple, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cell_probs[(r["question_id"], r["model_api_id"])][r["level"]].append(r["prob"])

    # Per-question per-model slope and Brier-by-level.
    qm_slopes: Dict[str, Dict[str, float]] = defaultdict(dict)  # model -> qid -> slope
    qm_briers: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))  # model -> qid -> level -> brier
    qm_aligned: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    model_meta: Dict[str, Dict[str, Any]] = {}
    for (qid, model), levels in cell_probs.items():
        if qid not in truth:
            continue
        y = 1.0 if truth[qid] == "YES" else 0.0
        xs, ys = [], []
        for lvl, plist in levels.items():
            if not plist:
                continue
            mean_p = float(np.mean(plist))
            mean_aligned = mean_p if y == 1.0 else 1.0 - mean_p
            xs.append(LEVEL_INDEX[lvl])
            ys.append(mean_aligned)
            qm_briers[model][qid][lvl] = (mean_p - y) ** 2
            qm_aligned[model][qid][lvl] = mean_aligned
        if len(xs) >= 2:
            slope, _ = np.polyfit(xs, ys, 1)
            qm_slopes[model][qid] = float(slope)

    # Build model -> meta.
    for provider, api_id, display, family, reasoning in MODELS:
        model_meta[api_id] = {
            "display": display, "family": family, "reasoning": reasoning,
        }

    # Per-model summary.
    summary: Dict[str, Any] = {"models": {}}
    rationality = []
    for model, qslopes in qm_slopes.items():
        if not qslopes:
            continue
        slopes = np.array(list(qslopes.values()))
        briers_l3 = [qm_briers[model][q].get("L3") for q in qslopes if "L3" in qm_briers[model][q]]
        briers_l0 = [qm_briers[model][q].get("L0") for q in qslopes if "L0" in qm_briers[model][q]]
        meta = model_meta.get(model, {"display": model, "family": "?", "reasoning": False})
        summary["models"][model] = {
            **meta,
            "n_questions": int(len(slopes)),
            "mean_slope": float(np.mean(slopes)),
            "median_slope": float(np.median(slopes)),
            "std_slope": float(np.std(slopes, ddof=1)) if len(slopes) > 1 else 0.0,
            "frac_positive_slope": float(np.mean(slopes > 0)),
            "brier_L0": float(np.mean(briers_l0)) if briers_l0 else None,
            "brier_L3": float(np.mean(briers_l3)) if briers_l3 else None,
            "brier_reduction": (
                float(np.mean(briers_l0) - np.mean(briers_l3))
                if briers_l0 and briers_l3 else None
            ),
        }
        rationality.append((meta["reasoning"], np.mean(slopes), np.mean(briers_l3) if briers_l3 else np.nan))

    # Reasoning vs non-reasoning: split per-(model, question) slopes.
    reasoning_slopes, non_reasoning_slopes = [], []
    for model, qslopes in qm_slopes.items():
        meta = model_meta.get(model, {})
        target = reasoning_slopes if meta.get("reasoning") else non_reasoning_slopes
        target.extend(qslopes.values())
    if reasoning_slopes and non_reasoning_slopes:
        u, p = stats.mannwhitneyu(reasoning_slopes, non_reasoning_slopes, alternative="two-sided")
        summary["reasoning_vs_non"] = {
            "n_reasoning": len(reasoning_slopes),
            "n_non_reasoning": len(non_reasoning_slopes),
            "mean_slope_reasoning": float(np.mean(reasoning_slopes)),
            "mean_slope_non_reasoning": float(np.mean(non_reasoning_slopes)),
            "mannwhitney_u": float(u),
            "p_value": float(p),
        }

    # Correlation: per-model mean_slope vs brier_L3.
    pairs = [(s["mean_slope"], s["brier_L3"]) for s in summary["models"].values()
             if s["brier_L3"] is not None]
    if len(pairs) >= 3:
        x, y = zip(*pairs)
        r, p = stats.pearsonr(x, y)
        summary["slope_brier_correlation"] = {"pearson_r": float(r), "p_value": float(p), "n_models": len(pairs)}

    # Brier@L0 vs Brier@L3 across models (prior calibration story).
    triples = [(s["mean_slope"], s["brier_L0"], s["brier_L3"])
               for s in summary["models"].values()
               if s["brier_L0"] is not None and s["brier_L3"] is not None]
    if len(triples) >= 3:
        slopes_, l0s, l3s = (list(z) for z in zip(*triples))
        import numpy as _np
        slopes_arr, l0_arr, l3_arr = _np.array(slopes_), _np.array(l0s), _np.array(l3s)
        r_l0l3, p_l0l3 = stats.pearsonr(l0_arr, l3_arr)
        summary["brier_l0_l3_correlation"] = {
            "pearson_r": float(r_l0l3), "p_value": float(p_l0l3), "n_models": len(triples),
        }
        # Partial r(slope, brier_L3 | brier_L0).
        def _pearson(a, b): return stats.pearsonr(a, b)[0]
        r_sl_l3 = _pearson(slopes_arr, l3_arr)
        r_sl_l0 = _pearson(slopes_arr, l0_arr)
        r_l0_l3 = _pearson(l0_arr, l3_arr)
        denom = ((1 - r_sl_l0 ** 2) * (1 - r_l0_l3 ** 2)) ** 0.5
        partial = (r_sl_l3 - r_sl_l0 * r_l0_l3) / denom if denom else float("nan")
        summary["slope_brier_partial_r"] = {
            "partial_r": float(partial), "controls_for": "brier_L0", "n_models": len(triples),
        }

    # Variance decomposition of per-(model, question) Brier@L3.
    qm_b_l3 = []
    for model, qbriers in qm_briers.items():
        for qid, lvls in qbriers.items():
            if "L3" in lvls:
                qm_b_l3.append((model, qid, lvls["L3"]))
    if qm_b_l3:
        import numpy as _np
        all_b = _np.array([b for _, _, b in qm_b_l3])
        # Group by question, by model.
        bq, bm = defaultdict(list), defaultdict(list)
        for m, q, b in qm_b_l3:
            bq[q].append(b)
            bm[m].append(b)
        qmeans = _np.array([_np.mean(v) for v in bq.values() if len(v) > 1])
        mmeans = _np.array([_np.mean(v) for v in bm.values() if len(v) > 1])
        total = float(_np.var(all_b))
        between_q = float(_np.var(qmeans)) if len(qmeans) > 1 else 0.0
        between_m = float(_np.var(mmeans)) if len(mmeans) > 1 else 0.0
        summary["variance_decomposition"] = {
            "total_var": total,
            "between_question_var": between_q,
            "between_model_var": between_m,
            "frac_question": between_q / total if total > 0 else 0.0,
            "frac_model": between_m / total if total > 0 else 0.0,
        }

    # (per-model update efficiency is computed below, after per_model_curves is set)

    # Per-level mean aligned forecast across all (model, question), for the main figure.
    per_level_aligned = defaultdict(list)
    for model, qaligned in qm_aligned.items():
        for qid, lvls in qaligned.items():
            for lvl, val in lvls.items():
                per_level_aligned[lvl].append(val)
    summary["aggregate_aligned_by_level"] = {
        lvl: {"mean": float(np.mean(v)), "n": len(v), "se": float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0}
        for lvl, v in per_level_aligned.items()
    }

    # Per-model curves (aligned mean per level), for grouped bar / line figures.
    per_model_curves: Dict[str, Dict[str, float]] = {}
    for model, qaligned in qm_aligned.items():
        per_lvl = defaultdict(list)
        for qid, lvls in qaligned.items():
            for lvl, v in lvls.items():
                per_lvl[lvl].append(v)
        per_model_curves[model] = {lvl: float(np.mean(v)) for lvl, v in per_lvl.items()}
    summary["per_model_curves"] = per_model_curves

    # Update efficiency per model: fraction of available headroom closed L0->L3.
    per_model_eff: Dict[str, float] = {}
    for model, lvls in per_model_curves.items():
        a0 = lvls.get("L0")
        a3 = lvls.get("L3")
        if a0 is None or a3 is None:
            continue
        avail = 1.0 - a0
        per_model_eff[model] = (a3 - a0) / avail if avail > 0 else float("nan")
    summary["per_model_update_efficiency"] = per_model_eff

    # Per-question raw (kept compact for plotting).
    summary["per_question"] = {
        f"{model}__{qid}": {
            "slope": qm_slopes[model][qid],
            "brier_L3": qm_briers[model][qid].get("L3"),
            "aligned_by_level": qm_aligned[model][qid],
        }
        for model in qm_slopes for qid in qm_slopes[model]
    }

    RESULTS.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {RESULTS}")
    if "reasoning_vs_non" in summary:
        rv = summary["reasoning_vs_non"]
        print(f"\nReasoning slope: {rv['mean_slope_reasoning']:.4f}  "
              f"Non-reasoning slope: {rv['mean_slope_non_reasoning']:.4f}  "
              f"p={rv['p_value']:.4g}")
    if "slope_brier_correlation" in summary:
        c = summary["slope_brier_correlation"]
        print(f"Slope vs Brier r={c['pearson_r']:.3f}  p={c['p_value']:.4g}  n_models={c['n_models']}")


if __name__ == "__main__":
    main()
