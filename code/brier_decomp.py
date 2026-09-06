"""Brier-score decomposition (Murphy 1973) per (model, level).

Brier(p, y) = Reliability - Resolution + Uncertainty.

For each (model, level) we bin per-question raw forecasts p_{m,q,ell}
against binary outcomes y_q, then compute the three components.

The motivating question: which component of forecasting skill does
prior calibration set, and which does updating modify? If prior-confound
applies to reliability (calibration), updating mostly improves that;
if it applies to resolution (discrimination), updating mostly improves
discrimination instead.

We also report per-model OLS regressions of Reliability@L3 and
Resolution@L3 on the corresponding L0 quantities, to isolate which
component carries the prior-vs-post correlation.

Output -> results.json under robustness.brier_decomp.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from config import RESULTS, QUESTIONS_LADDER, ELICITATIONS, MODELS


N_BINS = 5  # 0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
BIN_EDGES = np.linspace(0.0, 1.0, N_BINS + 1)


def _load_per_cell():
    """Returns dict[(model, q, level)] -> mean prob, and y[q]."""
    qs = {q["id"]: q for q in json.loads(QUESTIONS_LADDER.read_text())}
    ys = {qid: (1.0 if q["binary_resolution"] == "YES" else 0.0) for qid, q in qs.items()}
    cells = defaultdict(list)
    with ELICITATIONS.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("prob") is None:
                continue
            cells[(r["model_api_id"], r["question_id"], r["level"])].append(r["prob"])
    means = {k: float(np.mean(v)) for k, v in cells.items()}
    return means, ys


def brier_decomposition(probs: np.ndarray, outcomes: np.ndarray) -> dict:
    """Standard Murphy decomposition with N_BINS equal-width bins."""
    n = len(probs)
    if n == 0:
        return {"reliability": None, "resolution": None, "uncertainty": None, "brier": None}
    o_bar = float(np.mean(outcomes))
    uncertainty = o_bar * (1.0 - o_bar)
    bin_idx = np.clip(np.digitize(probs, BIN_EDGES[1:-1], right=False), 0, N_BINS - 1)
    rel, res = 0.0, 0.0
    for k in range(N_BINS):
        mask = bin_idx == k
        nk = int(mask.sum())
        if nk == 0:
            continue
        f_k = float(np.mean(probs[mask]))
        o_k = float(np.mean(outcomes[mask]))
        rel += (nk / n) * (f_k - o_k) ** 2
        res += (nk / n) * (o_k - o_bar) ** 2
    brier = float(np.mean((probs - outcomes) ** 2))
    return {
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": float(uncertainty),
        "brier": brier,
        "identity_check": float(rel - res + uncertainty),  # should ~= brier
        "n": int(n),
    }


def main():
    means, ys = _load_per_cell()
    # Aggregate per (model, level): collect (prob, y) pairs across questions.
    by_cell: dict = defaultdict(list)
    for (m, q, lvl), p in means.items():
        if q not in ys:
            continue
        by_cell[(m, lvl)].append((p, ys[q]))

    decomp: dict = {}  # decomp[model][level] = {...}
    for (m, lvl), pairs in by_cell.items():
        probs = np.array([p for p, _ in pairs])
        outs = np.array([y for _, y in pairs])
        d = brier_decomposition(probs, outs)
        decomp.setdefault(m, {})[lvl] = d

    # Per-model reliability/resolution at L0 vs L3.
    rows = []
    for m, by_lvl in decomp.items():
        d0 = by_lvl.get("L0", {})
        d3 = by_lvl.get("L3", {})
        if not d0 or not d3:
            continue
        rows.append({
            "model": m,
            "rel_L0": d0.get("reliability"),
            "rel_L3": d3.get("reliability"),
            "res_L0": d0.get("resolution"),
            "res_L3": d3.get("resolution"),
            "rel_reduction": (d0.get("reliability") or 0) - (d3.get("reliability") or 0),
            "res_gain": (d3.get("resolution") or 0) - (d0.get("resolution") or 0),
        })

    # Cross-model: does Reliability@L0 predict Reliability@L3?
    # Does Resolution@L0 predict Resolution@L3?
    def _corr(xs, ys):
        if len(xs) < 3:
            return None
        r, p = stats.pearsonr(xs, ys)
        return {"r": float(r), "p": float(p), "n": int(len(xs))}

    rel_l0 = np.array([r["rel_L0"] for r in rows if r["rel_L0"] is not None])
    rel_l3 = np.array([r["rel_L3"] for r in rows if r["rel_L3"] is not None])
    res_l0 = np.array([r["res_L0"] for r in rows if r["res_L0"] is not None])
    res_l3 = np.array([r["res_L3"] for r in rows if r["res_L3"] is not None])

    summary = {
        "per_model": rows,
        "rel_L0_vs_L3": _corr(rel_l0, rel_l3),
        "res_L0_vs_L3": _corr(res_l0, res_l3),
        "rel_reduction_mean": float(np.mean([r["rel_reduction"] for r in rows])),
        "res_gain_mean": float(np.mean([r["res_gain"] for r in rows])),
    }

    results = json.loads(Path(RESULTS).read_text())
    results.setdefault("robustness", {})["brier_decomp"] = {
        "decomp_per_model_level": {m: {lvl: d for lvl, d in by_lvl.items()} for m, by_lvl in decomp.items()},
        "summary": summary,
    }
    Path(RESULTS).write_text(json.dumps(results, indent=2))

    # Print.
    print("Per-model Brier decomposition (L0 vs L3):")
    print(f"{'model':<26} {'rel_L0':>7} {'rel_L3':>7} {'res_L0':>7} {'res_L3':>7} {'Δrel':>7} {'Δres':>7}")
    for r in rows:
        print(f"  {r['model']:<24} {r['rel_L0']:>7.4f} {r['rel_L3']:>7.4f} "
              f"{r['res_L0']:>7.4f} {r['res_L3']:>7.4f} "
              f"{-r['rel_reduction']:>+7.4f} {r['res_gain']:>+7.4f}")
    print()
    print(f"Mean reliability reduction L0->L3: {summary['rel_reduction_mean']:+.4f}")
    print(f"Mean resolution gain L0->L3:       {summary['res_gain_mean']:+.4f}")
    print()
    print(f"Rel@L0 vs Rel@L3 correlation: {summary['rel_L0_vs_L3']}")
    print(f"Res@L0 vs Res@L3 correlation: {summary['res_L0_vs_L3']}")


if __name__ == "__main__":
    main()
