"""Compare Manifold (original), ForecastBench (primary), and ACLED (boundary)
pools on the metrics pre-registered in PREREGISTRATION.md:

P1/B1. OLS coefficient of per-model mean slope on sqrt(Brier@L0)
P2.    Pearson r(Brier@L0, Brier@L3)
P3.    Variance shares: question vs model
P4.    DRR^Res Kendall tau vs Brier@L3 (descriptive)
B2.    Per-model slope spread (max - min)
B3.    Per-model Brier@L0 spread (max - min)

Reads each pool's results.json from data/<run>/results.json and writes
a single comparison JSON to data/pool_comparison.json plus a markdown
summary printed to stdout.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

POOLS = [
    ("Manifold", DATA / "results.json"),
    ("ForecastBench primary", DATA / "forecastbench" / "results.json"),
    ("ACLED boundary", DATA / "acled" / "results.json"),
]


def _load_per_model(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract per-model summaries with Br@L0, Br@L3, mean_slope."""
    rows = []
    for api_id, m in results.get("models", {}).items():
        if m.get("brier_L0") is None or m.get("brier_L3") is None:
            continue
        rows.append({
            "api_id": api_id,
            "display": m.get("display", api_id),
            "reasoning": m.get("reasoning", False),
            "brier_L0": m["brier_L0"],
            "brier_L3": m["brier_L3"],
            "mean_slope": m["mean_slope"],
        })
    return rows


def _compute(pool_name: str, results: Dict[str, Any]) -> Dict[str, Any]:
    per_model = _load_per_model(results)
    if len(per_model) < 3:
        return {"pool": pool_name, "error": f"only {len(per_model)} models with data"}

    brier_L0 = np.array([m["brier_L0"] for m in per_model])
    brier_L3 = np.array([m["brier_L3"] for m in per_model])
    slopes = np.array([m["mean_slope"] for m in per_model])
    sqrt_br_L0 = np.sqrt(brier_L0)

    # P1/B1: OLS coefficient of slope on sqrt(Br@L0)
    slope_coef, intercept = np.polyfit(sqrt_br_L0, slopes, 1)

    # P2: Pearson r(Br@L0, Br@L3)
    pearson_r, pearson_p = stats.pearsonr(brier_L0, brier_L3)

    # P3: variance shares (taken directly from results.json)
    vd = results.get("variance_decomposition", {})

    # B2, B3: ranges
    slope_spread = float(slopes.max() - slopes.min())
    brier_L0_spread = float(brier_L0.max() - brier_L0.min())

    return {
        "pool": pool_name,
        "n_models": len(per_model),
        "slope_on_sqrt_brierL0_coef": float(slope_coef),
        "pearson_r_brierL0_brierL3": float(pearson_r),
        "pearson_p": float(pearson_p),
        "frac_question_var": vd.get("frac_question"),
        "frac_model_var": vd.get("frac_model"),
        "slope_spread_max_minus_min": slope_spread,
        "brierL0_spread_max_minus_min": brier_L0_spread,
        "per_model": per_model,
    }


def main():
    comparisons: List[Dict[str, Any]] = []
    for name, path in POOLS:
        if not path.exists():
            comparisons.append({"pool": name, "missing": str(path)})
            continue
        comparisons.append(_compute(name, json.loads(path.read_text())))

    (DATA / "pool_comparison.json").write_text(json.dumps(comparisons, indent=2))

    print("\n## Cross-pool comparison (pre-registration check)\n")
    print(f"{'Pool':<28} {'n':<3} {'slope|sqrtBrL0':>16} {'r(BrL0,BrL3)':>14} {'qVar%':>7} {'mVar%':>7} {'βspread':>9} {'BrL0spread':>11}")
    print("-" * 100)
    for c in comparisons:
        if "error" in c or "missing" in c:
            print(f"{c['pool']:<28} -- {c.get('missing') or c.get('error')}")
            continue
        qv = f"{100*c['frac_question_var']:.1f}" if c.get("frac_question_var") is not None else "?"
        mv = f"{100*c['frac_model_var']:.1f}" if c.get("frac_model_var") is not None else "?"
        print(
            f"{c['pool']:<28} {c['n_models']:<3} "
            f"{c['slope_on_sqrt_brierL0_coef']:>+16.4f} "
            f"{c['pearson_r_brierL0_brierL3']:>+14.3f} "
            f"{qv:>7} {mv:>7} "
            f"{c['slope_spread_max_minus_min']:>9.4f} "
            f"{c['brierL0_spread_max_minus_min']:>11.4f}"
        )
    print()
    print("Pre-registered targets (PREREGISTRATION.md):")
    print("  P1 (primary):  slope-on-sqrt(BrL0) point estimate in [0.07, 0.20], positive sign")
    print("  P2 (primary):  Pearson r(BrL0, BrL3) >= 0.40, positive")
    print("  P3 (primary):  question-share / model-share ratio >= 5:1")
    print("  B1 (boundary): coef positive sign, magnitude in [0, 0.07], < primary")
    print("  B2 (boundary): slope spread < 0.025 (Manifold = 0.031)")
    print("  B3 (boundary): BrL0 spread < 0.10 (Manifold = 0.127)")


if __name__ == "__main__":
    main()
