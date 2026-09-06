"""Re-fit V-A confirmatory hierarchical regression on the BLINDED ladders
(see build_ladder_blinded.py + elicit_blinded.py).

Mirrors hierarchical.py but points the loader at the blinded ladder file
and blinded elicitations file. Reports naive (Brier@L3 ~ slope) and
prior-controlled (Brier@L3 ~ slope + brier_L0) fits, and compares the
controlled slope coefficient to the original V-A confirmatory value
(gamma_1^orig = -1.781, 95% CI [-1.952, -1.610]).

Per RE_ELICITATION_DECISION_RULE.md, the pre-committed criterion is:
the V-A claim "survives content-blind ladders" iff the controlled slope
coefficient remains negative and its 95% CI excludes zero.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

from config import DATA, RESULTS
from hierarchical import _fit_and_extract

QUESTIONS_BLINDED = DATA / "questions_ladder_blinded.json"
ELICITATIONS_BLINDED = DATA / "elicitations_blinded.jsonl"


def _load_long_df_blinded():
    """Same shape as robustness._load_long_df, but reads blinded files."""
    qs = {q["id"]: q for q in json.loads(QUESTIONS_BLINDED.read_text())}
    rows = []
    with ELICITATIONS_BLINDED.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("prob") is None:
                continue
            rows.append(r)
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["model_api_id"], r["question_id"], r["level"])].append(r["prob"])
    cells = {k: float(np.mean(v)) for k, v in by_cell.items()}

    long_rows = []
    qids = sorted({q for _, q, _ in cells.keys()})
    models = sorted({m for m, _, _ in cells.keys()})
    for m in models:
        for q in qids:
            br = qs[q]["binary_resolution"]
            yq = 1.0 if (br == "YES" or br == 1 or br is True) else 0.0
            levels_p = {}
            for lvl in ("L0", "L1", "L2", "L3"):
                if (m, q, lvl) in cells:
                    levels_p[lvl] = cells[(m, q, lvl)]
            if "L0" not in levels_p or "L3" not in levels_p:
                continue
            aligned = {lvl: (p if yq == 1.0 else 1.0 - p) for lvl, p in levels_p.items()}
            xs = sorted(aligned.keys(), key=lambda l: int(l[1]))
            x_idx = [int(l[1]) for l in xs]
            y_val = [aligned[l] for l in xs]
            slope = float(np.polyfit(x_idx, y_val, 1)[0]) if len(xs) >= 2 else None
            long_rows.append({
                "model": m, "question": q, "yq": yq,
                "p_L0": levels_p.get("L0"), "p_L3": levels_p.get("L3"),
                "aligned_L0": aligned["L0"], "aligned_L3": aligned["L3"],
                "brier_L0": (levels_p["L0"] - yq) ** 2,
                "brier_L3": (levels_p["L3"] - yq) ** 2,
                "slope": slope,
            })
    return pd.DataFrame(long_rows)


# Pre-committed reference values from the original (un-blinded) V-A fit:
ORIG_NAIVE_SLOPE = -0.531        # naive coefficient on slope
ORIG_CONTROLLED_SLOPE = -1.781   # prior-controlled coefficient on slope
ORIG_CONTROLLED_CI = (-1.952, -1.610)


def main():
    df = _load_long_df_blinded()
    df = df.dropna(subset=["brier_L3", "brier_L0", "slope", "model", "question"]).copy()
    df["model"] = df["model"].astype("category")
    df["question"] = df["question"].astype("category")
    n_cells = len(df)
    print(f"Hierarchical (BLINDED) fit on n = {n_cells} (model, question) cells.\n")

    out = {"n_cells": int(n_cells)}
    fit_naive = _fit_and_extract(
        "brier_L3 ~ slope", df, vc_formula={"Model": "0 + C(model)"},
        label="naive (no prior control) [blinded]",
    )
    out["naive"] = fit_naive
    fit_ctrl = _fit_and_extract(
        "brier_L3 ~ slope + brier_L0", df, vc_formula={"Model": "0 + C(model)"},
        label="prior-controlled [blinded]",
    )
    out["controlled"] = fit_ctrl

    for k in ("naive", "controlled"):
        v = out[k]
        print(f"=== {v['label']} ===")
        print(f"formula: {v['formula']}  converged: {v['converged']}")
        for name, fe in v["fixed_effects"].items():
            print(f"  {name:<14} est = {fe['estimate']:+.4f}  se = {fe['se']:.4f}  "
                  f"t = {fe['t']:+.2f}  p = {fe['p']:.3g}  "
                  f"95% CI [{fe['ci_lo']:+.4f}, {fe['ci_hi']:+.4f}]")
        print()

    # Compare against pre-committed reference
    blind_naive = out["naive"]["fixed_effects"]["slope"]
    blind_ctrl = out["controlled"]["fixed_effects"]["slope"]
    print("=== DECISION RULE OUTCOME ===")
    print(f"Original naive      gamma_1 = {ORIG_NAIVE_SLOPE:+.3f}")
    print(f"Blinded  naive      gamma_1 = {blind_naive['estimate']:+.3f}  "
          f"95% CI [{blind_naive['ci_lo']:+.3f}, {blind_naive['ci_hi']:+.3f}]")
    print(f"Original controlled gamma_1 = {ORIG_CONTROLLED_SLOPE:+.3f}  "
          f"95% CI [{ORIG_CONTROLLED_CI[0]:+.3f}, {ORIG_CONTROLLED_CI[1]:+.3f}]")
    print(f"Blinded  controlled gamma_1 = {blind_ctrl['estimate']:+.3f}  "
          f"95% CI [{blind_ctrl['ci_lo']:+.3f}, {blind_ctrl['ci_hi']:+.3f}]")

    ci_excludes_zero = blind_ctrl['ci_hi'] < 0 or blind_ctrl['ci_lo'] > 0
    is_negative = blind_ctrl['estimate'] < 0
    ci_overlaps_orig = not (blind_ctrl['ci_hi'] < ORIG_CONTROLLED_CI[0]
                             or blind_ctrl['ci_lo'] > ORIG_CONTROLLED_CI[1])
    print(f"\nSurvives V-A criterion (controlled gamma_1 < 0 with 95% CI excl. 0)?")
    print(f"  negative: {is_negative}")
    print(f"  CI excludes zero: {ci_excludes_zero}")
    print(f"  CI overlaps original CI [-1.952, -1.610]: {ci_overlaps_orig}")
    print(f"  -> V-A claim {'SURVIVES' if (is_negative and ci_excludes_zero) else 'DOES NOT SURVIVE'} blinding.")

    # Append to results.json under robustness.hierarchical_blinded
    results_path = Path(RESULTS)
    if results_path.exists():
        results = json.loads(results_path.read_text())
    else:
        results = {}
    results.setdefault("robustness", {})["hierarchical_blinded"] = {
        **out,
        "decision_rule": {
            "blind_controlled_estimate": blind_ctrl["estimate"],
            "blind_controlled_ci": [blind_ctrl["ci_lo"], blind_ctrl["ci_hi"]],
            "blind_naive_estimate": blind_naive["estimate"],
            "blind_naive_ci": [blind_naive["ci_lo"], blind_naive["ci_hi"]],
            "orig_controlled_estimate": ORIG_CONTROLLED_SLOPE,
            "orig_controlled_ci": list(ORIG_CONTROLLED_CI),
            "is_negative": is_negative,
            "ci_excludes_zero": ci_excludes_zero,
            "ci_overlaps_orig": ci_overlaps_orig,
            "survives": bool(is_negative and ci_excludes_zero),
        },
    }
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote results to {results_path}")


if __name__ == "__main__":
    main()
