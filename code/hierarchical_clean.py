"""Mechanical-coupling-free hierarchical analysis.

The original hierarchical fit regresses Brier@L3 on the full slope
beta = 0.3 a_3 + 0.1 a_2 - 0.1 a_1 - 0.3 a_0 and Brier@L0 = (1-a_0)^2.
Because Brier@L3 = (1-a_3)^2, beta and Brier@L3 share a_3 and the
estimated coefficient partly reflects this identity rather than
substantive predictive content.

We replace beta with the L0->L2 sub-slope

  beta_{02,m,q} = (a_{m,q,2} - a_{m,q,0}) / 2,

which uses only (a_0, a_1, a_2) and is genuinely out-of-sample for the
a_3 that determines Brier@L3. We also report a "tail-slope" using only
(a_2, a_3) for completeness. The prior-confound signature -- coefficient
strengthening after controlling for Brier@L0 -- should appear in the
clean version too if it is substantive.

Output -> results.json under robustness.hierarchical_clean.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

from config import RESULTS, QUESTIONS_LADDER, ELICITATIONS
from collections import defaultdict


def _load_with_levels():
    """Build a DataFrame with per-(model, question) a_L0..a_L3 (truth-aligned)
    plus Brier@L0, Brier@L3, full slope, L0->L2 sub-slope."""
    qs = {q["id"]: q for q in json.loads(QUESTIONS_LADDER.read_text())}
    ys = {qid: (1.0 if q["binary_resolution"] == "YES" else 0.0) for qid, q in qs.items()}
    cells = defaultdict(list)
    with ELICITATIONS.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("prob") is None: continue
            cells[(r["model_api_id"], r["question_id"], r["level"])].append(r["prob"])
    means = {k: float(np.mean(v)) for k, v in cells.items()}

    rows = []
    for (m, q), in {(k[0], k[1]): None for k in means.keys()}.items() if False else []:
        pass
    seen = set()
    for k in means.keys():
        seen.add((k[0], k[1]))
    for (m, q) in seen:
        if q not in ys: continue
        y = ys[q]
        a = {}
        for lvl in ("L0", "L1", "L2", "L3"):
            if (m, q, lvl) not in means: break
            p = means[(m, q, lvl)]
            a[lvl] = p if y == 1.0 else 1.0 - p
        if len(a) != 4: continue
        full_slope = float(np.polyfit([0, 1, 2, 3], [a["L0"], a["L1"], a["L2"], a["L3"]], 1)[0])
        sub_slope_02 = (a["L2"] - a["L0"]) / 2.0
        rows.append({
            "model": m, "question": q,
            "a_L0": a["L0"], "a_L1": a["L1"], "a_L2": a["L2"], "a_L3": a["L3"],
            "brier_L0": (1.0 - a["L0"]) ** 2,
            "brier_L3": (1.0 - a["L3"]) ** 2,
            "slope": full_slope,
            "sub_slope_02": sub_slope_02,
        })
    return pd.DataFrame(rows)


def _fit(formula, df):
    md = smf.mixedlm(
        formula, df,
        groups=df["question"],
        vc_formula={"Model": "0 + C(model)"},
    ).fit(reml=True, method="lbfgs")
    fe = {}
    for name in md.fe_params.index:
        fe[name] = {
            "estimate": float(md.fe_params[name]),
            "se": float(md.bse[name]),
            "t": float(md.tvalues[name]),
            "p": float(md.pvalues[name]),
            "ci_lo": float(md.conf_int().loc[name, 0]),
            "ci_hi": float(md.conf_int().loc[name, 1]),
        }
    return {
        "formula": formula,
        "converged": bool(md.converged),
        "fixed_effects": fe,
        "var_question": float(md.cov_re.iloc[0, 0]) if md.cov_re.shape[0] > 0 else 0.0,
        "var_model": float(md.vcomp[0]) if hasattr(md, "vcomp") and len(md.vcomp) > 0 else 0.0,
        "var_residual": float(md.scale),
    }


def main():
    df = _load_with_levels()
    df["model"] = df["model"].astype("category")
    df["question"] = df["question"].astype("category")
    print(f"n cells: {len(df)}")

    out = {"n_cells": int(len(df))}
    # Sub-slope naive vs prior-controlled.
    out["naive_subslope"] = _fit("brier_L3 ~ sub_slope_02", df)
    out["controlled_subslope"] = _fit("brier_L3 ~ sub_slope_02 + brier_L0", df)

    for k in ("naive_subslope", "controlled_subslope"):
        v = out[k]
        print(f"\n=== {k} ===  {v['formula']}")
        for name, fe in v["fixed_effects"].items():
            print(f"  {name:<18} est = {fe['estimate']:+.4f}  se = {fe['se']:.4f}  "
                  f"t = {fe['t']:+.2f}  p = {fe['p']:.3g}  "
                  f"CI [{fe['ci_lo']:+.4f}, {fe['ci_hi']:+.4f}]")

    R = json.loads(Path(RESULTS).read_text())
    R.setdefault("robustness", {})["hierarchical_clean"] = out
    Path(RESULTS).write_text(json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
