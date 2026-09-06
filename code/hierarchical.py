"""Hierarchical mixed-effects analysis (referee response Option 2).

Replaces the cross-model n=7 Pearson r analysis with a hierarchical
model that uses all 280 (model, question) cells:

  Brier@L3 ~ slope_{m,q} + Brier@L0_{m,q} + (1|model) + (1|question)

The fixed-effect coefficient on slope (controlling for Brier@L0) is the
confirmatory analogue of the n=7 partial correlation. It has a proper
standard error driven by the within-cell residual variance rather than
by between-model noise at n=7.

We also report a simpler version (without Brier@L0 control) so the
prior-confound is visible in the t-statistics.

Output -> results.json under robustness.hierarchical.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

from config import RESULTS
from robustness import _load_long_df


def _fit_and_extract(formula: str, df: pd.DataFrame, vc_formula=None, label: str = "") -> dict:
    """Fit a mixed model and extract the fixed-effect summary."""
    kwargs = dict(groups=df["question"])
    if vc_formula is not None:
        kwargs["vc_formula"] = vc_formula
    md = smf.mixedlm(formula, df, **kwargs).fit(reml=True, method="lbfgs")
    out = {"formula": formula, "label": label, "converged": bool(md.converged)}
    fixed = {}
    for name in md.fe_params.index:
        fixed[name] = {
            "estimate": float(md.fe_params[name]),
            "se": float(md.bse[name]),
            "t": float(md.tvalues[name]),
            "p": float(md.pvalues[name]),
            "ci_lo": float(md.conf_int().loc[name, 0]),
            "ci_hi": float(md.conf_int().loc[name, 1]),
        }
    out["fixed_effects"] = fixed
    out["var_question"] = float(md.cov_re.iloc[0, 0]) if md.cov_re.shape[0] > 0 else 0.0
    out["var_model"] = float(md.vcomp[0]) if hasattr(md, "vcomp") and len(md.vcomp) > 0 else 0.0
    out["var_residual"] = float(md.scale)
    return out


def main():
    df = _load_long_df()
    df = df.dropna(subset=["brier_L3", "brier_L0", "slope", "model", "question"]).copy()
    df["model"] = df["model"].astype("category")
    df["question"] = df["question"].astype("category")
    n_cells = len(df)
    print(f"Hierarchical fit on n = {n_cells} (model, question) cells.\n")

    out = {"n_cells": int(n_cells)}

    # 1. Naive: Brier@L3 ~ slope, no controls.
    fit_naive = _fit_and_extract(
        "brier_L3 ~ slope",
        df, vc_formula={"Model": "0 + C(model)"},
        label="naive (no prior control)",
    )
    out["naive"] = fit_naive

    # 2. Controlled: Brier@L3 ~ slope + Brier@L0
    fit_ctrl = _fit_and_extract(
        "brier_L3 ~ slope + brier_L0",
        df, vc_formula={"Model": "0 + C(model)"},
        label="prior-controlled",
    )
    out["controlled"] = fit_ctrl

    # 3. Just prior, for completeness.
    fit_prior = _fit_and_extract(
        "brier_L3 ~ brier_L0",
        df, vc_formula={"Model": "0 + C(model)"},
        label="prior only",
    )
    out["prior_only"] = fit_prior

    results = json.loads(Path(RESULTS).read_text())
    results.setdefault("robustness", {})["hierarchical"] = out
    Path(RESULTS).write_text(json.dumps(results, indent=2))

    for k in ("naive", "controlled", "prior_only"):
        v = out[k]
        print(f"=== {v['label']} ===")
        print(f"formula: {v['formula']}")
        print(f"converged: {v['converged']}")
        for name, fe in v["fixed_effects"].items():
            print(f"  {name:<18} est = {fe['estimate']:+.4f}  se = {fe['se']:.4f}  "
                  f"t = {fe['t']:+.2f}  p = {fe['p']:.3g}  "
                  f"95% CI [{fe['ci_lo']:+.4f}, {fe['ci_hi']:+.4f}]")
        print(f"  Var(question) = {v['var_question']:.5f}  Var(model) = {v['var_model']:.5f}  "
              f"Var(resid) = {v['var_residual']:.5f}")
        print()


if __name__ == "__main__":
    main()
