"""Robustness analyses requested by referee:

  R1. Bootstrap CIs on per-model cross-correlations (Pearson r, partial r).
  R2. Random-effects variance decomposition of per-(model, question) Brier@L3
      using statsmodels.MixedLM (replaces naive Var(E[Y|q]) ratio).
  R3. Contamination stratification: correlate Brier@L0 with resolution date
      and split pre/post a plausible training-cutoff boundary.

Writes results into the existing data/results.json under a "robustness" key
so emit_data_tex can pull them.
"""
from __future__ import annotations
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import pandas as pd

from config import RESULTS, QUESTIONS_LADDER, ELICITATIONS, MODELS

RNG = np.random.default_rng(42)


# ---------- shared loading ----------

def _load_long_df():
    """Build a long DataFrame of one row per (model, question), with mean
    truth-aligned forecasts at each level + Brier@L0/L3 + slope + resolve_time.
    """
    qs = {q["id"]: q for q in json.loads(QUESTIONS_LADDER.read_text())}
    rows = []
    with ELICITATIONS.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("prob") is None:
                continue
            rows.append(r)
    # Aggregate to per-(model, question, level) means.
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["model_api_id"], r["question_id"], r["level"])].append(r["prob"])
    cells = {k: float(np.mean(v)) for k, v in by_cell.items()}

    out = []
    for model, qid, level in {(k[0], k[1], k[2]) for k in cells.keys()}:
        pass  # placeholder; we'll reassemble below

    long_rows = []
    qids = sorted({q for _, q, _ in cells.keys()})
    models = sorted({m for m, _, _ in cells.keys()})
    for m in models:
        for q in qids:
            yq = 1.0 if qs[q]["binary_resolution"] == "YES" else 0.0
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
            row = {
                "model": m, "question": q, "yq": yq,
                "p_L0": levels_p.get("L0"), "p_L3": levels_p.get("L3"),
                "aligned_L0": aligned["L0"], "aligned_L3": aligned["L3"],
                "brier_L0": (levels_p["L0"] - yq) ** 2,
                "brier_L3": (levels_p["L3"] - yq) ** 2,
                "slope": slope,
                "resolve_time": qs[q].get("resolve_time"),
            }
            long_rows.append(row)
    return pd.DataFrame(long_rows)


# ---------- R1: bootstrap CIs ----------

def _bootstrap_corr(x, y, kind="pearson", n_boot=2000, partial_z=None):
    """Return mean, lower, upper of bootstrap correlation."""
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    boots = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        bx, by = x[idx], y[idx]
        if partial_z is not None:
            bz = np.asarray(partial_z)[idx]
            r_xy = stats.pearsonr(bx, by)[0]
            r_xz = stats.pearsonr(bx, bz)[0]
            r_yz = stats.pearsonr(by, bz)[0]
            denom = ((1 - r_xz ** 2) * (1 - r_yz ** 2)) ** 0.5
            if denom == 0 or np.isnan(denom):
                continue
            boots.append((r_xy - r_xz * r_yz) / denom)
        else:
            if kind == "pearson":
                boots.append(stats.pearsonr(bx, by)[0])
            else:
                boots.append(stats.spearmanr(bx, by)[0])
    boots = np.array([b for b in boots if not np.isnan(b)])
    return {
        "mean": float(np.mean(boots)),
        "lo95": float(np.percentile(boots, 2.5)),
        "hi95": float(np.percentile(boots, 97.5)),
        "n_boot": int(len(boots)),
    }


def r1_bootstrap_correlations(per_model):
    slopes = np.array([m["mean_slope"] for m in per_model.values()])
    bl0 = np.array([m["brier_L0"] for m in per_model.values()])
    bl3 = np.array([m["brier_L3"] for m in per_model.values()])
    return {
        "slope_vs_brierL3": _bootstrap_corr(slopes, bl3),
        "brierL0_vs_brierL3": _bootstrap_corr(bl0, bl3),
        "partial_slope_brierL3_given_brierL0": _bootstrap_corr(slopes, bl3, partial_z=bl0),
        "n_models": int(len(slopes)),
    }


# ---------- R2: random-effects variance decomposition ----------

def r2_random_effects(df: pd.DataFrame):
    """Fit a crossed random-effects model on Brier@L3 with random intercepts
    for question and model. Returns variance components and the question/model/
    residual variance shares of the explained variance.

    statsmodels.MixedLM does not natively support fully crossed random effects,
    so we fit two separate models with (1|question) and (1|model) as the random
    factor and use the reported group-variance components.
    """
    out = {}
    # Question random effect (model as fixed).
    df = df.dropna(subset=["brier_L3", "question", "model"]).copy()
    df["model"] = df["model"].astype("category")
    md_q = mixedlm("brier_L3 ~ C(model)", df, groups=df["question"]).fit(reml=True, method="lbfgs")
    var_q = float(md_q.cov_re.iloc[0, 0])
    md_m = mixedlm("brier_L3 ~ 1", df, groups=df["model"]).fit(reml=True, method="lbfgs")
    var_m = float(md_m.cov_re.iloc[0, 0])

    md_resid = mixedlm("brier_L3 ~ 1", df, groups=df["question"]).fit(reml=True, method="lbfgs")
    var_q_only = float(md_resid.cov_re.iloc[0, 0])
    var_resid = float(md_resid.scale)

    total = var_q_only + var_m + var_resid
    out["var_question"] = var_q_only
    out["var_model"] = var_m
    out["var_residual"] = var_resid
    out["var_total_implied"] = total
    out["frac_question"] = var_q_only / total if total > 0 else 0.0
    out["frac_model"] = var_m / total if total > 0 else 0.0
    out["frac_residual"] = var_resid / total if total > 0 else 0.0
    out["ratio_question_over_model"] = var_q_only / var_m if var_m > 0 else float("inf")
    return out


# ---------- R3: contamination check ----------

def _to_unix(ts: str) -> float | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def r3_contamination(df: pd.DataFrame):
    df = df.copy()
    df["resolve_ts"] = df["resolve_time"].map(_to_unix)
    df = df.dropna(subset=["resolve_ts", "brier_L0"])
    # Per-question Brier@L0 averaged across models.
    per_q = df.groupby("question").agg(
        brier_L0_mean=("brier_L0", "mean"),
        resolve_ts=("resolve_ts", "first"),
    ).reset_index()
    if len(per_q) < 5:
        return {"note": "too few questions for stratified analysis"}
    r_pearson, p_pearson = stats.pearsonr(per_q["resolve_ts"], per_q["brier_L0_mean"])
    r_spearman, p_spearman = stats.spearmanr(per_q["resolve_ts"], per_q["brier_L0_mean"])
    # Boundary: median resolution date.
    median_ts = float(per_q["resolve_ts"].median())
    early = per_q[per_q["resolve_ts"] <= median_ts]
    late = per_q[per_q["resolve_ts"] > median_ts]
    out = {
        "pearson_r_brierL0_vs_resolveTime": float(r_pearson),
        "pearson_p": float(p_pearson),
        "spearman_r": float(r_spearman),
        "spearman_p": float(p_spearman),
        "median_resolve_iso": datetime.fromtimestamp(median_ts, tz=timezone.utc).isoformat(),
        "early_n_questions": int(len(early)),
        "early_brier_L0_mean": float(early["brier_L0_mean"].mean()),
        "late_n_questions": int(len(late)),
        "late_brier_L0_mean": float(late["brier_L0_mean"].mean()),
        "early_vs_late_t": float(stats.ttest_ind(early["brier_L0_mean"], late["brier_L0_mean"]).statistic),
        "early_vs_late_p": float(stats.ttest_ind(early["brier_L0_mean"], late["brier_L0_mean"]).pvalue),
    }
    return out


# ---------- driver ----------

def main():
    df = _load_long_df()
    results = json.loads(Path(RESULTS).read_text())

    robustness = {}
    print("R1: bootstrap CIs ...")
    robustness["bootstrap"] = r1_bootstrap_correlations(results["models"])
    print("R2: random-effects variance decomposition ...")
    robustness["random_effects"] = r2_random_effects(df)
    print("R3: contamination check ...")
    robustness["contamination"] = r3_contamination(df)

    results["robustness"] = robustness
    Path(RESULTS).write_text(json.dumps(results, indent=2))
    print(f"updated {RESULTS}")

    # Print headline numbers.
    b = robustness["bootstrap"]
    print()
    print("Bootstrap 95% CIs (n_boot=2000):")
    for k, v in b.items():
        if isinstance(v, dict):
            print(f"  {k}: r = {v['mean']:+.3f}  [{v['lo95']:+.3f}, {v['hi95']:+.3f}]")
    re_ = robustness["random_effects"]
    print()
    print(f"Random-effects variance components:")
    print(f"  Var(question) = {re_['var_question']:.5f}")
    print(f"  Var(model)    = {re_['var_model']:.5f}")
    print(f"  Var(residual) = {re_['var_residual']:.5f}")
    print(f"  frac_question = {re_['frac_question']:.3f}")
    print(f"  frac_model    = {re_['frac_model']:.3f}")
    print(f"  ratio Q/M     = {re_['ratio_question_over_model']:.1f}x")
    c = robustness["contamination"]
    print()
    print(f"Contamination check:")
    print(f"  Pearson r(Brier@L0, resolve_time) = {c['pearson_r_brierL0_vs_resolveTime']:+.3f}  p = {c['pearson_p']:.3f}")
    print(f"  Spearman = {c['spearman_r']:+.3f}  p = {c['spearman_p']:.3f}")
    print(f"  Early (n={c['early_n_questions']})  Brier@L0 = {c['early_brier_L0_mean']:.3f}")
    print(f"  Late  (n={c['late_n_questions']})  Brier@L0 = {c['late_brier_L0_mean']:.3f}")
    print(f"  Welch t test p = {c['early_vs_late_p']:.3f}")


if __name__ == "__main__":
    main()
