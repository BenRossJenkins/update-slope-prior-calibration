"""Round 2 robustness analyses (referee response):

  S1. Marginal Manifold capture ratio: (a_L3 - a_L0) / (P_late - a_L0),
      separates "conservatism on evidence from same starting point" from
      the total-vs-total ratio's prior-knowledge confound.
  S2. Single crossed random-effects model via vc_formula:
      Brier ~ 1 + (1|Question) + (1|Model). Reports Var(question),
      Var(model), Var(residual), ratio of point estimates.
  S3. Dose-response residual (DRR): residual of per-model slope from
      OLS regression on sqrt(Brier@L0). Names a metric and ranks models.
  S4. Leave-one-model-out bootstrap on variance components.

Appends results.robustness2 to data/results.json.
"""
from __future__ import annotations
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import warnings
warnings.filterwarnings("ignore")

from config import RESULTS, QUESTIONS_LADDER, ELICITATIONS, MODELS
from robustness import _load_long_df

RNG = np.random.default_rng(7)


# ---------- S1: marginal Manifold capture ----------

def s1_marginal_manifold():
    """Compute both Manifold ratios per (model, question) and summarize.

    total_ratio    = (a_3 - a_0) / (P_late - P_early)
    marginal_ratio = (a_3 - a_0) / (P_late - a_0)
    """
    results = json.loads(Path(RESULTS).read_text())
    rb = results.get("robustness", {})
    qs = {q["id"]: q for q in json.loads(QUESTIONS_LADDER.read_text())}

    # Per-(model, question) aligned forecasts.
    pq = results.get("per_question", {})
    cells: dict = {}
    for key, val in pq.items():
        api_id, qid = key.split("__", 1)
        a = val.get("aligned_by_level", {})
        if "L0" in a and "L3" in a:
            cells[(api_id, qid)] = (a["L0"], a["L3"])

    # Manifold anchors per question (truth-aligned p_early, p_late).
    # Reuse the stored sample but recompute on the fly to get all questions.
    # The earlier manifold_anchor.py saved only a sample; we need fresh data.
    # Instead: reuse the saved sample for n_questions; we recorded the medians,
    # so reconstruct from the stored intermediate output below.
    anchor_raw = rb.get("manifold_anchor", {}).get("per_question_anchor_sample", {})

    # If we only have a sample, re-fetch via cached per_question_anchor dictionary.
    # The earlier script wrote only 5 sample anchors; we need them all. We
    # rebuild from a separate JSON if present, else from cells alone we can't.
    # Simpler: rerun the anchor lookup logic locally using stored Manifold data
    # if present. For now we use whatever sample is there to demonstrate logic,
    # and compute the full set by also re-fetching IF needed -- but we already
    # have everything we need in the previous manifold pass; re-emit the full
    # per_question_anchor from that pass.
    return None  # placeholder; we'll compute from a separate script if needed


def s1_marginal_from_full(per_q_anchor: dict, cells: dict, qs: dict):
    """Both ratios summarised over (model, question) pairs.

    Inputs:
      per_q_anchor[qid] = {"p_early":..,"p_late":..,"manifold_update":..,"y":..}
      cells[(api_id, qid)] = (a_L0, a_L3)   truth-aligned
    """
    total, marginal = [], []
    per_model_total: dict = defaultdict(list)
    per_model_marg: dict = defaultdict(list)
    for (api_id, qid), (a0, a3) in cells.items():
        if qid not in per_q_anchor:
            continue
        info = per_q_anchor[qid]
        y = info["y"]
        # Truth-aligned market probs
        p_early_aligned = info["p_early"] if y == 1.0 else 1.0 - info["p_early"]
        p_late_aligned = info["p_late"] if y == 1.0 else 1.0 - info["p_late"]
        man_update = p_late_aligned - p_early_aligned
        llm_update = a3 - a0
        # Denominators: skip if too small
        if abs(man_update) > 0.05:
            total.append(llm_update / man_update)
            per_model_total[api_id].append(np.clip(llm_update / man_update, -5, 5))
        # Marginal: LLM update normalized by how much room LLM had to move
        # from its own prior to the human posterior.
        marg_denom = p_late_aligned - a0
        if abs(marg_denom) > 0.05:
            marginal.append(llm_update / marg_denom)
            per_model_marg[api_id].append(np.clip(llm_update / marg_denom, -5, 5))

    def _summary(arr):
        if not arr:
            return {}
        a = np.clip(np.array(arr), -5, 5)
        rng = np.random.default_rng(123)
        boots = [np.median(a[rng.integers(0, len(a), size=len(a))]) for _ in range(2000)]
        return {
            "n": int(len(a)),
            "median": float(np.median(a)),
            "mean": float(np.mean(a)),
            "lo95": float(np.percentile(boots, 2.5)),
            "hi95": float(np.percentile(boots, 97.5)),
        }
    return {
        "total_ratio": _summary(total),
        "marginal_ratio": _summary(marginal),
        "per_model_total_median": {k: float(np.median(v)) for k, v in per_model_total.items()},
        "per_model_marginal_median": {k: float(np.median(v)) for k, v in per_model_marg.items()},
    }


# ---------- S2: single crossed random-effects model ----------

def s2_crossed_re(df: pd.DataFrame):
    df = df.dropna(subset=["brier_L3", "question", "model"]).copy()
    # Single crossed model: Brier ~ 1, with vc_formula providing crossed RE.
    md = mixedlm(
        "brier_L3 ~ 1", df, groups=df["question"],
        vc_formula={"Model": "0 + C(model)"},
        re_formula="1",
    ).fit(reml=True, method="lbfgs")
    var_q = float(md.cov_re.iloc[0, 0]) if md.cov_re.shape[0] > 0 else 0.0
    var_m = float(md.vcomp[0]) if len(md.vcomp) > 0 else 0.0
    var_resid = float(md.scale)
    total = var_q + var_m + var_resid
    return {
        "var_question": var_q,
        "var_model": var_m,
        "var_residual": var_resid,
        "total_implied": total,
        "frac_question": var_q / total if total > 0 else 0.0,
        "frac_model": var_m / total if total > 0 else 0.0,
        "frac_residual": var_resid / total if total > 0 else 0.0,
        "ratio_question_over_model": (var_q / var_m) if var_m > 0 else float("inf"),
        "model_converged": bool(md.converged),
    }


# ---------- S3: dose-response residual (DRR) ----------

def s3_drr(per_model: dict):
    """Per-model slope residualized on sqrt(Brier@L0)."""
    items = [(k, v["mean_slope"], v["brier_L0"], v.get("display", k), v.get("reasoning"))
             for k, v in per_model.items()
             if v.get("brier_L0") is not None and v.get("mean_slope") is not None]
    slopes = np.array([x[1] for x in items])
    sqrt_b = np.sqrt(np.array([x[2] for x in items]))
    # OLS: slope = a + b*sqrt(Brier@L0) + eps
    X = np.column_stack([np.ones_like(sqrt_b), sqrt_b])
    beta, *_ = np.linalg.lstsq(X, slopes, rcond=None)
    pred = X @ beta
    resid = slopes - pred
    out = {}
    for (api_id, sl, b0, disp, reas), r, p in zip(items, resid, pred):
        out[api_id] = {
            "display": disp,
            "reasoning": reas,
            "slope": float(sl),
            "brier_L0": float(b0),
            "sqrt_brier_L0": float(np.sqrt(b0)),
            "predicted_slope_from_prior": float(p),
            "DRR": float(r),
        }
    return {
        "ols_intercept": float(beta[0]),
        "ols_coef_sqrt_brier_L0": float(beta[1]),
        "per_model": out,
    }


# ---------- S4: leave-one-model-out variance ratio ----------

def s4_loo_variance(df: pd.DataFrame):
    df = df.dropna(subset=["brier_L3", "question", "model"]).copy()
    models = sorted(df["model"].unique())
    ratios = []
    fracs_q = []
    for held in models:
        sub = df[df["model"] != held]
        try:
            md_q = mixedlm("brier_L3 ~ 1", sub, groups=sub["question"]).fit(reml=True, method="lbfgs")
            md_m = mixedlm("brier_L3 ~ 1", sub, groups=sub["model"]).fit(reml=True, method="lbfgs")
            vq = float(md_q.cov_re.iloc[0, 0])
            vm = float(md_m.cov_re.iloc[0, 0])
            vresid = float(md_q.scale)
            tot = vq + vm + vresid
            fracs_q.append(vq / tot if tot > 0 else 0.0)
            ratios.append((vq / vm) if vm > 0 else float("inf"))
        except Exception:
            continue
    finite_ratios = [r for r in ratios if np.isfinite(r)]
    return {
        "n_loo_runs": len(fracs_q),
        "frac_question_loo_mean": float(np.mean(fracs_q)) if fracs_q else None,
        "frac_question_loo_min": float(np.min(fracs_q)) if fracs_q else None,
        "frac_question_loo_max": float(np.max(fracs_q)) if fracs_q else None,
        "ratio_loo_median": float(np.median(finite_ratios)) if finite_ratios else None,
        "ratio_inf_count": int(len(ratios) - len(finite_ratios)),
    }


# ---------- driver ----------

def main():
    df = _load_long_df()
    results = json.loads(Path(RESULTS).read_text())
    rb2 = {}

    # We need the per-question Manifold anchors. Recompute the full anchor
    # dictionary from manifold_anchor.py's output if present; otherwise fall
    # back to what's saved (only a sample of 5).
    cached_anchor_file = Path(__file__).parent.parent / "data" / "manifold_anchor_full.json"
    if cached_anchor_file.exists():
        per_q_anchor = json.loads(cached_anchor_file.read_text())
    else:
        # Re-run the anchor pull inline -- expensive but only once.
        from manifold_anchor import _fetch_bets, _window_consensus
        from tqdm import tqdm
        qs = json.loads(QUESTIONS_LADDER.read_text())
        per_q_anchor = {}
        print("Recomputing Manifold anchors for all questions...")
        for q in tqdm(qs):
            if q["source"] != "manifold":
                continue
            market_id = q["id"].split("-", 1)[1]
            bets = _fetch_bets(market_id)
            if len(bets) < 5: continue
            ts = sorted(b["createdTime"] for b in bets)
            try:
                close_str = q.get("close_time")
                t_close = int(datetime.fromisoformat(close_str.replace("Z","+00:00")).timestamp()*1000) if close_str else ts[-1]
            except Exception:
                t_close = ts[-1]
            t_open = ts[0]
            span = t_close - t_open
            if span <= 0: continue
            p_early = _window_consensus(bets, t_open, t_open + 0.2*span)
            p_late = _window_consensus(bets, t_open + 0.8*span, t_close)
            if p_early is None or p_late is None: continue
            per_q_anchor[q["id"]] = {
                "p_early": p_early,
                "p_late": p_late,
                "manifold_update": p_late - p_early,
                "y": 1.0 if q["binary_resolution"] == "YES" else 0.0,
            }
        cached_anchor_file.write_text(json.dumps(per_q_anchor, indent=2))

    pq = results.get("per_question", {})
    cells = {}
    for key, val in pq.items():
        api_id, qid = key.split("__", 1)
        a = val.get("aligned_by_level", {})
        if "L0" in a and "L3" in a:
            cells[(api_id, qid)] = (a["L0"], a["L3"])
    qs = {q["id"]: q for q in json.loads(QUESTIONS_LADDER.read_text())}

    print("S1: marginal Manifold ratios ...")
    rb2["manifold"] = s1_marginal_from_full(per_q_anchor, cells, qs)
    print("S2: single crossed random-effects ...")
    rb2["crossed_re"] = s2_crossed_re(df)
    print("S3: dose-response residual ...")
    rb2["drr"] = s3_drr(results["models"])
    print("S4: leave-one-model-out ...")
    rb2["loo_variance"] = s4_loo_variance(df)

    results.setdefault("robustness", {})["v2"] = rb2
    Path(RESULTS).write_text(json.dumps(results, indent=2))

    # Print
    print()
    m = rb2["manifold"]
    print(f"Manifold total    ratio: median {m['total_ratio'].get('median'):.3f}  CI [{m['total_ratio'].get('lo95'):.3f}, {m['total_ratio'].get('hi95'):.3f}]  n={m['total_ratio'].get('n')}")
    print(f"Manifold marginal ratio: median {m['marginal_ratio'].get('median'):.3f}  CI [{m['marginal_ratio'].get('lo95'):.3f}, {m['marginal_ratio'].get('hi95'):.3f}]  n={m['marginal_ratio'].get('n')}")
    re = rb2["crossed_re"]
    print(f"\nCrossed RE: Var(Q)={re['var_question']:.4f}  Var(M)={re['var_model']:.4f}  Var(resid)={re['var_residual']:.4f}")
    print(f"           frac_Q={re['frac_question']:.3f}  frac_M={re['frac_model']:.3f}  ratio Q/M = {re['ratio_question_over_model']:.1f}")
    drr = rb2["drr"]
    print(f"\nDRR (sorted): coef sqrt(Br@L0) = {drr['ols_coef_sqrt_brier_L0']:+.3f}")
    rows = sorted(drr["per_model"].items(), key=lambda kv: -kv[1]["DRR"])
    for k, v in rows:
        flag = "R" if v["reasoning"] else "N"
        print(f"  {v['display']:<22} [{flag}] slope={v['slope']:+.4f}  DRR={v['DRR']:+.4f}")
    loo = rb2["loo_variance"]
    print(f"\nLOO variance: frac_Q range [{loo['frac_question_loo_min']:.3f}, {loo['frac_question_loo_max']:.3f}]  median Q/M ratio = {loo['ratio_loo_median']}")


if __name__ == "__main__":
    main()
