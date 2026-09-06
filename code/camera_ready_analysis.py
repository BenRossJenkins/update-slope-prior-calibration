"""Camera-ready analysis for ICTAI 2026 paper 322.

Computes, from the raw elicitation logs, every quantity added or corrected
for the camera-ready revision, and writes data/camera_ready.json:

  1. Cell accounting: 280 total cells, complete-cell counts per dataset,
     the 12 incomplete original cells and why (null-prob records at L0/L1).
  2. Headroom-utilization table: per-model a0, H = 1 - a0, slope (partial-
     ladder convention matching analyze.py), u = slope / (K3 * H); panel
     stats and the K3 * mean(u) mechanism check against the observed
     slope-on-sqrt(Brier@L0) coupling.
  3. Corrected hierarchical 2x2 (naive / prior-controlled x original /
     blinded) using the L0->L2 sub-slope spec of hierarchical_clean.py,
     plus the blinded fit restricted to the 268 common cells and the
     a0-free L1->L2 robustness fits. Supersedes hierarchical_blinded.py,
     whose regressor was the full L0->L3 slope (which shares a3 with
     Brier@L3) and is not comparable to the original sub-slope fit.
  4. Cell-level monotonicity: fraction weakly monotone, dip distribution.
  5. Full-precision Brier@L3 per model (Kendall-tie check).
  6. Classical change-score comparators: Hake normalized gain per model,
     Oldham correlation (change vs mean) next to the naive change-vs-
     baseline correlation, and a Blomqvist-style measurement-error-
     corrected slope of change on baseline using the K=3 within-cell
     sample variance.
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

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
K3 = 0.4
LEVELS = ["L0", "L1", "L2", "L3"]


def load_probs(path):
    probs = defaultdict(list)
    n_null = 0
    for line in open(path):
        r = json.loads(line)
        if r["prob"] is None:
            n_null += 1
            continue
        probs[(r["model_api_id"], r["question_id"], r["level"])].append(float(r["prob"]))
    return probs, n_null


def main():
    truth = {q["id"]: (1.0 if q["binary_resolution"] == "YES" else 0.0)
             for q in json.load(open(DATA / "questions_raw.json"))}
    displays = {}
    for line in open(DATA / "elicitations.jsonl"):
        r = json.loads(line)
        displays[r["model_api_id"]] = r["model_display"]
    orig, n_null_orig = load_probs(DATA / "elicitations.jsonl")
    blind, n_null_blind = load_probs(DATA / "elicitations_blinded.jsonl")
    models = sorted(displays)

    def aligned_mean(probs, m, q, lvl):
        ps = probs.get((m, q, lvl))
        if not ps:
            return None
        p = float(np.mean(ps))
        return p if truth[q] == 1.0 else 1.0 - p

    def complete_cells(probs):
        out = {}
        for m in models:
            for q in truth:
                traj = [aligned_mean(probs, m, q, lvl) for lvl in LEVELS]
                if all(t is not None for t in traj):
                    out[(m, q)] = traj
        return out

    co, cb = complete_cells(orig), complete_cells(blind)
    common = sorted(set(co) & set(cb))
    incomplete = [
        {"model": displays[m], "question": q,
         "valid_samples": {lvl: len([p for p in orig.get((m, q, lvl), []) if p is not None])
                          for lvl in LEVELS}}
        for m in models for q in truth if (m, q) not in co
    ]

    # ---- 2. u-table (analyze.py conventions: slope fit on available levels >=2,
    #         a0/Brier@L0 over questions with L0), for both ladders ----
    # ---- 2. u-table (both ladders) ----
    def build_u(probs):
        u_table = []
        for m in models:
            slopes, a0s = [], []
            for q in truth:
                pts = [(i, aligned_mean(probs, m, q, lvl)) for i, lvl in enumerate(LEVELS)]
                pts = [(x, y) for x, y in pts if y is not None]
                if len(pts) >= 2:
                    xs, ys = zip(*pts)
                    slopes.append(float(np.polyfit(xs, ys, 1)[0]))
                a0 = aligned_mean(probs, m, q, "L0")
                if a0 is not None:
                    a0s.append(a0)
            a0 = float(np.mean(a0s)); H = 1.0 - a0
            slope = float(np.mean(slopes))
            brier0 = float(np.mean([(1 - a) ** 2 for a in a0s]))
            u_table.append({"model": displays[m], "api_id": m, "a0": a0, "H": H,
                            "slope": slope, "u": slope / (K3 * H),
                            "brier_L0": brier0, "sqrt_brier_L0": float(np.sqrt(brier0))})
        us = np.array([r["u"] for r in u_table]); Hs = np.array([r["H"] for r in u_table])
        sl = np.array([r["slope"] for r in u_table]); sb = np.array([r["sqrt_brier_L0"] for r in u_table])
        mech = {"u_mean": float(us.mean()), "u_sd": float(us.std(ddof=1)),
                "u_min": float(us.min()), "u_max": float(us.max()),
                "u_cv": float(us.std(ddof=1) / us.mean()),
                "K3_times_u_mean": float(K3 * us.mean()),
                "ols_slope_on_sqrt_brier_L0": float(np.polyfit(sb, sl, 1)[0]),
                "ols_slope_on_H": float(np.polyfit(Hs, sl, 1)[0]),
                "corr_u_H": float(np.corrcoef(us, Hs)[0, 1])}
        return u_table, mech

    u_table, mechanism = build_u(orig)
    u_table_blind, mechanism_blind = build_u(blind)

    # aggregate aligned means per level, both ladders
    def agg_levels(probs):
        out = {}
        for i, lvl in enumerate(LEVELS):
            vals = [aligned_mean(probs, m, q, lvl) for m in models for q in truth]
            vals = [v for v in vals if v is not None]
            out[lvl] = float(np.mean(vals))
        return out
    agg = {"orig": agg_levels(orig), "blind": agg_levels(blind)}

    # ---- 3. hierarchical fits (hierarchical_clean.py spec) ----
    def frame(cdict, keys):
        return pd.DataFrame([
            {"model": m, "question": q,
             "brier_L0": (1 - t[0]) ** 2, "brier_L3": (1 - t[3]) ** 2,
             "sub_slope_02": (t[2] - t[0]) / 2.0,
             "slope_12": t[2] - t[1]}
            for (m, q) in keys for t in [cdict[(m, q)]]])

    def fit(df, formula, term):
        # Default optimizer reproduces the submitted paper's original fit
        # (coef -1.781, SE 0.086, CI [-1.949, -1.614]) exactly.
        md = smf.mixedlm(formula, df, groups=df["question"],
                         vc_formula={"Model": "0 + C(model)"}).fit()
        ci = md.conf_int().loc[term]
        return {"coef": float(md.fe_params[term]), "se": float(md.bse[term]),
                "ci_lo": float(ci[0]), "ci_hi": float(ci[1]), "n": int(len(df))}

    fo, fb = frame(co, sorted(co)), frame(cb, sorted(cb))
    fbc = frame(cb, common)
    hier = {
        "orig_naive": fit(fo, "brier_L3 ~ sub_slope_02", "sub_slope_02"),
        "orig_ctrl": fit(fo, "brier_L3 ~ sub_slope_02 + brier_L0", "sub_slope_02"),
        "blind_naive": fit(fb, "brier_L3 ~ sub_slope_02", "sub_slope_02"),
        "blind_ctrl": fit(fb, "brier_L3 ~ sub_slope_02 + brier_L0", "sub_slope_02"),
        "blind_ctrl_common268": fit(fbc, "brier_L3 ~ sub_slope_02 + brier_L0", "sub_slope_02"),
        "orig_ctrl_slope12": fit(fo, "brier_L3 ~ slope_12 + brier_L0", "slope_12"),
        "blind_ctrl_slope12": fit(fb, "brier_L3 ~ slope_12 + brier_L0", "slope_12"),
    }

    # ---- 4. cell-level monotonicity ----
    dips = []
    for t in co.values():
        dips.append(max(max(0.0, t[i] - t[i + 1]) for i in range(3)))
    dips = np.array(dips)
    mono = {
        "n_cells": int(len(dips)),
        "frac_weakly_monotone": float((dips == 0).mean()),
        "eps_median": float(np.median(dips)), "eps_mean": float(dips.mean()),
        "eps_p90": float(np.percentile(dips, 90)), "eps_max": float(dips.max()),
    }

    # ---- 5. full-precision Brier@L3 ----
    brier_l3 = {}
    for m in models:
        vals = []
        for q, y in truth.items():
            ps = orig.get((m, q, "L3"))
            if ps:
                vals.append((float(np.mean(ps)) - y) ** 2)
        brier_l3[displays[m]] = float(np.mean(vals))

    # ---- 6. classical comparators ----
    # per-model change and endpoints (questions with both L0 and L3)
    comp_rows = []
    for m in models:
        a0s, a3s, se2_terms = [], [], []
        for q in truth:
            p0 = orig.get((m, q, "L0"))
            p3 = orig.get((m, q, "L3"))
            if not p0 or not p3:
                continue
            a0 = float(np.mean(p0)) if truth[q] == 1.0 else 1 - float(np.mean(p0))
            a3 = float(np.mean(p3)) if truth[q] == 1.0 else 1 - float(np.mean(p3))
            a0s.append(a0)
            a3s.append(a3)
            # sampling variance of the cell-mean L0 prob (K<=3 draws)
            if len(p0) > 1:
                se2_terms.append(float(np.var(p0, ddof=1)) / len(p0))
            else:
                se2_terms.append(0.0)
        a0m, a3m = float(np.mean(a0s)), float(np.mean(a3s))
        comp_rows.append({
            "model": displays[m], "a0": a0m, "a3": a3m, "change": a3m - a0m,
            "hake_gain": (a3m - a0m) / (1 - a0m),
            "se2_a0_mean": float(np.sum(se2_terms)) / (len(a0s) ** 2),
        })
    a0v = np.array([r["a0"] for r in comp_rows])
    chg = np.array([r["change"] for r in comp_rows])
    meanv = (a0v + np.array([r["a3"] for r in comp_rows])) / 2.0
    b_obs = float(np.polyfit(a0v, chg, 1)[0])
    S2 = float(np.var(a0v, ddof=1))
    sigma_e2 = float(np.mean([r["se2_a0_mean"] for r in comp_rows]))
    b_blomqvist = (b_obs * S2 + sigma_e2) / (S2 - sigma_e2) if S2 > sigma_e2 else None
    comparators = {
        "per_model": comp_rows,
        "corr_change_baseline": float(np.corrcoef(a0v, chg)[0, 1]),
        "corr_change_oldham_mean": float(np.corrcoef(meanv, chg)[0, 1]),
        "slope_change_on_baseline_obs": b_obs,
        "baseline_var_S2": S2,
        "measurement_var_sigma_e2": sigma_e2,
        "blomqvist_corrected_slope": b_blomqvist,
    }

    out = {
        "cells": {
            "total": len(models) * len(truth),
            "orig_complete": len(co), "blind_complete": len(cb),
            "common": len(common), "orig_null_records": n_null_orig,
            "blind_null_records": n_null_blind, "incomplete_cells": incomplete,
        },
        "u_table": u_table, "mechanism": mechanism,
        "u_table_blind": u_table_blind, "mechanism_blind": mechanism_blind,
        "aggregate_aligned": agg, "hierarchical": hier,
        "monotonicity": mono, "brier_L3_full_precision": brier_l3,
        "comparators": comparators,
    }
    (DATA / "camera_ready.json").write_text(json.dumps(out, indent=1))
    print("wrote", DATA / "camera_ready.json")
    print(json.dumps({k: out[k] for k in ["mechanism", "hierarchical", "monotonicity", "comparators"]
                      if k != "comparators"}, indent=1)[:1200])
    c = out["comparators"]
    print("comparators: corr(change, baseline) =", round(c["corr_change_baseline"], 3),
          "| Oldham corr(change, mean) =", round(c["corr_change_oldham_mean"], 3))
    print("slope(change~baseline) obs =", round(c["slope_change_on_baseline_obs"], 3),
          "| Blomqvist-corrected =", round(c["blomqvist_corrected_slope"], 3) if c["blomqvist_corrected_slope"] else None,
          "| sigma_e2/S2 =", round(c["measurement_var_sigma_e2"] / c["baseline_var_S2"], 4))


if __name__ == "__main__":
    main()
