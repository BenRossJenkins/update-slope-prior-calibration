"""DRR leaderboard re-ranking + Resolution-based DRR variant.

  L1. Leaderboard comparison: rank models by (i) Brier@L3, (ii) naive
      slope, (iii) DRR. Report Kendall tau / Spearman rho between pairs
      and the specific pairs that get re-ranked.
  L2. DRR_Res: residual of per-model Resolution gain (L0 -> L3) on
      Resolution@L0 (or on sqrt(Brier@L0)) -- "dose-response residual
      on the resolution component" rather than raw slope. Tighter
      interpretation because Resolution is the discrimination part of
      Brier that updating should actually modify.

Output -> results.json under robustness.drr_extensions.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config import RESULTS


def main():
    R = json.loads(Path(RESULTS).read_text())
    models = R["models"]
    bd_summary = R.get("robustness", {}).get("brier_decomp", {}).get("summary", {}) or {}
    bd_rows = bd_summary.get("per_model", []) or []
    bd_by_model = {row["model"]: row for row in bd_rows}

    drr_v1 = R.get("robustness", {}).get("v2", {}).get("drr", {}).get("per_model", {}) or {}

    # ---------- L1: leaderboard comparison ----------
    rows = []
    for api_id, m in models.items():
        bd = bd_by_model.get(api_id, {})
        rows.append({
            "model": api_id,
            "display": m.get("display", api_id),
            "reasoning": m.get("reasoning"),
            "brier_L3": m.get("brier_L3"),
            "slope": m.get("mean_slope"),
            "DRR_slope": drr_v1.get(api_id, {}).get("DRR"),
            "rel_L0": bd.get("rel_L0"),
            "rel_L3": bd.get("rel_L3"),
            "res_L0": bd.get("res_L0"),
            "res_L3": bd.get("res_L3"),
            "rel_reduction": bd.get("rel_reduction"),
            "res_gain": bd.get("res_gain"),
        })
    df = pd.DataFrame(rows).dropna(subset=["brier_L3", "slope", "DRR_slope", "res_gain", "res_L0"])

    # ---------- L2: DRR_Res ----------
    # Residualize res_gain on sqrt(rel_L0) (poor priors -> more room to improve resolution).
    # Or on res_L0 itself; we'll use sqrt(brier_L0) for parity with DRR_slope.
    df["sqrt_brier_L0"] = np.sqrt(np.array([models[m]["brier_L0"] for m in df["model"]]))
    X = np.column_stack([np.ones(len(df)), df["sqrt_brier_L0"].values])
    beta, *_ = np.linalg.lstsq(X, df["res_gain"].values, rcond=None)
    df["res_gain_predicted"] = X @ beta
    df["DRR_Res"] = df["res_gain"] - df["res_gain_predicted"]
    drr_res_ols_coef = float(beta[1])

    # ---------- Rankings ----------
    ranks = {}
    for key in ("brier_L3", "slope", "DRR_slope", "DRR_Res"):
        # lower Brier = better (rank 1); higher slope/DRR = better updater (rank 1).
        ascending = (key == "brier_L3")
        df[f"rank_{key}"] = df[key].rank(ascending=ascending, method="min")
        ranks[key] = df.set_index("display")[f"rank_{key}"].to_dict()

    # Pairwise rank correlations.
    pairs = [("brier_L3", "slope"), ("brier_L3", "DRR_slope"), ("brier_L3", "DRR_Res"),
             ("slope", "DRR_slope"), ("DRR_slope", "DRR_Res")]
    pair_stats = {}
    for a, b in pairs:
        ra = df[f"rank_{a}"].values
        rb = df[f"rank_{b}"].values
        kt = stats.kendalltau(ra, rb)
        sp = stats.spearmanr(ra, rb)
        pair_stats[f"{a}__vs__{b}"] = {
            "kendall_tau": float(kt.statistic), "kendall_p": float(kt.pvalue),
            "spearman_r": float(sp.statistic), "spearman_p": float(sp.pvalue),
        }

    # Identify re-ranked pairs (any pair whose order flips between two metrics).
    def _flipped(a_key, b_key):
        out = []
        models_list = df["display"].tolist()
        for i in range(len(models_list)):
            for j in range(i + 1, len(models_list)):
                mi, mj = models_list[i], models_list[j]
                a_i, a_j = df.iloc[i][f"rank_{a_key}"], df.iloc[j][f"rank_{a_key}"]
                b_i, b_j = df.iloc[i][f"rank_{b_key}"], df.iloc[j][f"rank_{b_key}"]
                if (a_i < a_j) != (b_i < b_j):
                    out.append((mi, mj))
        return out
    flipped = {f"{a}__vs__{b}": _flipped(a, b) for a, b in pairs}

    # Pack rows for table emission.
    table_rows = []
    for _, r in df.sort_values("DRR_Res", ascending=False).iterrows():
        table_rows.append({
            "display": r["display"],
            "reasoning": bool(r["reasoning"]),
            "brier_L3": float(r["brier_L3"]),
            "rank_brier_L3": int(r["rank_brier_L3"]),
            "slope": float(r["slope"]),
            "rank_slope": int(r["rank_slope"]),
            "DRR_slope": float(r["DRR_slope"]),
            "rank_DRR_slope": int(r["rank_DRR_slope"]),
            "res_gain": float(r["res_gain"]),
            "DRR_Res": float(r["DRR_Res"]),
            "rank_DRR_Res": int(r["rank_DRR_Res"]),
        })

    out = {
        "drr_res_ols_coef_sqrt_brier_L0": drr_res_ols_coef,
        "ranks": ranks,
        "pair_stats": pair_stats,
        "flipped_pairs": flipped,
        "table_rows": table_rows,
    }
    R.setdefault("robustness", {})["drr_extensions"] = out
    Path(RESULTS).write_text(json.dumps(R, indent=2))

    # Print
    print("=== Per-model leaderboard ===")
    print(f"{'model':<22} {'B@L3':>6} {'rank':>4}  {'slope':>7} {'rank':>4}  "
          f"{'DRR':>7} {'rank':>4}  {'DRR_Res':>8} {'rank':>4}")
    for t in table_rows:
        print(f"{t['display']:<22} {t['brier_L3']:>6.3f} {t['rank_brier_L3']:>4}  "
              f"{t['slope']:>7.4f} {t['rank_slope']:>4}  "
              f"{t['DRR_slope']:>+7.4f} {t['rank_DRR_slope']:>4}  "
              f"{t['DRR_Res']:>+8.4f} {t['rank_DRR_Res']:>4}")
    print()
    print(f"DRR_Res OLS coef on sqrt(Brier@L0): {drr_res_ols_coef:+.4f}")
    print()
    print("=== Pair rank statistics ===")
    for k, v in pair_stats.items():
        print(f"  {k}: kendall={v['kendall_tau']:+.3f} (p={v['kendall_p']:.3g})  "
              f"spearman={v['spearman_r']:+.3f}")
    print()
    print("=== Flipped pairs ===")
    for k, pairs_ in flipped.items():
        if pairs_:
            print(f"  {k}:")
            for a, b in pairs_:
                print(f"    {a}  <->  {b}")


if __name__ == "__main__":
    main()
