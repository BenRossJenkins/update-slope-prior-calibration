"""Score original vs blinded Manifold ladders against the pre-committed
re-elicitation decision rule. See RE_ELICITATION_DECISION_RULE.md.

For each of the 40 Manifold questions, score L2 (the load-bearing rung for
the V-A confirmatory test) on three leakage dimensions:

  1. hindsight-marker phrases ("as it turned out", "ultimately", past
     tense for outcomes that were future at question open, etc.)
  2. date-of-statement leakage (mention of dates between open and
     resolution that themselves imply the outcome or a precursor)
  3. evidence-selection asymmetry (a resolution-blind reader would
     predict the realized direction at >70% from reading L2 alone)

We score automatically on (1) and (2) via regex/keyword matching, and on
(3) by asking a separate LLM (resolution-blind) to predict the resolution
from L2 alone.

Output: tabular per-question scores plus aggregate decision.
"""
from __future__ import annotations
import json
import re
import os
import time
from pathlib import Path
from typing import Dict, List

from anthropic import Anthropic
import httpx

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ORIG_PATH = DATA / "questions_ladder.json"
BLIND_PATH = DATA / "questions_ladder_blinded.json"
OUT_PATH = DATA / "blinded_comparison.json"

# Dimension 1: hindsight phrases (case-insensitive whole-word-ish match).
HINDSIGHT_PHRASES = [
    r"\bas it turned out\b",
    r"\bultimately\b",
    r"\bin the event\b",
    r"\beventually\b",
    r"\bturned out to\b",
    r"\bended up\b",
    r"\bwound up\b",
    r"\bwent on to\b",
    r"\bsubsequently (?:announced|launched|signed|happened|occurred|resigned|stepped down|fired|appointed)\b",
    r"\bafter all\b",
    r"\bas of [a-z]+ \d{4}\b",   # "as of June 2025"
    r"\bnow \w+",                 # weak; flag and adjudicate
    r"\bsince then\b",
    r"\blater that (?:year|month|week|day)\b",
    r"\bsubsequently\b",
]

def score_hindsight(text: str) -> int:
    text_lower = text.lower()
    return int(any(re.search(p, text_lower) for p in HINDSIGHT_PHRASES))


# Dimension 2: date leakage.
DATE_PATTERNS = [
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december) \d{1,2},? \d{4}\b",
    r"\b\d{1,2} (?:january|february|march|april|may|june|july|august|september|october|november|december) \d{4}\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
]

def score_dates(text: str, question_open: str = "", question_resolved: str = "") -> int:
    text_lower = text.lower()
    matches = []
    for p in DATE_PATTERNS:
        matches.extend(re.findall(p, text_lower))
    # We can't easily compare dates to open/resolved without parsing every
    # question; flag presence of any specific date as a heuristic, since
    # the original ladder builder shouldn't be naming specific dates in a
    # pre-resolution-style evidence summary.
    return int(bool(matches))


# Dimension 3: resolution-blind prediction.
PREDICT_PROMPT = """You are a careful forecaster. Read the question and the evidence below. Output ONLY a single number between 0 and 1 representing the probability that the question resolves YES, based only on the evidence shown. Do not output anything else.

QUESTION: {title}

EVIDENCE: {evidence}

PROBABILITY:"""


def blind_predict(client: Anthropic, title: str, evidence: str,
                  model: str = "claude-haiku-4-5-20251001",
                  retries: int = 2) -> float:
    if not evidence.strip():
        return 0.5
    prompt = PREDICT_PROMPT.format(title=title, evidence=evidence[:3000])
    for _ in range(retries + 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=20,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            m = re.search(r"\d?\.\d+|\d+", text)
            if m:
                p = float(m.group(0))
                if p > 1: p /= 100.0
                return max(0.001, min(0.999, p))
        except Exception:
            time.sleep(1.0)
    return 0.5


def main():
    orig = {q["id"]: q for q in json.loads(ORIG_PATH.read_text())}
    blind = {q["id"]: q for q in json.loads(BLIND_PATH.read_text())}

    ids = sorted(set(orig.keys()) & set(blind.keys()))
    print(f"Comparing {len(ids)} questions present in both ladders")

    client = Anthropic(timeout=httpx.Timeout(connect=15.0, read=60.0, write=15.0, pool=15.0),
                       max_retries=2)

    rows = []
    for qid in ids:
        q_o = orig[qid]
        q_b = blind[qid]
        l2_o = (q_o.get("ladder") or {}).get("L2", "")
        l2_b = (q_b.get("ladder") or {}).get("L2", "")
        title = q_o.get("title", "")
        # binary_resolution can be "YES"/"NO" string or 0/1
        raw_res = q_o.get("binary_resolution", 0)
        if isinstance(raw_res, str):
            resolution = 1 if raw_res.strip().upper() == "YES" else 0
        else:
            resolution = int(raw_res)

        # Dimensions 1, 2: regex-based
        h_o = score_hindsight(l2_o)
        h_b = score_hindsight(l2_b)
        d_o = score_dates(l2_o)
        d_b = score_dates(l2_b)

        # Dimension 3: resolution-blind prediction from L2 only
        p_o = blind_predict(client, title, l2_o)
        p_b = blind_predict(client, title, l2_b)
        # asymmetry: prediction matches realized resolution at >0.7
        a_o = int((resolution == 1 and p_o > 0.7) or (resolution == 0 and p_o < 0.3))
        a_b = int((resolution == 1 and p_b > 0.7) or (resolution == 0 and p_b < 0.3))

        rows.append({
            "id": qid, "title": title[:80], "resolution": resolution,
            "hindsight_orig": h_o, "hindsight_blind": h_b,
            "dates_orig": d_o, "dates_blind": d_b,
            "blind_pred_orig": round(p_o, 3),
            "blind_pred_blind": round(p_b, 3),
            "asym_orig": a_o, "asym_blind": a_b,
        })

    # Aggregate: leakage-removed-by-blinding per dimension
    h_removed = sum(r["hindsight_orig"] and not r["hindsight_blind"] for r in rows)
    d_removed = sum(r["dates_orig"] and not r["dates_blind"] for r in rows)
    a_removed = sum(r["asym_orig"] and not r["asym_blind"] for r in rows)

    n = len(rows)
    aggregate_rate = (h_removed + d_removed + a_removed) / (3 * n)
    asym_rate = a_removed / n

    decision = {
        "n_questions": n,
        "hindsight_removed": h_removed,
        "dates_removed": d_removed,
        "asymmetry_removed": a_removed,
        "aggregate_leakage_rate": round(aggregate_rate, 4),
        "asymmetry_alone_rate": round(asym_rate, 4),
        "decision_threshold_aggregate": 0.20,
        "decision_threshold_asym": 0.30,
        "trigger_aggregate": aggregate_rate >= 0.20,
        "trigger_asym": asym_rate >= 0.30,
        "decision": "RE_ELICIT" if (aggregate_rate >= 0.20 or asym_rate >= 0.30)
                    else "DO_NOT_RE_ELICIT (claim minimal leakage)",
    }
    OUT_PATH.write_text(json.dumps({"rows": rows, "decision": decision}, indent=2))
    print("\n=== AGGREGATE ===")
    print(json.dumps(decision, indent=2))
    print(f"\nFull comparison written to {OUT_PATH}")


if __name__ == "__main__":
    main()
