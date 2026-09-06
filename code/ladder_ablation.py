"""Block 3: regenerate 10 ladders with GPT-5 instead of Claude Opus 4.7,
re-elicit Claude Opus 4.7 on those new ladders, compare its per-question
slope on Opus-built vs GPT-5-built ladders. Tests whether Opus updates
unusually on Opus-authored ladders (same-family contamination).

Output: data/ladder_ablation.json with per-question Opus slopes under both
ladder authors + a paired-test summary.
"""
from __future__ import annotations
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from anthropic import Anthropic
from openai import OpenAI
import httpx
from scipy import stats
from tqdm import tqdm

from config import QUESTIONS_LADDER, DATA

OUT = DATA / "ladder_ablation.json"
N_QUESTIONS = 10
ELICIT_PANEL_MODEL = "claude-opus-4-7"

LADDER_PROMPT = """You are constructing a graded evidence ladder for a forecasting research study.

QUESTION: {title}

RESOLUTION CRITERIA: {criteria}

ACTUAL RESOLUTION (already known, post-hoc): {resolution}

BACKGROUND (was available at the time the question was open): {background}

Your task: produce four versions of evidence for a forecaster, varying in informativeness.
All four must contain ONLY factual statements that a fair observer would call true.
Do NOT include speculation, predictions, or the resolution itself.

L0 -- empty string (just the question is shown; no context).
L1 -- Neutral pre-evidence background (2-3 sentences). Must NOT hint at the resolution.
L2 -- Concise factual evidence (2-4 sentences) summarizing the most important
     publicly-available information that points toward the eventual resolution.
     Should be the kind of evidence a careful reader would have synthesized as the
     situation developed. Do NOT state the resolution.
L3 -- Rich evidence (1-2 paragraphs) with multiple corroborating facts plus a
     short factual analysis of what they imply. Still must NOT state the resolution.

Output valid JSON with exactly these keys: L0, L1, L2, L3."""

ELICIT_PROMPT = """You are a careful probabilistic forecaster.

QUESTION: {title}

RESOLUTION CRITERIA: {criteria}

CONTEXT:
{context}

Give your forecast as a single probability that the question resolves YES.
Reply with exactly one line in this format and nothing else:

PROBABILITY: 0.XX

Where 0.XX is between 0.00 and 1.00."""

PROB_RE = re.compile(r"PROBABILITY[:\s]+([01]?\.\d+|0|1)", re.IGNORECASE)


def _build_gpt5_ladder(client: OpenAI, q: dict) -> dict:
    prompt = LADDER_PROMPT.format(
        title=q["title"],
        criteria=(q.get("resolution_criteria") or "")[:1500],
        resolution=q["binary_resolution"],
        background=(q.get("background") or "")[:2500],
    )
    resp = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=4000,
        reasoning_effort="low",
    )
    text = resp.choices[0].message.content or ""
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        raise ValueError("no JSON in GPT-5 response")
    return json.loads(text[s:e+1])


def _elicit_opus(client: Anthropic, q: dict, level_text: str, samples: int = 3) -> List[float]:
    probs = []
    prompt = ELICIT_PROMPT.format(
        title=q["title"],
        criteria=(q.get("resolution_criteria") or "")[:1500],
        context=level_text or "(no additional context)",
    )
    for _ in range(samples):
        resp = client.messages.create(
            model=ELICIT_PANEL_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text or ""
        m = PROB_RE.search(text)
        if m:
            try:
                p = float(m.group(1))
                if 0.0 <= p <= 1.0:
                    probs.append(p)
            except ValueError:
                pass
    return probs


def main():
    all_q = json.loads(QUESTIONS_LADDER.read_text())
    # Pick 10 questions balanced YES/NO.
    yes_qs = [q for q in all_q if q["binary_resolution"] == "YES"][:5]
    no_qs = [q for q in all_q if q["binary_resolution"] == "NO"][:5]
    questions = yes_qs + no_qs
    print(f"Selected {len(questions)} questions for ladder ablation.")

    oa = OpenAI(timeout=180)
    an = Anthropic(timeout=httpx.Timeout(connect=15, read=180, write=15, pool=15), max_retries=2)

    out = []
    for q in tqdm(questions, desc="ablation"):
        # 1. Build GPT-5 ladder.
        try:
            g_ladder = _build_gpt5_ladder(oa, q)
            g_ladder["L0"] = ""
            for k in ("L0", "L1", "L2", "L3"):
                if k not in g_ladder:
                    raise ValueError(f"GPT-5 ladder missing {k}")
        except Exception as e:
            print(f"skip {q['id']} (ladder fail): {e}")
            continue

        # 2. Elicit Opus on both ladders.
        opus_orig: Dict[str, List[float]] = {}
        opus_gpt5: Dict[str, List[float]] = {}
        try:
            for lvl in ("L0", "L1", "L2", "L3"):
                opus_orig[lvl] = _elicit_opus(an, q, q["ladder"][lvl])
                opus_gpt5[lvl] = _elicit_opus(an, q, g_ladder[lvl])
        except Exception as e:
            print(f"skip {q['id']} (elicit fail): {e}")
            continue

        y = 1.0 if q["binary_resolution"] == "YES" else 0.0
        def slope_from(probs_by_lvl):
            xs, ys = [], []
            for i, lvl in enumerate(("L0", "L1", "L2", "L3")):
                vs = probs_by_lvl.get(lvl, [])
                if not vs:
                    return None
                mean_p = float(np.mean(vs))
                a = mean_p if y == 1.0 else 1.0 - mean_p
                xs.append(i); ys.append(a)
            return float(np.polyfit(xs, ys, 1)[0])

        s_orig = slope_from(opus_orig)
        s_gpt5 = slope_from(opus_gpt5)
        out.append({
            "question_id": q["id"], "title": q["title"][:100],
            "y": y, "slope_opus_ladder": s_orig, "slope_gpt5_ladder": s_gpt5,
        })

    # Summarize.
    pairs = [(r["slope_opus_ladder"], r["slope_gpt5_ladder"]) for r in out
             if r["slope_opus_ladder"] is not None and r["slope_gpt5_ladder"] is not None]
    if pairs:
        a, b = zip(*pairs)
        a = np.array(a); b = np.array(b)
        diff = a - b
        t = stats.ttest_rel(a, b)
        w = stats.wilcoxon(a, b) if len(pairs) >= 5 else None
        summary = {
            "n_pairs": len(pairs),
            "mean_slope_opus_ladder": float(np.mean(a)),
            "mean_slope_gpt5_ladder": float(np.mean(b)),
            "mean_diff": float(np.mean(diff)),
            "paired_t_statistic": float(t.statistic),
            "paired_t_p": float(t.pvalue),
            "wilcoxon_p": float(w.pvalue) if w else None,
        }
    else:
        summary = {"n_pairs": 0}

    payload = {"questions": out, "summary": summary}
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
