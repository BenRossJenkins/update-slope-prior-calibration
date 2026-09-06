"""Construct a 4-level evidence ladder for each question.

Levels:
  L0: question only, no context.
  L1: question + neutral pre-evidence background (no resolution-pointing info).
  L2: question + concise factual evidence pointing toward the resolution.
  L3: question + rich evidence (multiple facts + interpretation), still factual.

We use a separate "ladder builder" model (Claude Opus 4.7) so that the elicited
models are not asked to grade evidence they themselves produced. The ladder
builder is given the resolution outcome explicitly; the levels are designed to
contain only facts a fair evaluator would consider true post-resolution.

Output schema (data/questions_ladder.json):
[
  { ...question fields...,
    "ladder": {
      "L0": "...",
      "L1": "...",
      "L2": "...",
      "L3": "..."
    }
  }, ...
]
"""
from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, List

from anthropic import Anthropic
import httpx
from tqdm import tqdm

from config import (
    QUESTIONS_RAW,
    QUESTIONS_LADDER,
    EVIDENCE_BUILDER_MODEL,
)

# Explicit per-request timeout so a stuck connection cannot hang the whole run.
_HTTP_TIMEOUT = httpx.Timeout(connect=15.0, read=120.0, write=15.0, pool=15.0)

LADDER_PROMPT = """You are constructing a graded evidence ladder for a forecasting research study.

QUESTION: {title}

RESOLUTION CRITERIA: {criteria}

ACTUAL RESOLUTION (already known, post-hoc): {resolution}

BACKGROUND (was available at the time the question was open): {background}

Your task: produce four versions of evidence for a forecaster, varying in informativeness.
All four must contain ONLY factual statements that a fair observer would call true.
Do NOT include speculation, predictions, or the resolution itself.

L0 — empty string (just the question is shown; no context).
L1 — Neutral pre-evidence background (2-3 sentences). Must NOT hint at the resolution.
L2 — Concise factual evidence (2-4 sentences) summarizing the most important
     publicly-available information that points toward the eventual resolution.
     Should be the kind of evidence a careful reader would have synthesized as the
     situation developed. Do NOT state the resolution.
L3 — Rich evidence (1-2 paragraphs) with multiple corroborating facts plus a
     short factual analysis of what they imply. Still must NOT state the resolution.

Output valid JSON with exactly these keys: L0, L1, L2, L3.
"""

def build_ladder(client: Anthropic, q: Dict[str, Any], retries: int = 2) -> Dict[str, str]:
    prompt = LADDER_PROMPT.format(
        title=q["title"],
        criteria=q.get("resolution_criteria", "")[:1500],
        resolution=q["binary_resolution"],
        background=q.get("background", "")[:2500],
    )
    last_err = None
    for _ in range(retries + 1):
        try:
            kwargs = {
                "model": EVIDENCE_BUILDER_MODEL[1],
                "max_tokens": 1500,
                "messages": [{"role": "user", "content": prompt}],
            }
            # Some Anthropic models (e.g. opus-4-7) no longer accept temperature.
            if not EVIDENCE_BUILDER_MODEL[1].startswith("claude-opus-4"):
                kwargs["temperature"] = 0.0
            resp = client.messages.create(**kwargs)
            text = resp.content[0].text
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1:
                raise ValueError("no JSON found in response")
            ladder = json.loads(text[start : end + 1])
            for k in ("L0", "L1", "L2", "L3"):
                if k not in ladder:
                    raise ValueError(f"missing key {k}")
            ladder["L0"] = ""  # canonical
            return ladder
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5)
    raise RuntimeError(f"ladder build failed: {last_err}")


def main():
    questions: List[Dict[str, Any]] = json.loads(QUESTIONS_RAW.read_text())
    client = Anthropic(timeout=_HTTP_TIMEOUT, max_retries=2)
    out = []
    # Incremental write so partial progress survives a crash/hang.
    if QUESTIONS_LADDER.exists():
        try:
            existing = json.loads(QUESTIONS_LADDER.read_text())
            done_ids = {q["id"] for q in existing if "ladder" in q}
            out.extend(existing)
            print(f"Resuming: {len(done_ids)} ladders already built")
        except Exception:
            done_ids = set()
    else:
        done_ids = set()

    for q in tqdm(questions, desc="building ladders"):
        if q["id"] in done_ids:
            continue
        try:
            ladder = build_ladder(client, q)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {q['id']}: {e}")
            continue
        out.append({**q, "ladder": ladder})
        # Persist after each success so a hang doesn't lose progress.
        QUESTIONS_LADDER.write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(out)} ladder questions to {QUESTIONS_LADDER}")


if __name__ == "__main__":
    main()
