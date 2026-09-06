"""Build evidence ladders WITHOUT showing the resolution to the builder.

This is the blinded counterpart to build_ladder.py. The reviewer concern it
addresses: the original ladder builder is given the resolution outcome
explicitly and instructed not to leak it, but subtle leakage through fact
selection, phrasing, and hindsight framing is possible even with that
instruction. To produce a hindsight-clean ladder, we omit the
ACTUAL RESOLUTION field from the builder prompt entirely and rely on the
question, resolution criteria, and pre-question background.

Output schema is identical to build_ladder.py; output path is
data/<RUN_NAME>/questions_ladder_blinded.json so the original ladder is
preserved and the two can be compared.

To re-elicit on the blinded ladder, set:
  LADDER_FILE=questions_ladder_blinded.json
and re-run elicit.py + the downstream pipeline.
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
    DATA,
    EVIDENCE_BUILDER_MODEL,
)

QUESTIONS_LADDER_BLINDED = DATA / "questions_ladder_blinded.json"

_HTTP_TIMEOUT = httpx.Timeout(connect=15.0, read=120.0, write=15.0, pool=15.0)

LADDER_PROMPT_BLINDED = """You are constructing a graded evidence ladder for a forecasting research study.

QUESTION: {title}

RESOLUTION CRITERIA: {criteria}

BACKGROUND (was available at the time the question was open): {background}

Your task: produce four versions of evidence for a forecaster, varying in
informativeness. You are NOT told how this question resolved; build the
ladder using only the question, the resolution criteria, and the background
above, as a careful researcher would have at the time the question was open.

All four must contain ONLY factual statements that a fair observer at the time
of the question would have considered true. Do NOT include speculation,
predictions, or anything that would only be knowable post-resolution.

L0 -- empty string (just the question is shown; no context).
L1 -- Neutral pre-evidence background (2-3 sentences). Must be plausibly
      pre-resolution and must not assert a resolution direction.
L2 -- Concise factual evidence (2-4 sentences) summarizing the most
      important publicly-available pre-resolution information bearing on the
      question. Should be the kind of evidence a careful reader would have
      synthesized as the situation developed.
L3 -- Rich evidence (1-2 paragraphs) with multiple corroborating
      pre-resolution facts plus a short factual analysis of what they imply.
      Still must not state or strongly imply the resolution.

Output valid JSON with exactly these keys: L0, L1, L2, L3.
"""


def build_ladder_blinded(client: Anthropic, q: Dict[str, Any],
                          retries: int = 2) -> Dict[str, str]:
    prompt = LADDER_PROMPT_BLINDED.format(
        title=q["title"],
        criteria=q.get("resolution_criteria", "")[:1500],
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
            ladder["L0"] = ""
            return ladder
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5)
    raise RuntimeError(f"blinded ladder build failed: {last_err}")


def main():
    questions: List[Dict[str, Any]] = json.loads(QUESTIONS_RAW.read_text())
    client = Anthropic(timeout=_HTTP_TIMEOUT, max_retries=2)
    out = []
    if QUESTIONS_LADDER_BLINDED.exists():
        try:
            existing = json.loads(QUESTIONS_LADDER_BLINDED.read_text())
            done_ids = {q["id"] for q in existing if "ladder" in q}
            out.extend(existing)
            print(f"Resuming: {len(done_ids)} blinded ladders already built")
        except Exception:
            done_ids = set()
    else:
        done_ids = set()

    for q in tqdm(questions, desc="building blinded ladders"):
        if q["id"] in done_ids:
            continue
        try:
            ladder = build_ladder_blinded(client, q)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {q['id']}: {e}")
            continue
        out.append({**q, "ladder": ladder})
        QUESTIONS_LADDER_BLINDED.write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(out)} blinded ladder questions to {QUESTIONS_LADDER_BLINDED}")


if __name__ == "__main__":
    main()
