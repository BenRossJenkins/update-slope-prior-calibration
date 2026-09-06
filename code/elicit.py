"""Elicit forecasts from a panel of LLMs across the 4-level evidence ladder.

For each (question, model, level) cell we draw SAMPLES_PER_CELL forecasts at a
fixed elicitation prompt. Forecasts are extracted as a probability in [0,1]
that the question resolves YES.

Outputs JSONL to data/elicitations.jsonl. Re-running is incremental: cells
already present in the file are skipped, so a partial run can be resumed.
"""
from __future__ import annotations
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from config import (
    QUESTIONS_LADDER,
    ELICITATIONS,
    LEVELS,
    SAMPLES_PER_CELL,
    MODELS,
)

ELICIT_PROMPT = """You are a careful probabilistic forecaster.

QUESTION: {title}

RESOLUTION CRITERIA: {criteria}

CONTEXT:
{context}

Give your forecast as a single probability that the question resolves YES.
Reply with exactly one line in this format and nothing else:

PROBABILITY: 0.XX

Where 0.XX is between 0.00 and 1.00.
"""

PROB_RE = re.compile(r"PROBABILITY[:\s]+([01]?\.\d+|0|1)", re.IGNORECASE)


def _parse_prob(text: str) -> Optional[float]:
    m = PROB_RE.search(text)
    if not m:
        # fallback: first decimal in [0,1] in the text
        for tok in re.findall(r"[01]?\.\d+", text):
            try:
                p = float(tok)
                if 0.0 <= p <= 1.0:
                    return p
            except ValueError:
                pass
        return None
    try:
        p = float(m.group(1))
        return p if 0.0 <= p <= 1.0 else None
    except ValueError:
        return None


def _build_prompt(q: Dict[str, Any], level: str) -> str:
    return ELICIT_PROMPT.format(
        title=q["title"],
        criteria=q.get("resolution_criteria", "")[:1500],
        context=q["ladder"][level] or "(no additional context)",
    )


_OPENAI_TIMEOUT = 120.0
_ANTHROPIC_TIMEOUT_S = 120.0


def _call_openai(model: str, prompt: str, sample_idx: int) -> str:
    from openai import OpenAI
    client = OpenAI(timeout=_OPENAI_TIMEOUT, max_retries=2)
    is_reasoning = model.startswith("o3") or model.startswith("gpt-5")
    if is_reasoning:
        # Reasoning models share their token budget between hidden thinking and
        # the visible reply. Give them enough headroom that the one-line answer
        # actually gets produced; cap reasoning at "low" effort so the budget
        # is dominated by output, not chain-of-thought.
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 4000,
            "reasoning_effort": "low",
        }
    else:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400,
            "temperature": 0.7 if sample_idx > 0 else 0.0,
        }
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def _call_anthropic(model: str, prompt: str, sample_idx: int) -> str:
    from anthropic import Anthropic
    import httpx
    client = Anthropic(
        timeout=httpx.Timeout(connect=15.0, read=_ANTHROPIC_TIMEOUT_S, write=15.0, pool=15.0),
        max_retries=2,
    )
    # Sonnet 4.6 and Haiku 4.5 frequently write multi-paragraph reasoning
    # before the final PROBABILITY line. 400 tokens was the original budget;
    # 2000 cut the failure rate from 33-39% to 7%; the residual ~7% are
    # cells where reasoning runs past 2000. 4000 should drain those.
    # Pricing is per actual output token so raising the cap is free for
    # cells that wouldn't have hit it anyway.
    kwargs = {
        "model": model,
        "max_tokens": 4000,
        "messages": [{"role": "user", "content": prompt}],
    }
    # opus-4-7 deprecates temperature; other Anthropic models still accept it.
    if not model.startswith("claude-opus-4"):
        kwargs["temperature"] = 0.7 if sample_idx > 0 else 0.0
    resp = client.messages.create(**kwargs)
    return resp.content[0].text


def _existing_cells(path: Path) -> set:
    """Only count cells with a successfully parsed probability as 'done'.

    Cells with `prob is None` (e.g. empty completions from reasoning models
    starved of output tokens) are retried on resume.
    """
    if not path.exists():
        return set()
    seen = set()
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("prob") is None:
                    continue
                seen.add((rec["question_id"], rec["model_api_id"], rec["level"], rec["sample"]))
            except Exception:  # noqa: BLE001
                pass
    return seen


def _elicit_one(
    provider: str,
    api_id: str,
    display_name: str,
    family: str,
    reasoning: bool,
    q: Dict[str, Any],
    level: str,
    sample: int,
) -> Dict[str, Any]:
    prompt = _build_prompt(q, level)
    try:
        if provider == "openai":
            text = _call_openai(api_id, prompt, sample)
        elif provider == "anthropic":
            text = _call_anthropic(api_id, prompt, sample)
        else:
            raise ValueError(provider)
        prob = _parse_prob(text or "")
    except Exception as e:  # noqa: BLE001
        return {
            "question_id": q["id"],
            "model_api_id": api_id,
            "model_display": display_name,
            "model_family": family,
            "reasoning": reasoning,
            "level": level,
            "sample": sample,
            "prob": None,
            "raw": f"ERROR: {e}",
            "ts": time.time(),
        }
    return {
        "question_id": q["id"],
        "model_api_id": api_id,
        "model_display": display_name,
        "model_family": family,
        "reasoning": reasoning,
        "level": level,
        "sample": sample,
        "prob": prob,
        "raw": text,
        "ts": time.time(),
    }


def main(max_workers: int = 6):
    questions: List[Dict[str, Any]] = json.loads(QUESTIONS_LADDER.read_text())
    seen = _existing_cells(ELICITATIONS)
    print(f"Resuming: {len(seen)} cells already complete.")

    jobs: List[Tuple] = []
    for q in questions:
        for model in MODELS:
            provider, api_id, display, family, reasoning = model
            for level in LEVELS:
                for sample in range(SAMPLES_PER_CELL):
                    key = (q["id"], api_id, level, sample)
                    if key in seen:
                        continue
                    jobs.append((provider, api_id, display, family, reasoning, q, level, sample))

    print(f"Running {len(jobs)} elicitations (max_workers={max_workers})")

    with ELICITATIONS.open("a") as f, ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_elicit_one, *j) for j in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            rec = fut.result()
            f.write(json.dumps(rec) + "\n")
            f.flush()

    print(f"Done. Appended results to {ELICITATIONS}")


if __name__ == "__main__":
    main()
