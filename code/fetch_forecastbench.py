"""Fetch resolved binary questions from ForecastBench public datasets.

Pulls every 2026 ForecastBench resolution_set + matching question_set, filters
to source ∈ {infer, metaculus}, resolution_date ≥ FB_MIN_RESOLVED, deduplicates
by question id (questions may appear in multiple rounds), enriches Metaculus
question text via the authenticated Metaculus API (since FB publishes only
title + URL pointer for Metaculus entries), and writes the result in the
canonical schema that build_ladder.py / elicit.py / analyze.py consume.

Pre-registration (see PREREGISTRATION.md):
  * binding training-data cutoff across panel: 2026-01
  * resolution-date filter: 2026-03-01 (one-month buffer past worst-case)
  * primary pool: INFER (~9) + Metaculus enriched (~31), target n=40
  * boundary pool: ACLED single-round, target n≈40

Environment:
  RUN_NAME            (default: forecastbench) -> writes to data/<RUN_NAME>/
  FB_SOURCE           (default: "infer,metaculus") comma-separated
  FB_MIN_RESOLVED     (default: 2026-03-01) ISO date filter on resolution_date
  PILOT_N_Q           (from config) target question count
  METACULUS_TOKEN     required if metaculus is in FB_SOURCE
  FB_ACLED_ROUND      (default: latest) round date for ACLED single-round fetch

Run:
  RUN_NAME=forecastbench python fetch_forecastbench.py
  RUN_NAME=acled FB_SOURCE=acled FB_MIN_RESOLVED=2026-03-01 python fetch_forecastbench.py
"""
from __future__ import annotations
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from config import DATA, TARGET_N_QUESTIONS

FB_RAW_BASE = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/datasets"
FB_TREE_API = "https://api.github.com/repos/forecastingresearch/forecastbench-datasets/git/trees/main?recursive=1"

METACULUS_TOKEN = os.environ.get("METACULUS_TOKEN", "").strip()
METACULUS_QUESTION_URL = "https://www.metaculus.com/api2/questions/{id}/"

FB_SOURCES = [s.strip() for s in os.environ.get("FB_SOURCE", "infer,metaculus").split(",") if s.strip()]
FB_MIN_RESOLVED = os.environ.get("FB_MIN_RESOLVED", "2026-03-01").strip()
FB_ACLED_ROUND = os.environ.get("FB_ACLED_ROUND", "").strip()

QUESTIONS_RAW = DATA / "questions_raw.json"

# -------------------------- Repo traversal --------------------------

def _list_round_dates() -> List[str]:
    """Discover all (resolution_set, question_set) round dates available in
    the ForecastBench-datasets repo."""
    r = requests.get(FB_TREE_API, timeout=30)
    r.raise_for_status()
    tree = r.json()["tree"]
    res_dates: Set[str] = set()
    q_dates: Set[str] = set()
    for entry in tree:
        path = entry["path"]
        m = re.match(r"datasets/resolution_sets/(\d{4}-\d{2}-\d{2})_resolution_set\.json$", path)
        if m:
            res_dates.add(m.group(1))
            continue
        m = re.match(r"datasets/question_sets/(\d{4}-\d{2}-\d{2})-llm\.json$", path)
        if m:
            q_dates.add(m.group(1))
    return sorted(res_dates & q_dates)


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)


# -------------------------- Metaculus enrichment --------------------------

def _metaculus_enrich(qid: str) -> Optional[Dict[str, str]]:
    """Hit the Metaculus API for a question id and return
    (description, resolution_criteria, fine_print). The API nests
    question metadata under a `question` key; we look in both top-level
    and nested locations."""
    if not METACULUS_TOKEN:
        return None
    url = METACULUS_QUESTION_URL.format(id=qid)
    try:
        r = requests.get(
            url,
            headers={"Authorization": f"Token {METACULUS_TOKEN}"},
            timeout=30,
        )
    except requests.RequestException as e:
        print(f"  metaculus {qid}: request failed: {e}")
        return None
    if r.status_code != 200:
        print(f"  metaculus {qid}: HTTP {r.status_code}")
        return None
    try:
        d = r.json()
    except json.JSONDecodeError:
        return None
    q = d.get("question") or {}
    return {
        "description": (d.get("description") or q.get("description") or "").strip(),
        "resolution_criteria": (d.get("resolution_criteria") or q.get("resolution_criteria") or "").strip(),
        "fine_print": (q.get("fine_print") or "").strip(),
    }


# -------------------------- Schema normalisation --------------------------

def _normalize(fb_q: Dict[str, Any], res: Dict[str, Any],
               enrichment: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """Map a (ForecastBench question, resolution, optional Metaculus enrichment)
    tuple to the canonical schema consumed by build_ladder.py / elicit.py."""
    source = fb_q.get("source") or res.get("source")
    raw_id = str(fb_q.get("id") or res.get("id"))
    resolved_to = res.get("resolved_to")
    if resolved_to not in (0.0, 1.0):
        return None
    binary_res = "YES" if resolved_to == 1.0 else "NO"

    title = (fb_q.get("question") or "").strip()
    # Sources differ in where the substantive text lives.
    fb_background = (fb_q.get("background") or "").strip()
    fb_criteria = (fb_q.get("resolution_criteria") or "").strip()
    fb_market_criteria = (fb_q.get("market_info_resolution_criteria") or "").strip()
    if fb_market_criteria and fb_market_criteria.lower() not in ("n/a", ""):
        fb_criteria_text = fb_market_criteria
    else:
        fb_criteria_text = fb_criteria

    if enrichment:
        bg = enrichment["description"] or fb_background
        criteria = enrichment["resolution_criteria"] or fb_criteria_text
        if enrichment["fine_print"]:
            criteria = (criteria + "\n\nFine print: " + enrichment["fine_print"]).strip()
    else:
        bg = fb_background
        criteria = fb_criteria_text

    # ACLED / Wikipedia templates have {resolution_date} placeholders.
    resolve_date_str = res.get("resolution_date") or ""
    if "{resolution_date}" in title:
        title = title.replace("{resolution_date}", resolve_date_str)
    if "{resolution_date}" in bg:
        bg = bg.replace("{resolution_date}", resolve_date_str)
    if "{resolution_date}" in criteria:
        criteria = criteria.replace("{resolution_date}", resolve_date_str)

    if not title or len(title) < 20:
        return None
    # Require *some* substance. ACLED has rich backgrounds; Metaculus enriched
    # has rich descriptions; INFER has rich backgrounds. If a Metaculus entry
    # came back empty even after enrichment, skip it.
    if len(bg) < 100 and len(criteria) < 100:
        return None

    return {
        "id": f"forecastbench-{source}-{raw_id}",
        "title": title,
        "background": bg[:4000],
        "resolution_criteria": criteria[:1500],
        "binary_resolution": binary_res,
        "open_time": fb_q.get("market_info_open_datetime"),
        "close_time": fb_q.get("market_info_close_datetime"),
        "resolve_time": resolve_date_str + "T00:00:00+00:00" if resolve_date_str else None,
        "community_prediction_history": [],
        "url": fb_q.get("url") or "",
        "source": f"forecastbench-{source}",
        "volume": None,
        "unique_bettors": None,
    }


# -------------------------- Main fetch --------------------------

def fetch(target_n: int = TARGET_N_QUESTIONS) -> List[Dict[str, Any]]:
    cache = Path("/tmp/fb_cache")
    cache.mkdir(exist_ok=True)

    print(f"Filter: source ∈ {FB_SOURCES}, resolution_date ≥ {FB_MIN_RESOLVED}")
    all_round_dates = _list_round_dates()
    # Only 2026 rounds matter under the cutoff filter (anything earlier resolves
    # before our 2026-03-01 floor by construction).
    rounds = [d for d in all_round_dates if d >= "2026-01-01"]
    if FB_ACLED_ROUND:
        rounds = [FB_ACLED_ROUND] if FB_ACLED_ROUND in rounds else []
    print(f"Walking {len(rounds)} round(s): {rounds[:3]}...{rounds[-3:] if len(rounds) > 3 else ''}")

    # Per (source, qid), keep the LATEST resolution we see across all rounds
    # so we get the most-recent resolution_date for templated questions.
    by_qid: Dict[Tuple[str, str], Dict[str, Any]] = {}
    qmeta_by_id: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for rd in rounds:
        rdest = cache / f"res_{rd}.json"
        qdest = cache / f"qs_{rd}.json"
        try:
            _download(f"{FB_RAW_BASE}/resolution_sets/{rd}_resolution_set.json", rdest)
            _download(f"{FB_RAW_BASE}/question_sets/{rd}-llm.json", qdest)
        except requests.RequestException as e:
            print(f"  {rd}: download failed ({e}); skipping round")
            continue
        res_set = json.loads(rdest.read_text())
        q_set = json.loads(qdest.read_text())
        qmap = {(q["source"], str(q["id"])): q for q in q_set["questions"]}

        for r in res_set["resolutions"]:
            if not r.get("resolved"):
                continue
            if r.get("resolved_to") not in (0.0, 1.0):
                continue
            if r["source"] not in FB_SOURCES:
                continue
            if (r.get("resolution_date") or "") < FB_MIN_RESOLVED:
                continue
            key = (r["source"], str(r["id"]))
            prev = by_qid.get(key)
            if prev is None or (r.get("resolution_date") or "") > (prev.get("resolution_date") or ""):
                by_qid[key] = r
            if key in qmap and key not in qmeta_by_id:
                qmeta_by_id[key] = qmap[key]

    print(f"  collected {len(by_qid)} candidate (source, qid) pairs across rounds")

    # Enrich Metaculus questions through the API; budget a polite delay.
    enriched_pool: List[Dict[str, Any]] = []
    skipped = 0
    metaculus_enrich_count = 0
    for key, res in by_qid.items():
        src, qid = key
        fb_q = qmeta_by_id.get(key)
        if fb_q is None:
            # Resolution exists but question metadata isn't in any matched
            # question_set. Synthesize a stub so the {resolution_date}
            # substitution at least runs (mainly affects ACLED/Wikipedia).
            fb_q = {
                "id": qid, "source": src,
                "question": "", "background": "", "resolution_criteria": "",
                "market_info_open_datetime": None, "market_info_close_datetime": None,
                "url": "",
            }
        enrichment = None
        if src == "metaculus":
            enrichment = _metaculus_enrich(qid)
            metaculus_enrich_count += 1
            time.sleep(0.25)  # polite to Metaculus
        normalized = _normalize(fb_q, res, enrichment)
        if normalized is None:
            skipped += 1
            continue
        enriched_pool.append(normalized)

    print(f"  normalized {len(enriched_pool)} questions (skipped {skipped}; "
          f"metaculus enrichments attempted: {metaculus_enrich_count})")

    # Balance YES / NO so base rate cannot be trivially exploited.
    yeses = [q for q in enriched_pool if q["binary_resolution"] == "YES"]
    nos = [q for q in enriched_pool if q["binary_resolution"] == "NO"]
    print(f"  pre-balance: YES={len(yeses)} NO={len(nos)}")
    half = target_n // 2
    out = (yeses[:half] + nos[: target_n - half])[:target_n]
    return out


def main():
    qs = fetch()
    yes = sum(1 for q in qs if q["binary_resolution"] == "YES")
    no = sum(1 for q in qs if q["binary_resolution"] == "NO")
    print(f"\nFetched {len(qs)} ForecastBench questions. YES={yes} NO={no}")
    print(f"Writing to {QUESTIONS_RAW}")
    QUESTIONS_RAW.parent.mkdir(parents=True, exist_ok=True)
    QUESTIONS_RAW.write_text(json.dumps(qs, indent=2))


if __name__ == "__main__":
    main()
