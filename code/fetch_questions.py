"""Fetch resolved binary forecasting questions from Manifold Markets.

Manifold has a public, unauthenticated API. We pull recent resolved
binary markets, filter for substantive questions (high volume, real
question text, YES/NO resolution, resolved post-cutoff), then fetch
full descriptions for the kept set.

Output schema: same as before — see paper Sec. 4.
"""
from __future__ import annotations
import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from config import QUESTIONS_RAW, TARGET_N_QUESTIONS, RESOLUTION_AFTER

SEARCH_URL = "https://api.manifold.markets/v0/search-markets"
MARKET_URL = "https://api.manifold.markets/v0/market"

MIN_VOLUME = 1000.0
MIN_UNIQUE_BETTORS = 25
MIN_QUESTION_LEN = 40
MIN_DESCRIPTION_LEN = 150

# Patterns that flag low-quality markets we want to skip.
_FIRST_PERSON_STARTS = (
    "i ", "i'll", "i'm", "i've", "my ", "me ", "will i ",
    "should i", "am i ", "did i ",
)
# Sports markets are typically formatted "<Team> - <Team> <date>".
_SPORTS_RE = re.compile(r"^[A-Z][a-zA-Z ]+\s+[-–vs]+\s+[A-Z][a-zA-Z ]+")
# Personal/banal patterns in titles.
_PERSONAL_RE = re.compile(r"\b(this week|today|tonight|tomorrow|by friday)\b", re.IGNORECASE)

_after_ms = int(datetime.fromisoformat(RESOLUTION_AFTER + "T00:00:00+00:00").timestamp() * 1000)


def _is_low_quality(title: str) -> bool:
    t = title.strip().lower()
    if any(t.startswith(p) for p in _FIRST_PERSON_STARTS):
        return True
    if _SPORTS_RE.match(title):
        return True
    if _PERSONAL_RE.search(title):
        # Allow if title also mentions a public event/entity (uppercase 4+ letter token outside the personal phrase).
        if not re.search(r"\b(election|GDP|Fed|Court|Russia|Ukraine|China|Trump|Biden|Harris|OpenAI|Anthropic|SpaceX|NASA|AI |GPT|FDA|SEC)\b", title, re.IGNORECASE):
            return True
    return False


def _candidates(want: int) -> List[Dict[str, Any]]:
    """Page through resolved binary markets sorted by resolve-date desc.

    Manifold caps offset at 1000, so we paginate offset 0..1000 (~1100 markets
    seen) and then stop. With strict filters this yields plenty of candidates
    when there is a healthy backlog of resolved markets.
    """
    out: List[Dict[str, Any]] = []
    offset = 0
    pages_seen = 0
    while len(out) < want and offset <= 1000:
        params = {
            "term": "",
            "sort": "resolve-date",
            "filter": "resolved",
            "contractType": "BINARY",
            "limit": 100,
            "offset": offset,
        }
        r = requests.get(SEARCH_URL, params=params, timeout=15)
        r.raise_for_status()
        page = r.json()
        if not isinstance(page, list):
            # Manifold returns {"message": "..."} when offset exceeds the cap.
            print(f"  Manifold paging halted at offset={offset}: {page}")
            break
        if not page:
            break
        for m in page:
            res = m.get("resolution")
            if res not in ("YES", "NO"):
                continue
            vol = m.get("volume") or 0
            if vol < MIN_VOLUME:
                continue
            bettors = m.get("uniqueBettorCount") or 0
            if bettors < MIN_UNIQUE_BETTORS:
                continue
            q = (m.get("question") or "").strip()
            if len(q) < MIN_QUESTION_LEN:
                continue
            if _is_low_quality(q):
                continue
            rt_ms = m.get("resolutionTime")
            if not rt_ms or rt_ms < _after_ms:
                continue
            out.append(m)
        pages_seen += 1
        print(f"  page {pages_seen} (offset {offset}): {len(out)} candidates so far")
        offset += 100
        time.sleep(0.15)
    return out


def _full(market_id: str) -> Optional[Dict[str, Any]]:
    r = requests.get(f"{MARKET_URL}/{market_id}", timeout=30)
    if r.status_code != 200:
        return None
    return r.json()


def _description_text(d: Any) -> str:
    """Manifold returns description as a TipTap doc dict or a string."""
    if isinstance(d, str):
        return d.strip()
    if isinstance(d, dict):
        out: List[str] = []
        def walk(node):
            if isinstance(node, dict):
                if node.get("type") == "text" and node.get("text"):
                    out.append(node["text"])
                for child in node.get("content") or []:
                    walk(child)
        walk(d)
        return " ".join(out).strip()
    return ""


def _normalize(full: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    res = full.get("resolution")
    if res not in ("YES", "NO"):
        return None
    rt = full.get("resolutionTime")
    iso_resolve = (
        datetime.fromtimestamp(rt / 1000, tz=timezone.utc).isoformat()
        if rt else None
    )
    iso_close = (
        datetime.fromtimestamp(full["closeTime"] / 1000, tz=timezone.utc).isoformat()
        if full.get("closeTime") else None
    )
    iso_open = (
        datetime.fromtimestamp(full["createdTime"] / 1000, tz=timezone.utc).isoformat()
        if full.get("createdTime") else None
    )
    desc = _description_text(full.get("description"))[:4000]
    return {
        "id": f"manifold-{full['id']}",
        "title": full["question"].strip(),
        "background": desc,
        "resolution_criteria": "",  # Manifold encodes this in the description; left empty
        "binary_resolution": res,
        "open_time": iso_open,
        "close_time": iso_close,
        "resolve_time": iso_resolve,
        "community_prediction_history": [],
        "url": full.get("url") or "",
        "source": "manifold",
        "volume": full.get("volume"),
        "unique_bettors": full.get("uniqueBettorCount"),
    }


def fetch(target_n: int = TARGET_N_QUESTIONS) -> List[Dict[str, Any]]:
    cand = _candidates(want=target_n * 3)
    print(f"  fetching full details for up to {len(cand)} candidates...")
    out: List[Dict[str, Any]] = []
    seen_ids = set()
    for i, m in enumerate(cand):
        if m["id"] in seen_ids:
            continue
        seen_ids.add(m["id"])
        full = _full(m["id"])
        if not full:
            continue
        n = _normalize(full)
        if n is None:
            continue
        if len(n["background"]) < MIN_DESCRIPTION_LEN:
            continue
        if _is_low_quality(n["title"]):
            continue
        out.append(n)
        if (i + 1) % 20 == 0:
            print(f"    processed {i + 1}/{len(cand)} candidates, kept {len(out)}")
        if len(out) >= target_n * 2:
            break
        time.sleep(0.05)
    # Balance YES/NO so the panel can't trivially exploit base rate.
    yeses = [q for q in out if q["binary_resolution"] == "YES"]
    nos = [q for q in out if q["binary_resolution"] == "NO"]
    half = target_n // 2
    out = (yeses[:half] + nos[:target_n - half])[:target_n]
    return out


def main():
    qs = fetch()
    yes = sum(1 for q in qs if q["binary_resolution"] == "YES")
    no = sum(1 for q in qs if q["binary_resolution"] == "NO")
    print(f"Fetched {len(qs)} resolved binary questions. YES={yes}  NO={no}")
    QUESTIONS_RAW.write_text(json.dumps(qs, indent=2))
    print(f"Wrote {QUESTIONS_RAW}")


if __name__ == "__main__":
    main()
