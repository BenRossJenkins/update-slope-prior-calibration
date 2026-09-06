"""Fetch resolved binary forecasting questions from Polymarket.

Polymarket exposes market metadata via the Gamma API
(https://gamma-api.polymarket.com/markets). We pull recent resolved
binary markets ordered by closedTime desc, apply the same
substantive-question filters as the Manifold fetcher, and produce
output in the IDENTICAL schema so `build_ladder.py`, `elicit.py`,
and `analyze.py` run unchanged on the result.

DEFAULT OUTPUT PATH: data/questions_raw_polymarket.json
(distinct from Manifold's `questions_raw.json` to avoid clobbering).

TO USE FOR A POLYMARKET REPLICATION RUN:
  # 1. fetch
  python fetch_polymarket.py
  # 2. swap input file
  cp data/questions_raw_polymarket.json data/questions_raw.json
  # 3. run downstream stages (unchanged)
  python run.py --skip-fetch

Or set PYM_OUTPUT_PATH env var to override the output path.

Filter parity with fetch_questions.py:
  * resolved, binary (Yes/No outcome pair)
  * volume >= MIN_VOLUME (USD)
  * description length >= MIN_DESCRIPTION_LEN
  * title length >= MIN_QUESTION_LEN
  * resolved post RESOLUTION_AFTER
  * not sports / first-person / banal
  * balanced YES/NO 50/50
"""
from __future__ import annotations
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from config import DATA, TARGET_N_QUESTIONS, RESOLUTION_AFTER

MARKETS_URL = "https://gamma-api.polymarket.com/markets"

# Polymarket volume is in USD, not Manifold mana. $1000 traded is a meaningful
# threshold; tighten with PYM_MIN_VOLUME if you want stricter filtering.
MIN_VOLUME = float(os.environ.get("PYM_MIN_VOLUME", 1000.0))
MIN_QUESTION_LEN = 40
MIN_DESCRIPTION_LEN = 150
# Minimum days a market was open before it resolved. Daily crypto/weather/sports
# lottery markets dominate Polymarket's recent-resolved feed; a real forecasting
# question needs at least a week of open trading to have non-trivial evidence
# accumulation. Tighten via PYM_MIN_DAYS.
MIN_OPEN_DAYS = int(os.environ.get("PYM_MIN_DAYS", 14))
# Skip lopsided lottery markets: resolved price > LOTTERY_PRICE_THRESHOLD on
# the winning side AND closing-time price near same suggests the market was
# always near 0/1, not a real forecasting question. We don't have closing
# prices in this endpoint, so we approximate by ratio of 24h volume to
# total volume (a market with all action in the final hours is more likely
# to be a real updating question).
DEFAULT_OUTPUT_PATH = DATA / "questions_raw_polymarket.json"
OUTPUT_PATH = (
    DATA / os.environ["PYM_OUTPUT_PATH"]
    if os.environ.get("PYM_OUTPUT_PATH")
    else DEFAULT_OUTPUT_PATH
)

_FIRST_PERSON_STARTS = (
    "i ", "i'll", "i'm", "i've", "my ", "me ", "will i ",
    "should i", "am i ", "did i ",
)
_PERSONAL_RE = re.compile(r"\b(this week|today|tonight|tomorrow|by friday)\b", re.IGNORECASE)
# Polymarket sports market patterns: title often "TeamA vs TeamB ..." or
# "X Total Y Over/Under Z" or series.title like "Dota 2", "NBA", etc.
# Unicode-aware to catch international team names (Avaí FC vs. Cuiabá EC).
_SPORTS_TEAM_VS_RE = re.compile(
    r"^[\w][\w\s.'-]+\s+(?:vs\.?|v\.?|@|-)\s+[\w][\w\s.'-]+",
    re.UNICODE,
)
# Weather/event markets that are shallow forecasting questions.
_WEATHER_RE = re.compile(
    r"\b(temperature|degrees?|°[CF]|rainfall|snowfall|hurricane|tornado|wildfire)\b",
    re.IGNORECASE,
)
# Lottery-style markets that often slip through.
_LOTTERY_RE = re.compile(
    r"\b(odds|coin\s*flip|halftime|both\s+teams\s+to\s+score|btts|over\s*/\s*under|spread)\b",
    re.IGNORECASE,
)
# Crypto / index price-band markets ("Will Bitcoin be between $X and $Y on date"
# or "above $X on date"). These resolve on a fixed-window price band that the
# LLM has no real reasoning advantage on.
_PRICE_BAND_RE = re.compile(
    r"price of (bitcoin|btc|ethereum|eth|xrp|solana|sol|doge|dogecoin|sp500|s&p)|"
    r"will (bitcoin|btc|ethereum|eth|xrp) (be|hit|reach|trade)|"
    r"\bbe (above|below|between) \$",
    re.IGNORECASE,
)
# Title-level sports filter. Catches markets where the parent event/series tag
# doesn't carry a sports slug (NBA Finals, Champions League etc. sometimes
# live under a generic "politics" or per-team series).
_SPORTS_TITLE_RE = re.compile(
    r"\b(NBA|NFL|MLB|NHL|UEFA|FIFA|EPL|MLS|"
    r"Champions League|Europa League|Premier League|Bundesliga|La Liga|Serie A|Ligue 1|"
    r"World Cup|Super Bowl|World Series|Stanley Cup|"
    r"Eastern Conference|Western Conference|"
    r"Wimbledon|US Open|French Open|Australian Open|Masters|PGA|"
    r"Heisman|Tour de France|Formula 1|F1 Grand Prix|"
    r"NCAA|March Madness|Final Four|"
    r"Olympics|Paralympics)\b",
    re.IGNORECASE,
)
# Pop culture / entertainment markets. Character-death-in-season-X etc. — not
# really forecasting questions that benefit from evidence.
_POP_CULTURE_RE = re.compile(
    r"\b(Stranger Things|Game of Thrones|House of the Dragon|Severance|"
    r"Succession|White Lotus|Yellowstone|Squid Game|Last of Us|"
    r"Marvel|MCU|DC Comics|Star Wars|"
    r"season \d|episode \d|"
    r"Oscars?|Academy Award|Emmys?|Grammys?|Golden Globes?|BAFTAs?|"
    r"Eurovision|MTV VMA|People's Choice|"
    r"Time Person of the Year)\b",
    re.IGNORECASE,
)
_SPORTS_SERIES_KEYWORDS = {
    # Major league/series slugs Polymarket categorises markets under.
    "nba", "nfl", "mlb", "nhl", "soccer", "ufc", "mma", "boxing", "tennis",
    "golf", "f1", "formula-1", "cricket", "wnba", "cs2", "csgo",
    "counter-strike", "valorant", "dota-2", "dota2", "lol", "league-of-legends",
    "esports", "kbo", "premier-league", "champions-league",
    "rugby", "esl", "iem", "uefa", "fifa", "ncaa", "ncaab", "ncaaf",
    "futebol", "brasileiro", "bundesliga", "la-liga", "serie-a", "ligue-1",
    "europa", "europa-league", "conference-league", "epl", "mls",
    "soccer-tournaments", "tournament", "playoff", "playoffs", "sports",
    "weather", "temperatures", "hurricane", "atp", "wta",
}

_after_iso = RESOLUTION_AFTER + "T00:00:00+00:00"
_after_dt = datetime.fromisoformat(_after_iso)


def _is_low_quality_title(title: str) -> bool:
    t = title.strip().lower()
    if any(t.startswith(p) for p in _FIRST_PERSON_STARTS):
        return True
    if _SPORTS_TEAM_VS_RE.match(title):
        return True
    if _SPORTS_TITLE_RE.search(title):
        return True
    if _POP_CULTURE_RE.search(title):
        return True
    if _WEATHER_RE.search(title):
        return True
    if _LOTTERY_RE.search(title):
        return True
    if _PRICE_BAND_RE.search(title):
        return True
    if _PERSONAL_RE.search(title):
        # Allow if title also mentions a public event/entity.
        if not re.search(
            r"\b(election|GDP|Fed|Court|Russia|Ukraine|China|Trump|Biden|Harris|OpenAI|Anthropic|SpaceX|NASA|AI |GPT|FDA|SEC)\b",
            title,
            re.IGNORECASE,
        ):
            return True
    return False


def _is_sports_event(market: Dict[str, Any]) -> bool:
    """Use events[].series to filter out sports markets."""
    for ev in market.get("events") or []:
        for s in ev.get("series") or []:
            slug = (s.get("slug") or "").lower()
            ticker = (s.get("ticker") or "").lower()
            if slug in _SPORTS_SERIES_KEYWORDS or ticker in _SPORTS_SERIES_KEYWORDS:
                return True
            # Substring guard for e.g. "nba-2026-playoffs".
            for kw in _SPORTS_SERIES_KEYWORDS:
                if kw in slug or kw in ticker:
                    return True
    return False


def _is_recurring_series(market: Dict[str, Any]) -> bool:
    """Recurring (daily/hourly/weekly) series are lottery-style price/event
    markets, not substantive forecasting questions."""
    for ev in market.get("events") or []:
        for s in ev.get("series") or []:
            rec = (s.get("recurrence") or "").strip().lower()
            if rec in ("daily", "hourly", "weekly"):
                return True
    return False


def _parse_outcomes(market: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """Return (outcomes_pair, binary_resolution_YES_or_NO) or None if not a clean
    Yes/No resolved binary market."""
    raw_out = market.get("outcomes") or "[]"
    raw_prices = market.get("outcomePrices") or "[]"
    try:
        outs = json.loads(raw_out) if isinstance(raw_out, str) else raw_out
        prices = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
    except json.JSONDecodeError:
        return None
    if not (isinstance(outs, list) and len(outs) == 2 and isinstance(prices, list) and len(prices) == 2):
        return None
    norm = [str(o).strip().lower() for o in outs]
    if set(norm) != {"yes", "no"}:
        return None
    yes_idx = norm.index("yes")
    no_idx = norm.index("no")
    try:
        yes_price = float(prices[yes_idx])
        no_price = float(prices[no_idx])
    except (ValueError, TypeError):
        return None
    # Resolved binary: exactly one side is 1.0 and the other 0.0.
    if abs(yes_price - 1.0) < 1e-6 and abs(no_price - 0.0) < 1e-6:
        return ("yes_no", "YES")
    if abs(yes_price - 0.0) < 1e-6 and abs(no_price - 1.0) < 1e-6:
        return ("yes_no", "NO")
    return None


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        # Polymarket returns variants like "2026-06-21T18:30:00Z" and
        # "2026-06-21 16:47:32+00".
        s2 = s.replace("Z", "+00:00").replace(" ", "T", 1)
        return datetime.fromisoformat(s2)
    except ValueError:
        return None


def _candidates(want: int) -> List[Dict[str, Any]]:
    """Page through Polymarket Gamma API for closed binary markets.

    `closed=true&archived=false` returns markets the API treats as
    finalized. We order by closedTime desc so the freshest resolutions come
    first; tighten further inside the filter loop.
    """
    out: List[Dict[str, Any]] = []
    offset = 0
    pages_seen = 0
    page_size = 100
    # Gamma API does not document a hard offset cap; we self-limit to keep
    # the run bounded.
    max_offset = int(os.environ.get("PYM_MAX_OFFSET", 8000))
    while len(out) < want and offset <= max_offset:
        # Sort by volume desc rather than closedTime desc: substantive markets
        # carry far more dollar volume than daily lottery markets, so this
        # surfaces the kind of question the LLM forecaster has a real
        # reasoning advantage on (elections, geopolitics, company events,
        # court cases) inside Gamma's ~2000-offset cap.
        params = {
            "closed": "true",
            "archived": "false",
            "order": "volumeNum",
            "ascending": "false",
            "limit": page_size,
            "offset": offset,
        }
        try:
            r = requests.get(MARKETS_URL, params=params, timeout=20)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  request failed at offset={offset}: {e}")
            break
        page = r.json()
        if not isinstance(page, list) or not page:
            break
        for m in page:
            # Resolution gate.
            res_status = (m.get("umaResolutionStatus") or "").strip().lower()
            if res_status != "resolved":
                continue
            parsed = _parse_outcomes(m)
            if parsed is None:
                continue
            _, binary_res = parsed
            vol = m.get("volumeNum") or 0
            try:
                vol = float(vol)
            except (ValueError, TypeError):
                vol = 0.0
            if vol < MIN_VOLUME:
                continue
            title = (m.get("question") or "").strip()
            if len(title) < MIN_QUESTION_LEN:
                continue
            if _is_low_quality_title(title):
                continue
            if _is_sports_event(m):
                continue
            if _is_recurring_series(m):
                continue
            closed_dt = _parse_dt(m.get("closedTime")) or _parse_dt(m.get("umaEndDate"))
            if closed_dt is None or closed_dt < _after_dt:
                continue
            opened_dt = _parse_dt(m.get("createdAt")) or _parse_dt(m.get("startDate"))
            if opened_dt is not None:
                days_open = (closed_dt - opened_dt).total_seconds() / 86400.0
                if days_open < MIN_OPEN_DAYS:
                    continue
            desc = (m.get("description") or "").strip()
            if len(desc) < MIN_DESCRIPTION_LEN:
                continue
            out.append({**m, "_resolved_to": binary_res, "_closed_dt": closed_dt})
        pages_seen += 1
        print(f"  page {pages_seen} (offset {offset}): {len(out)} candidates kept so far")
        offset += page_size
        time.sleep(0.15)
    # Event-level dedup: keep only the highest-volume market per parent event,
    # so the pool is not dominated by N variants of the same story.
    by_event: Dict[str, Dict[str, Any]] = {}
    for m in out:
        ev_slug = None
        for ev in m.get("events") or []:
            ev_slug = ev.get("slug") or ev.get("id")
            if ev_slug:
                break
        key = ev_slug or m.get("slug") or m.get("id") or ""
        vol = float(m.get("volumeNum") or 0)
        if key not in by_event or vol > float(by_event[key].get("volumeNum") or 0):
            by_event[key] = m
    deduped = list(by_event.values())
    print(f"  event-level dedup: {len(out)} -> {len(deduped)}")
    return deduped


def _normalize(market: Dict[str, Any]) -> Dict[str, Any]:
    slug = market.get("slug") or market.get("id") or "unknown"
    title = (market.get("question") or "").strip()
    description = (market.get("description") or "").strip()[:4000]
    binary_res = market["_resolved_to"]
    open_dt = _parse_dt(market.get("createdAt")) or _parse_dt(market.get("startDate"))
    close_dt = _parse_dt(market.get("endDate")) or market["_closed_dt"]
    resolve_dt = market["_closed_dt"]
    return {
        "id": f"polymarket-{slug}",
        "title": title,
        "background": description,
        "resolution_criteria": "",  # embedded in `background`; mirrors Manifold path
        "binary_resolution": binary_res,
        "open_time": open_dt.isoformat() if open_dt else None,
        "close_time": close_dt.isoformat() if close_dt else None,
        "resolve_time": resolve_dt.isoformat() if resolve_dt else None,
        "community_prediction_history": [],
        "url": f"https://polymarket.com/market/{slug}",
        "source": "polymarket",
        "volume": float(market.get("volumeNum") or 0.0),
        "unique_bettors": None,  # not exposed by Gamma; downstream tolerates None
    }


def fetch(target_n: int = TARGET_N_QUESTIONS) -> List[Dict[str, Any]]:
    cand = _candidates(want=target_n * 4)
    print(f"  normalizing {len(cand)} candidates...")
    out = [_normalize(m) for m in cand]
    # Balance YES/NO so base rate is not exploitable.
    yeses = [q for q in out if q["binary_resolution"] == "YES"]
    nos = [q for q in out if q["binary_resolution"] == "NO"]
    print(f"  pre-balance: YES={len(yeses)}  NO={len(nos)}")
    half = target_n // 2
    out = (yeses[:half] + nos[: target_n - half])[:target_n]
    return out


def main():
    qs = fetch()
    yes = sum(1 for q in qs if q["binary_resolution"] == "YES")
    no = sum(1 for q in qs if q["binary_resolution"] == "NO")
    print(f"Fetched {len(qs)} resolved binary Polymarket questions. YES={yes}  NO={no}")
    OUTPUT_PATH.write_text(json.dumps(qs, indent=2))
    print(f"Wrote {OUTPUT_PATH}")
    print()
    print("To run the downstream pipeline on this dataset:")
    print(f"  cp {OUTPUT_PATH} {OUTPUT_PATH.parent / 'questions_raw.json'}")
    print("  python run.py --skip-fetch")


if __name__ == "__main__":
    main()
