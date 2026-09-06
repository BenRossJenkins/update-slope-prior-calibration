"""Central configuration for the dose-response forecasting study.

Pilot/full-run knobs can be overridden via environment variables without
editing this file:

  PILOT_N_Q=3           — number of questions to fetch
  PILOT_SAMPLES=1       — samples per (model, question, level) cell
  PILOT_MODELS=a,b,c    — comma-separated api_ids to keep (subset of MODELS)
  PILOT_LADDER_MODEL=anthropic:claude-haiku-4-5-20251001  — cheap ladder builder
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Load secrets from code/.env if present (kept out of git, see .gitignore).
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass
# RUN_NAME lets us run multiple datasets in parallel without clobbering
# each other's outputs. Empty default = the original Manifold paths.
RUN_NAME = os.environ.get("RUN_NAME", "").strip()
DATA = (ROOT / "data" / RUN_NAME) if RUN_NAME else (ROOT / "data")
FIGS = ROOT / "figures"
DATA.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(exist_ok=True)

QUESTIONS_RAW = DATA / "questions_raw.json"
QUESTIONS_LADDER = DATA / "questions_ladder.json"
ELICITATIONS = DATA / "elicitations.jsonl"
RESULTS = DATA / "results.json"

LEVELS = ["L0", "L1", "L2", "L3"]

SAMPLES_PER_CELL = int(os.environ.get("PILOT_SAMPLES", 3))

# Model panel. Adjust to what you actually have access to.
# (provider, api_id, display_name, family, reasoning)
_ALL_MODELS = [
    ("openai", "gpt-5",                    "GPT-5",            "openai",   True),
    ("openai", "gpt-5-mini",               "GPT-5 mini",       "openai",   True),
    ("openai", "gpt-4o-2024-11-20",        "GPT-4o",           "openai",   False),
    ("openai", "o3",                       "o3",               "openai",   True),
    ("anthropic", "claude-opus-4-7",       "Claude Opus 4.7",  "anthropic", True),
    ("anthropic", "claude-sonnet-4-6",     "Claude Sonnet 4.6","anthropic", False),
    ("anthropic", "claude-haiku-4-5-20251001", "Claude Haiku 4.5", "anthropic", False),
]

if os.environ.get("PILOT_MODELS"):
    _keep = {x.strip() for x in os.environ["PILOT_MODELS"].split(",") if x.strip()}
    MODELS = [m for m in _ALL_MODELS if m[1] in _keep]
else:
    MODELS = _ALL_MODELS

TARGET_N_QUESTIONS = int(os.environ.get("PILOT_N_Q", 40))
RESOLUTION_AFTER = "2025-01-01"

if os.environ.get("PILOT_LADDER_MODEL"):
    _lp, _lid = os.environ["PILOT_LADDER_MODEL"].split(":", 1)
    EVIDENCE_BUILDER_MODEL = (_lp, _lid)
else:
    EVIDENCE_BUILDER_MODEL = ("anthropic", "claude-opus-4-7")
