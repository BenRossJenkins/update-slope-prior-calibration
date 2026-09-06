"""End-to-end orchestrator. Run from the code/ directory.

  $ export OPENAI_API_KEY=...
  $ export ANTHROPIC_API_KEY=...
  $ python run.py

Stages can be skipped with flags, e.g. `python run.py --skip-fetch`.
"""
from __future__ import annotations
import argparse
import importlib
import sys


STAGES = [
    ("fetch", "fetch_questions"),
    ("ladder", "build_ladder"),
    ("elicit", "elicit"),
    ("analyze", "analyze"),
    ("emit", "emit_data_tex"),
]


def main():
    p = argparse.ArgumentParser()
    for name, _ in STAGES:
        p.add_argument(f"--skip-{name}", action="store_true", help=f"skip {name} stage")
    p.add_argument("--only", choices=[n for n, _ in STAGES], default=None,
                   help="run only this stage")
    args = p.parse_args()

    for name, modname in STAGES:
        if args.only and args.only != name:
            continue
        if getattr(args, f"skip_{name}"):
            print(f"== skip: {name} ==")
            continue
        print(f"\n== stage: {name} ==")
        mod = importlib.import_module(modname)
        mod.main()


if __name__ == "__main__":
    sys.exit(main())
