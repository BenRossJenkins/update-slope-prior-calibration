"""Elicit forecasts from the 7-model panel on the BLINDED Manifold ladders
(see build_ladder_blinded.py). Mirrors elicit.py but points at the
blinded ladder file and writes to elicitations_blinded.jsonl.

Use to test whether the V-A confirmatory test
(naive gamma_1 -0.531 -> controlled -1.781) survives content-blind
ladders. See RE_ELICITATION_DECISION_RULE.md for the pre-committed
criteria.
"""
from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import config
import elicit as elicit_mod

# Override config paths via monkey-patch BEFORE elicit functions read them
DATA = config.DATA
QUESTIONS_BLINDED = DATA / "questions_ladder_blinded.json"
ELICITATIONS_BLINDED = DATA / "elicitations_blinded.jsonl"


def main(max_workers: int = 6):
    questions = json.loads(QUESTIONS_BLINDED.read_text())
    print(f"Loaded {len(questions)} blinded-ladder questions")

    seen = elicit_mod._existing_cells(ELICITATIONS_BLINDED)
    print(f"Resuming: {len(seen)} cells already complete.")

    jobs = []
    for q in questions:
        for model in config.MODELS:
            provider, api_id, display, family, reasoning = model
            for level in config.LEVELS:
                for sample in range(config.SAMPLES_PER_CELL):
                    key = (q["id"], api_id, level, sample)
                    if key in seen:
                        continue
                    jobs.append((provider, api_id, display, family, reasoning,
                                 q, level, sample))

    print(f"Running {len(jobs)} blinded elicitations (max_workers={max_workers})")

    with ELICITATIONS_BLINDED.open("a") as f, \
         ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(elicit_mod._elicit_one, *j) for j in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            rec = fut.result()
            f.write(json.dumps(rec) + "\n")
            f.flush()

    print(f"Done. Appended results to {ELICITATIONS_BLINDED}")


if __name__ == "__main__":
    main()
