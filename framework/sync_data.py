"""
sync_data.py
============
Bundle the experiment artifacts into ``framework/backend/data/`` so the
framework runs standalone (independent of the rq1/rq3/rq4 packages).

Run this after re-running the experiments to refresh the framework's data:

    python sync_data.py

It copies:
- the trained RQ3 model            -> data/rq3_models.json
- the RQ1 per-condition metrics     -> data/condition_metrics.json
- the RQ4 validation results        -> data/rq4_validation.json
- the inter-rater reliability        -> data/human_eval_kappa.json
and writes the 12-condition metadata -> data/condition_meta.json
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

FRAMEWORK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FRAMEWORK_DIR.parent
DATA_DIR = FRAMEWORK_DIR / "backend" / "data"

# Source artifacts in the experiment tree -> bundled name.
COPY_MAP = {
    PROJECT_ROOT / "rq3_experiment" / "results" / "rq3_models.json": "rq3_models.json",
    PROJECT_ROOT / "rq1_experiment" / "results" / "metrics" / "all_conditions_metrics.json": "condition_metrics.json",
    PROJECT_ROOT / "rq4_experiment" / "results" / "rq4_validation.json": "rq4_validation.json",
    PROJECT_ROOT / "rq1_experiment" / "results" / "metrics" / "human_eval_kappa.json": "human_eval_kappa.json",
}

# Authoritative 12-condition metadata (structural, mirrors rq1_experiment/config.py).
CONDITION_META = {
    "C1":  {"domain": "technology", "volatility": "high",   "freshness": "<= 1 week",     "source_config": "Academic only"},
    "C2":  {"domain": "technology", "volatility": "high",   "freshness": "<= 1 week",     "source_config": "Academic + News"},
    "C3":  {"domain": "technology", "volatility": "high",   "freshness": "<= 1 week",     "source_config": "Academic + News + Tech"},
    "C4":  {"domain": "technology", "volatility": "high",   "freshness": "1 wk - 1 mo",   "source_config": "Academic only"},
    "C5":  {"domain": "technology", "volatility": "high",   "freshness": "1 wk - 1 mo",   "source_config": "Academic + News"},
    "C6":  {"domain": "technology", "volatility": "high",   "freshness": "1 wk - 1 mo",   "source_config": "Academic + News + Tech"},
    "C7":  {"domain": "healthcare", "volatility": "medium", "freshness": "1-6 months",    "source_config": "Academic only"},
    "C8":  {"domain": "healthcare", "volatility": "medium", "freshness": "1-6 months",    "source_config": "Academic + News"},
    "C9":  {"domain": "healthcare", "volatility": "medium", "freshness": "6-12 months",   "source_config": "Academic + News + Tech"},
    "C10": {"domain": "history",    "volatility": "low",    "freshness": ">= 6 mo (adj.)", "source_config": "Academic only"},
    "C11": {"domain": "history",    "volatility": "low",    "freshness": ">= 6 mo (adj.)", "source_config": "Academic + Archival"},
    "C12": {"domain": "history",    "volatility": "low",    "freshness": ">= 6 mo (adj.)", "source_config": "Academic + Archival + Reference"},
}


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    missing = []

    for src, dest_name in COPY_MAP.items():
        dest = DATA_DIR / dest_name
        if src.exists():
            shutil.copyfile(src, dest)
            print(f"  copied  {src.relative_to(PROJECT_ROOT)} -> backend/data/{dest_name}")
        else:
            missing.append(src)
            print(f"  MISSING {src.relative_to(PROJECT_ROOT)} (run the matching experiment first)")

    meta_path = DATA_DIR / "condition_meta.json"
    meta_path.write_text(json.dumps(CONDITION_META, indent=2), encoding="utf-8")
    print(f"  wrote   backend/data/condition_meta.json ({len(CONDITION_META)} conditions)")

    if missing:
        print(f"\nWARNING: {len(missing)} source file(s) missing. The framework will "
              f"fail to load those endpoints until the experiments are run.")
        return 1
    print("\nData bundled. The framework can now run standalone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
