"""
rq2_experiment/config.py
========================
Configuration for RQ2 analysis.

RQ2 reuses all raw outputs and metrics from RQ1. No new RAG pipeline runs are needed.
This config simply extends rq1_experiment/config.py with RQ2-specific paths and mappings.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env from project root ────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)

# ── Reuse RQ1 data directories ─────────────────────────────────────────────────
from rq1_experiment.config import (
    ALL_CONDITIONS,
    CONDITION_META,
    DOMAINS,
    METRICS_DIR,
    RAW_OUTPUTS_DIR,
    ANALYSIS_DIR as RQ1_ANALYSIS_DIR,
    PLOTS_DIR as RQ1_PLOTS_DIR,
)

# ── RQ2-specific output directories ────────────────────────────────────────────
EXPERIMENT_DIR   = Path(__file__).resolve().parent
RESULTS_DIR      = _PROJECT_ROOT / "rq1_experiment" / "results"
ANALYSIS_DIR_RQ2 = RESULTS_DIR / "analysis" / "rq2"
PLOTS_DIR_RQ2    = RESULTS_DIR / "plots" / "rq2"

for _d in [ANALYSIS_DIR_RQ2, PLOTS_DIR_RQ2]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Source type configuration levels ──────────────────────────────────────────
# Level 1 = single-source (academic only / archival only)
# Level 2 = two-source mix  (academic + news / acad + archival)
# Level 3 = full diversity  (academic + news + tech / acad + arch + ref)
SOURCE_LEVEL_MAP = {
    # Technology
    "C1":  1,   # Academic only
    "C2":  2,   # Academic + News
    "C3":  3,   # Acad + News + Tech
    "C4":  1,
    "C5":  2,
    "C6":  3,
    # Healthcare
    "C7":  1,   # Academic only
    "C8":  2,   # Academic + News
    "C9":  3,   # Acad + News + Tech
    # History (uses archival sources instead of news)
    "C10": 1,   # Academic only
    "C11": 2,   # Acad + Archival
    "C12": 3,   # Acad + Arch + Ref
}

SOURCE_LEVEL_LABELS = {
    1: "Single-Source",
    2: "Two-Source Mix",
    3: "Full Diversity",
}

# ── Domain volatility numeric scores (matches CONDITION_META volatility strings) ──
VOLATILITY_SCORE = {
    "high":   3,
    "medium": 2,
    "low":    1,
}

# Domain → volatility label (for convenience)
DOMAIN_VOLATILITY = {
    "technology": "high",
    "healthcare":  "medium",
    "history":     "low",
}

# ── Metrics ───────────────────────────────────────────────────────────────────
PRIMARY_METRICS   = ["bertscore_f1", "hallucination_rate", "precision_at_5", "ndcg_at_5", "human_eval_score"]
SECONDARY_METRICS = ["rouge_l", "meteor"]

# ── DigitalOcean LLM (for any LLM-assisted steps) ─────────────────────────────
DO_API_KEY    = os.getenv("DO_API_KEY", "")
DO_MODEL_NAME = os.getenv("DO_MODEL_NAME", "llama-4-maverick")
DO_CHAT_URL   = "https://inference.do-ai.run/v1/chat/completions"
