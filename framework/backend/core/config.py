"""
core/config.py
==============
Self-contained configuration for the standalone framework.

Nothing here imports from the experiment packages (rq1/rq3/rq4).
All runtime data is read from ``backend/data/`` so the framework runs even if
the experiment folders are removed.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict

# ── Paths ─────────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"

MODELS_PATH = DATA_DIR / "rq3_models.json"
CONDITION_METRICS_PATH = DATA_DIR / "condition_metrics.json"
CONDITION_META_PATH = DATA_DIR / "condition_meta.json"
VALIDATION_PATH = DATA_DIR / "rq4_validation.json"
KAPPA_PATH = DATA_DIR / "human_eval_kappa.json"

# ── Domain volatility (model feature encoding) ─────────────────────────────────
# Must match the encoding used when the RQ3 model was trained.
DOMAIN_VOLATILITY_SCORE: Dict[str, float] = {
    "technology": 1.0,
    "healthcare": 0.5,
    "history": 0.0,
}

DOMAIN_LABELS: Dict[str, str] = {
    "technology": "Technology (High Volatility)",
    "healthcare": "Healthcare (Medium Volatility)",
    "history": "History (Low Volatility)",
}

# Half-life (days) used to convert document age -> freshness score.
# Matches the value used in the RQ3 experiment.
FRESHNESS_HALF_LIFE_DAYS = 180.0

# ── Metric metadata (drives the UI; no values hardcoded) ──────────────────────
# direction = "higher" means higher is better; "lower" for hallucination.
PRIMARY_METRICS: Dict[str, dict] = {
    "precision_at_5": {"label": "Precision@5", "direction": "higher", "scale": [0.0, 1.0]},
    "ndcg_at_5": {"label": "nDCG@5", "direction": "higher", "scale": [0.0, 1.0]},
    "bertscore_f1": {"label": "BERTScore F1", "direction": "higher", "scale": [0.0, 1.0]},
    "hallucination_rate": {"label": "Hallucination", "direction": "lower", "scale": [0.0, 1.0]},
    "human_eval_score": {"label": "Human Eval", "direction": "higher", "scale": [1.0, 5.0]},
}

# Weights used to compute a single "deployment readiness" score from predicted
# metrics. Transparent and editable; not per-domain magic numbers.
READINESS_WEIGHTS: Dict[str, float] = {
    "precision_at_5": 0.35,
    "bertscore_f1": 0.30,
    "hallucination_rate": 0.20,  # inverted (lower is better)
    "ndcg_at_5": 0.15,
}

# Domain-volatility thresholds -> recommended update cadence.
# Volatility is the documented driver of staleness (Rule 1 of the deployment
# guidelines); the score is the same encoding the RQ3 model is trained on.
UPDATE_CADENCE_BY_VOLATILITY = [
    (0.75, "Weekly", "High-volatility domain: information ages fast — refresh the knowledge base weekly."),
    (0.25, "Monthly", "Medium-volatility domain: refresh monthly to balance freshness against cost."),
    (0.0, "Quarterly", "Low-volatility domain: information is stable — quarterly (or slower) refresh is sufficient."),
]


# ── Data loaders (cached) ─────────────────────────────────────────────────────
def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required data file missing: {path}. "
            f"Run `python sync_data.py` from the framework folder to bundle it."
        )
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_models() -> dict:
    return _read_json(MODELS_PATH)


@lru_cache(maxsize=1)
def load_condition_metrics() -> dict:
    return _read_json(CONDITION_METRICS_PATH)


@lru_cache(maxsize=1)
def load_condition_meta() -> dict:
    return _read_json(CONDITION_META_PATH)


@lru_cache(maxsize=1)
def load_validation() -> dict:
    return _read_json(VALIDATION_PATH)


@lru_cache(maxsize=1)
def load_kappa() -> dict:
    return _read_json(KAPPA_PATH)


def clear_caches() -> None:
    for fn in (load_models, load_condition_metrics, load_condition_meta,
               load_validation, load_kappa):
        fn.cache_clear()
