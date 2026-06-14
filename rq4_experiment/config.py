"""
rq4_experiment/config.py
========================
Configuration for the RQ4 Framework Validation module.
"""
from pathlib import Path

# Paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR_RQ4 = _PROJECT_ROOT / "rq4_experiment" / "results"
PLOTS_DIR_RQ4 = RESULTS_DIR_RQ4 / "plots"

RESULTS_DIR_RQ4.mkdir(parents=True, exist_ok=True)
PLOTS_DIR_RQ4.mkdir(parents=True, exist_ok=True)

# Metrics and Thresholds
PRIMARY_METRICS = [
    "bertscore_f1",
    "hallucination_rate",
    "precision_at_5",
    "ndcg_at_5",
    "human_eval_score",
]

FEATURE_COLS = [
    "freshness_score",
    "source_diversity_index",
    "domain_volatility",
    "fresh_x_diversity",
    "fresh_x_volatility",
    "source_x_volatility",
]

# If prediction error > this threshold, flag as boundary condition
BOUNDARY_MAE_THRESHOLD = 0.25 
