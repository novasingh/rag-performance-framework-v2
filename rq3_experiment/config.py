"""
rq3_experiment/config.py
=======================
Configuration for the RQ3 Predictive Framework module.
"""
from pathlib import Path

# Paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = _PROJECT_ROOT / "rq1_experiment" / "results" / "metrics"
RAW_OUTPUTS_DIR = _PROJECT_ROOT / "rq1_experiment" / "results" / "raw_outputs"
RESULTS_DIR_RQ3 = _PROJECT_ROOT / "rq3_experiment" / "results"

RESULTS_DIR_RQ3.mkdir(parents=True, exist_ok=True)

# Metrics
PRIMARY_METRICS = [
    "bertscore_f1",
    "hallucination_rate",
    "precision_at_5",
    "ndcg_at_5",
    "human_eval_score",
]

# Base features (condition-level)
BASE_FEATURES = [
    "freshness_score",
    "source_diversity_index",
    "domain_volatility",
]

# Per-QUERY retrieval features (vary per query within condition)
PER_QUERY_FEATURES = [
    "avg_similarity",
    "min_similarity",
    "relevant_doc_count",
    "response_length",
]

# Per-QUERY objective metric features (for human_eval prediction)
# These are REAL per-query computed metrics that correlate with human ratings
OBJ_METRIC_FEATURES = [
    "per_query_bertscore",       # BERTScore for this specific query
    "per_query_hallucination",    # NLI hallucination rate for this specific query
    "per_query_rouge_l",          # ROUGE-L for this specific query
    "per_query_meteor",           # METEOR for this specific query
]

# Interaction terms
INTERACTION_FEATURES = [
    "fresh_x_diversity",
    "fresh_x_volatility",
    "source_x_volatility",
]

# Polynomial features
POLY_FEATURES = [
    "freshness_score_sq",
    "source_diversity_index_sq",
    "domain_volatility_sq",
]

# Full feature set for modelling (10 features: 3 base + 4 per-query + 3 interactions)
FEATURE_COLS = BASE_FEATURES + PER_QUERY_FEATURES + INTERACTION_FEATURES

# Feature set for human_eval prediction (adds per-query objective metrics)
FEATURE_COLS_HUMAN_EVAL = FEATURE_COLS + OBJ_METRIC_FEATURES

# Base features for OLS
FEATURE_COLS_BASE = BASE_FEATURES + INTERACTION_FEATURES

# For decay curve fitting — the OLS feature set so we can get the freshness coef
DECAY_FEATURE_COLS = ["freshness_score", "domain_volatility"]

# Random Forest Config
RF_N_ESTIMATORS = 500
CV_FOLDS = 10
RANDOM_SEED = 42

# XGBoost Config
XGB_N_ESTIMATORS = 300
XGB_LEARNING_RATE = 0.05
XGB_MAX_DEPTH = 4
XGB_EARLY_STOPPING_ROUNDS = 20

# Domain encoding
DOMAIN_VOLATILITY_SCORE = {
    "technology": 1.0,
    "healthcare":  0.5,
    "history":     0.0,
}