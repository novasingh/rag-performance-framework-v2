"""
rq1_experiment/config.py
========================
Central configuration for the RQ1 experiment pipeline.
All paths, model names, hyperparameters, and rate limits live here.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env from project root ────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DATASET_DIR        = _PROJECT_ROOT / "rag_dataset"
CONDITIONS_DIR     = DATASET_DIR / "final" / "conditions"
EXPERIMENT_DIR     = _PROJECT_ROOT / "rq1_experiment"
RESULTS_DIR        = EXPERIMENT_DIR / "results"
RAW_OUTPUTS_DIR    = RESULTS_DIR / "raw_outputs"
METRICS_DIR        = RESULTS_DIR / "metrics"
ANALYSIS_DIR       = RESULTS_DIR / "analysis"
PLOTS_DIR          = RESULTS_DIR / "plots"
QUERY_BANK_DIR     = EXPERIMENT_DIR / "query_bank" / "queries"
FAISS_INDEX_DIR    = RESULTS_DIR / "faiss_indexes"

# Create all output directories
for _d in [RAW_OUTPUTS_DIR, METRICS_DIR, ANALYSIS_DIR, PLOTS_DIR,
           QUERY_BANK_DIR, FAISS_INDEX_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Conditions — mirrors dataset_rag_builder/config.py RQ1_CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────
ALL_CONDITIONS = ["C1", "C2", "C3", "C4", "C5", "C6",
                  "C7", "C8", "C9", "C10", "C11", "C12"]

CONDITION_META = {
    "C1":  dict(domain="technology", volatility="high",   freshness="<= 1 week",      source_config="Academic only",           freshness_days=(0, 7)),
    "C2":  dict(domain="technology", volatility="high",   freshness="<= 1 week",      source_config="Academic + News",         freshness_days=(0, 7)),
    "C3":  dict(domain="technology", volatility="high",   freshness="<= 1 week",      source_config="Acad + News + Tech",      freshness_days=(0, 7)),
    "C4":  dict(domain="technology", volatility="high",   freshness="1 wk - 1 mo",    source_config="Academic only",           freshness_days=(7, 30)),
    "C5":  dict(domain="technology", volatility="high",   freshness="1 wk - 1 mo",    source_config="Academic + News",         freshness_days=(7, 30)),
    "C6":  dict(domain="technology", volatility="high",   freshness="1 wk - 1 mo",    source_config="Acad + News + Tech",      freshness_days=(7, 30)),
    "C7":  dict(domain="healthcare", volatility="medium", freshness="1-6 months",      source_config="Academic only",           freshness_days=(30, 180)),
    "C8":  dict(domain="healthcare", volatility="medium", freshness="1-6 months",      source_config="Academic + News",         freshness_days=(30, 180)),
    "C9":  dict(domain="healthcare", volatility="medium", freshness="6-12 months",     source_config="Acad + News + Tech",      freshness_days=(180, 365)),
    "C10": dict(domain="history",    volatility="low",    freshness=">= 6 mo (adj.)",  source_config="Academic only",           freshness_days=(180, None)),
    "C11": dict(domain="history",    volatility="low",    freshness=">= 6 mo (adj.)",  source_config="Acad + Archival",         freshness_days=(180, None)),
    "C12": dict(domain="history",    volatility="low",    freshness=">= 6 mo (adj.)",  source_config="Acad + Arch + Ref",       freshness_days=(180, None)),
}

DOMAIN_VOLATILITY = {
    "technology": "high",
    "healthcare":  "medium",
    "history":     "low",
}

DOMAIN_VOLATILITY_SCORE = {
    "technology": 1.0,
    "healthcare":  0.5,
    "history":     0.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Model
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM        = 384   # dimension of all-MiniLM-L6-v2
RETRIEVAL_TOP_K      = 5     # k for P@5 / nDCG@5


# ─────────────────────────────────────────────────────────────────────────────
# Google AI Studio — Gemma 3 27B
# ─────────────────────────────────────────────────────────────────────────────
GOOGLE_AI_API_KEY      = os.getenv("GOOGLE_AI_API_KEY", "")
# OpenAI API key for the verifier model (GPT‑OSS‑20B). If not set, calls will fail.
OPENAI_API_KEY        = os.getenv("OPENAI_API_KEY", "")
# Gemma models available on this API key (verified working):
#   models/gemma-3-27b-it  — Gemma 3 27B 
#   models/gemma-4-26b-a4b-it  — Gemma 4 26B (MoE) — WORKING (13.6s avg)
#   models/gemma-4-31b-it      — Gemma 4 31B — 500 errors (unavailable for this key)
GEMMA_MODEL_NAME       = "models/gemma-4-26b-a4b-it"  # Google AI Studio model ID
GEMMA_RPM_LIMIT        = 29                             # requests per minute
GEMMA_MIN_INTERVAL_SEC = 60.0 / GEMMA_RPM_LIMIT    # ~2.069 s between requests
GEMMA_MAX_OUTPUT_TOKENS = 1024
GEMMA_TEMPERATURE       = 0.0                       # deterministic for reproducibility

# ─────────────────────────────────────────────────────────────────────────────
# Ollama (Local) — llama3.1:8b
# ─────────────────────────────────────────────────────────────────────────────
# Benchmark results (2026-05-13):
#   ollama/llama3.1:8b         11.4s avg | 3/3 success | ~7 tok/s | no quota
#   google/gemma-4-26b-a4b-it  13.6s avg | 2/2 success | 29 RPM limit
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.1:8b")

# ─────────────────────────────────────────────────────────────────────────────
# DigitalOcean Inference — llama-4-maverick
# ─────────────────────────────────────────────────────────────────────────────
DO_API_KEY      = os.getenv("DO_API_KEY", "")
DO_MODEL_NAME   = os.getenv("DO_MODEL_NAME", "llama-4-maverick")
DO_EVAL_MODEL_A = os.getenv("DO_EVAL_MODEL_A", "llama-4-maverick")
DO_EVAL_MODEL_B = os.getenv("DO_EVAL_MODEL_B", "deepseek-3.2")
DO_EVAL_MODEL_C = os.getenv("DO_EVAL_MODEL_C", "deepseek-v4-pro")
DO_CHAT_URL     = "https://inference.do-ai.run/v1/chat/completions"
# Pricing constants
DO_INPUT_PRICE_PER_M  = 0.25 # $0.25 per 1M input tokens
DO_OUTPUT_PRICE_PER_M = 0.87 # $0.87 per 1M output tokens
DO_RPM_LIMIT          = 42   # Requests per minute (40-45 range requested)
DO_RPH_LIMIT          = 5000 # Requests per hour

# ─────────────────────────────────────────────────────────────────────────────
# Active backend — 'digitalocean' (fastest), 'ollama', or 'google'
# ─────────────────────────────────────────────────────────────────────────────
GENERATOR_BACKEND = os.getenv("GENERATOR_BACKEND", "digitalocean")

# Fixed RAG prompt template — held constant across ALL conditions
RAG_PROMPT_TEMPLATE = """\
You are a precise and factual assistant. Answer the question below using ONLY the provided context documents. \
If the context does not contain sufficient information to answer the question, say "I cannot determine this from the provided documents."

Context Documents:
{context}

Question: {question}

Answer:"""


# ─────────────────────────────────────────────────────────────────────────────
# Query Bank
# ─────────────────────────────────────────────────────────────────────────────
QUERIES_PER_DOMAIN     = 200   # target queries per domain (used across conditions)
TIME_SENSITIVE_RATIO   = 0.20  # ~20% of queries are time-sensitive
TIME_NEUTRAL_RATIO     = 0.80  # ~80% are time-neutral

QUERY_TYPES = ["factual", "analytical", "comparative"]
QUERY_TYPE_DISTRIBUTION = {
    "factual":     0.40,   # single verifiable answer
    "analytical":  0.35,   # multi-passage synthesis
    "comparative": 0.25,   # contrast two things
}

DOMAINS = ["technology", "healthcare", "history"]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
BERTSCORE_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"  # smaller, stable local scorer
BERTSCORE_LANG       = "en"
NLI_ENTAIL_THRESHOLD = 0.5    # probability threshold for "entailed" classification
HUMAN_EVAL_RATIO     = 0.20   # 20% stratified sample for human evaluation
HUMAN_EVAL_LIKERT    = 5      # 5-point Likert scale
COHEN_KAPPA_THRESHOLD = 0.70  # minimum inter-rater agreement
# Human evaluation raters and verifier configuration
# -------------------------------------------------
# CRITICAL: Both raters MUST use the SAME model to achieve kappa >= 0.70.
# Different models (e.g. llama-4-maverick vs gemma-4-31b-it) have different
# scoring personalities, which systematically reduces inter-rater agreement.
# Both raters use llama-4-maverick (faster, cheaper, consistent).
HUMAN_EVAL_PERSONAS = {
    "persona_a": {
        "label": "Rater A (Balanced Evaluator)",
        "model": "llama-4-maverick",
    },
    "persona_b": {
        "label": "Rater B (Balanced Evaluator)",
        "model": "llama-4-maverick",
    },
}

HUMAN_EVAL_VERIFIER = {
    "label": "Verification Judge",
    # Use Gemma 4 31B for verification (more balanced, less strict)
    "model": "gemma-4-31b-it",
}


# ─────────────────────────────────────────────────────────────────────────────
# Statistical Analysis
# ─────────────────────────────────────────────────────────────────────────────
ALPHA                = 0.05   # significance level
CV_FOLDS             = 10     # 10-fold cross-validation
RF_N_ESTIMATORS      = 500    # Random Forest trees
EFFECT_SIZE_SMALL    = 0.01   # η² thresholds
EFFECT_SIZE_MEDIUM   = 0.06
EFFECT_SIZE_LARGE    = 0.14
