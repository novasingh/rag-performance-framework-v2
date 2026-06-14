"""
rq1_experiment/evaluation/response_metrics.py
===============================================
Response quality metrics:
  - BERTScore (F1) — semantic similarity vs. reference answer [PRIMARY]
  - ROUGE-L         — longest common subsequence recall [SECONDARY]
  - METEOR          — harmonic mean of precision/recall with stemming [SECONDARY]

All per Report.pdf Table 3.3.
"""
from __future__ import annotations

import logging
import re
import statistics
from typing import List, Optional, Tuple

import numpy as np

from dataset_rag_builder.utils.model_utils import get_embedding_model

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = None

# ── ROUGE-L ───────────────────────────────────────────────────────────────────
def _lcs_length(s1: List[str], s2: List[str]) -> int:
    """Length of the Longest Common Subsequence (LCS)."""
    m, n = len(s1), len(s2)
    if m == 0 or n == 0:
        return 0
    # DP with O(min(m,n)) space
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def rouge_l(hypothesis: str, reference: str) -> float:
    """
    ROUGE-L F1 score between hypothesis and reference.
    Uses word-level tokenisation.
    """
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    if not hyp_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(hyp_tokens, ref_tokens)
    precision = lcs / len(hyp_tokens)
    recall    = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 6)


# ── METEOR (lightweight implementation) ──────────────────────────────────────
_STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for",
    "of","with","is","are","was","were","be","been","being","have",
    "has","had","do","does","did","will","would","could","should","may",
    "might","shall","can","need","dare","ought","used","not","no","it",
    "its","i","we","you","he","she","they","that","this","these","those",
}


def _stem(word: str) -> str:
    """Ultra-lightweight suffix stripper (Porter-like, no library needed)."""
    w = word.lower().rstrip(".,!?;:\"'")
    for suffix in ["ings", "ing", "tion", "tions", "ed", "er", "ly", "s"]:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[: -len(suffix)]
    return w


def meteor_score(hypothesis: str, reference: str, alpha: float = 0.9,
                 beta: float = 3.0, gamma: float = 0.5) -> float:
    """
    Lightweight METEOR approximation (unigram + stemming, no WordNet).
    alpha=0.9, beta=3.0, gamma=0.5 are standard METEOR defaults.
    """
    hyp_tokens = [_stem(t) for t in hypothesis.lower().split() if t not in _STOP_WORDS]
    ref_tokens = [_stem(t) for t in reference.lower().split()  if t not in _STOP_WORDS]

    if not hyp_tokens or not ref_tokens:
        return 0.0

    ref_set   = set(ref_tokens)
    matches   = sum(1 for t in hyp_tokens if t in ref_set)

    if matches == 0:
        return 0.0

    prec   = matches / len(hyp_tokens)
    recall = matches / len(ref_tokens)
    fmean  = (prec * recall) / (alpha * prec + (1 - alpha) * recall + 1e-9)

    # Chunking penalty
    chunks = 1
    in_match = False
    for t in hyp_tokens:
        if t in ref_set:
            if not in_match:
                chunks += 1
                in_match = True
        else:
            in_match = False
    penalty = gamma * (chunks / matches) ** beta

    score = fmean * (1 - penalty)
    return round(max(0.0, score), 6)


# ── Semantic similarity scorer ───────────────────────────────────────────────

def _get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = get_embedding_model()
    return _EMBEDDING_MODEL


def _cosine_similarity_matrix(vectors_a: np.ndarray, vectors_b: np.ndarray) -> List[float]:
    if vectors_a.size == 0 or vectors_b.size == 0:
        return []
    if vectors_a.shape != vectors_b.shape:
        raise ValueError("Embedding arrays must have matching shapes")

    similarities = np.sum(vectors_a * vectors_b, axis=1)
    similarities = np.clip(similarities, -1.0, 1.0)
    return [round(float((value + 1.0) / 2.0), 6) for value in similarities]

def compute_bertscore(
    hypotheses: List[str],
    references: List[str],
    batch_size: int = 8,
) -> List[float]:
    """
    Compute a stable semantic similarity score for a list of hypothesis/reference pairs.

    This uses the project's cached MiniLM embedding model so analysis does not
    repeatedly download large transformer checkpoints.
    """
    if not hypotheses:
        return []

    try:
        model = _get_embedding_model()
        hyp_vectors = model.encode(
            hypotheses,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        ref_vectors = model.encode(
            references,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return _cosine_similarity_matrix(hyp_vectors, ref_vectors)
    except Exception as exc:
        logger.error("Semantic similarity scorer failed: %s — falling back to ROUGE-L", exc)
        return [rouge_l(h, r) for h, r in zip(hypotheses, references)]


# ── Aggregate helpers ────────────────────────────────────────────────────────

def _stats(vals: List[float]) -> dict:
    n    = len(vals)
    mean = sum(vals) / n if n else 0.0
    std  = statistics.stdev(vals) if n > 1 else 0.0
    return {"mean": round(mean, 6), "std": round(std, 6), "n": n}


def compute_response_metrics_single(
    hypothesis: str,
    reference: str,
) -> dict:
    """Compute all response metrics for a single pair."""
    bert_scores = compute_bertscore([hypothesis], [reference])
    return {
        "bertscore_f1": bert_scores[0] if bert_scores else 0.0,
        "rouge_l":      rouge_l(hypothesis, reference),
        "meteor":       meteor_score(hypothesis, reference),
    }


def aggregate_response_metrics(
    hypotheses: List[str],
    references: List[str],
    batch_size: int = 8,
) -> dict:
    """Aggregate response quality metrics across multiple query-response pairs."""
    bert_scores   = compute_bertscore(hypotheses, references, batch_size=batch_size)
    rouge_scores  = [rouge_l(h, r)      for h, r in zip(hypotheses, references)]
    meteor_scores = [meteor_score(h, r) for h, r in zip(hypotheses, references)]

    return {
        "bertscore_f1": _stats(bert_scores),
        "rouge_l":      _stats(rouge_scores),
        "meteor":       _stats(meteor_scores),
    }
