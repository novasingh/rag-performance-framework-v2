"""
rq1_experiment/evaluation/retrieval_metrics.py
================================================
Computes retrieval precision metrics:
  - P@k  : Precision at k (proportion of top-k docs that are relevant)
  - nDCG@k: Normalised Discounted Cumulative Gain at k

These are PRIMARY metrics for RQ1 (Report.pdf Section 3.4.4).
Uses binary relevance labels from Retriever.retrieve().
"""
from __future__ import annotations

import math
from typing import List


def precision_at_k(relevance_labels: List[int], k: int = 5) -> float:
    """
    Precision@k — proportion of top-k retrieved documents that are relevant.

    Parameters
    ----------
    relevance_labels : binary list (1=relevant, 0=not), ordered by rank
    k                : cut-off (default 5)

    Returns
    -------
    float in [0, 1]
    """
    if not relevance_labels:
        return 0.0
    top_k = relevance_labels[:k]
    return round(sum(top_k) / len(top_k), 6)


def dcg_at_k(relevance_labels: List[int], k: int = 5) -> float:
    """Discounted Cumulative Gain@k."""
    dcg = 0.0
    for rank, rel in enumerate(relevance_labels[:k], start=1):
        if rel > 0:
            dcg += rel / math.log2(rank + 1)
    return dcg


def ndcg_at_k(relevance_labels: List[int], k: int = 5) -> float:
    """
    Normalised DCG@k.

    IDCG is computed from the ideal ranking (all relevant docs first).
    """
    if not relevance_labels:
        return 0.0

    actual_dcg = dcg_at_k(relevance_labels, k)

    # Ideal ranking: sort by relevance descending
    ideal = sorted(relevance_labels, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)

    if ideal_dcg == 0.0:
        return 0.0

    return round(actual_dcg / ideal_dcg, 6)


def compute_retrieval_metrics(relevance_labels: List[int], k: int = 5) -> dict:
    """
    Compute P@k and nDCG@k for a single query's retrieval result.

    Parameters
    ----------
    relevance_labels : ordered list of binary relevance (1/0) for top-k docs
    k                : cut-off

    Returns
    -------
    dict: {precision_at_k, ndcg_at_k}
    """
    return {
        f"precision_at_{k}": precision_at_k(relevance_labels, k),
        f"ndcg_at_{k}":      ndcg_at_k(relevance_labels, k),
    }


def aggregate_retrieval_metrics(all_labels: List[List[int]], k: int = 5) -> dict:
    """
    Aggregate P@k and nDCG@k across multiple queries (mean ± std).

    Parameters
    ----------
    all_labels : list of per-query relevance_labels lists

    Returns
    -------
    dict with mean and std for each metric
    """
    import statistics

    pk_scores   = [precision_at_k(labels, k) for labels in all_labels]
    ndcg_scores = [ndcg_at_k(labels, k)      for labels in all_labels]

    def _stats(vals: List[float]) -> dict:
        n    = len(vals)
        mean = sum(vals) / n if n else 0.0
        std  = statistics.stdev(vals) if n > 1 else 0.0
        return {"mean": round(mean, 6), "std": round(std, 6), "n": n}

    return {
        f"precision_at_{k}": _stats(pk_scores),
        f"ndcg_at_{k}":      _stats(ndcg_scores),
    }
