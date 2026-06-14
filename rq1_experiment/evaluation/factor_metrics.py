"""
rq1_experiment/evaluation/factor_metrics.py
============================================
Factor-level metrics that operationalize the independent variables
as continuous quantities for regression and framework phases.

Per Report.pdf Table 3.3:
  - Freshness Score    : (retrieval_date - publication_date) / max_observed_age, normalized to [0,1]
  - Source Diversity Index : Shannon entropy H = -Σ p·log₂(p) over source-type proportions

These are used in multiple regression and Random Forest (not as primary DVs).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional


_MAX_AGE_DAYS: Dict[str, int] = {
    "technology": 365,
    "healthcare":  730,
    "history":    3650,
}


def compute_freshness_score(
    publication_date_str: str,
    domain: str,
    retrieval_date: Optional[datetime] = None,
) -> float:
    """
    Freshness score = 1 - (age_days / max_age_days), clipped to [0, 1].

    Higher score = fresher document.

    Parameters
    ----------
    publication_date_str : ISO date string (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS…)
    domain               : 'technology' | 'healthcare' | 'history'
    retrieval_date       : reference date (defaults to now)
    """
    if retrieval_date is None:
        retrieval_date = datetime.now(tz=timezone.utc)

    if not publication_date_str:
        return 0.0

    # Parse date
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S%z", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            pub = datetime.strptime(publication_date_str[:19], fmt[:len(publication_date_str[:19])])
            break
        except ValueError:
            continue
    else:
        return 0.0   # unparseable

    if pub.tzinfo is None:
        pub = pub.replace(tzinfo=timezone.utc)

    age_days = (retrieval_date - pub).days
    max_age  = _MAX_AGE_DAYS.get(domain, 365)

    score = 1.0 - (age_days / max_age)
    return round(max(0.0, min(1.0, score)), 6)


def compute_source_diversity_index(source_types: List[str]) -> float:
    """
    Shannon entropy H = -Σ p·log₂(p) over source-type proportions.

    Maximum value = log₂(n_types).
    Returns 0.0 for a homogeneous set.
    """
    if not source_types:
        return 0.0

    counts: Dict[str, int] = {}
    for st in source_types:
        counts[st] = counts.get(st, 0) + 1

    n = len(source_types)
    entropy = 0.0
    for cnt in counts.values():
        p = cnt / n
        if p > 0:
            entropy -= p * math.log2(p)

    return round(entropy, 6)


def compute_factor_metrics(output_record: dict) -> dict:
    """
    Derive factor metrics from a pipeline output record.

    Parameters
    ----------
    output_record : dict produced by RAGPipeline.run_single()

    Returns
    -------
    dict with:
      - freshness_score_mean   : mean freshness of retrieved docs
      - source_diversity_index : Shannon entropy of source types
    """
    retrieved_docs = output_record.get("retrieved_docs", [])

    # Freshness: already stored in retriever output
    fresh_mean = output_record.get("avg_freshness", 0.0)

    # Recompute from individual docs for precision
    source_types = [d.get("source_type", "unknown") for d in retrieved_docs]
    diversity    = compute_source_diversity_index(source_types)

    return {
        "freshness_score_mean":   fresh_mean,
        "source_diversity_index": diversity,
    }
