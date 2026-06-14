"""
rq1_experiment/rag_system/retriever.py
========================================
k=5 FAISS exact retrieval — returns top-k documents and similarity scores.

Also computes:
- Relevance judgements for P@5 / nDCG@5 (keyword-based if no ground truth)
- Source diversity index (Shannon entropy across source types in top-k)
- Freshness score (normalized age) for retrieved documents
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from ..config import RETRIEVAL_TOP_K
from .embedder import DocumentEmbedder

logger = logging.getLogger(__name__)


def _shannon_entropy(source_types: List[str]) -> float:
    """Shannon entropy H = -Σ p·log(p) over source-type proportions."""
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


def _avg_freshness_score(docs: List[Dict]) -> float:
    """Average freshness_score of the retrieved set (0-1, higher = fresher)."""
    scores = [float(d.get("freshness_score") or 0.0) for d in docs]
    return round(sum(scores) / len(scores), 6) if scores else 0.0


class Retriever:
    """
    Retrieves top-k documents for a query from a pre-built FAISS index.

    Parameters
    ----------
    index    : FAISS IndexFlatIP (exact inner-product search)
    docs     : list of document dicts aligned with the index (same order)
    embedder : DocumentEmbedder instance (shared, not recreated per retrieval)
    k        : number of documents to retrieve (default 5, per protocol)
    """

    def __init__(
        self,
        index: faiss.IndexFlatIP,
        docs: List[Dict],
        embedder: DocumentEmbedder,
        k: int = RETRIEVAL_TOP_K,
    ) -> None:
        self.index   = index
        self.docs    = docs
        self.embedder = embedder
        self.k       = k

    def retrieve(
        self,
        query: str,
        relevance_keywords: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve top-k documents for *query*.

        Parameters
        ----------
        query               : the question string
        relevance_keywords  : optional list of gold-standard keywords for binary
                              relevance judgement (used to compute P@5, nDCG@5)

        Returns
        -------
        dict with:
          - retrieved_docs     : List[dict] — top-k documents (ranked)
          - similarities       : List[float] — cosine similarities
          - relevance_labels   : List[int] — binary 0/1 per doc (if keywords given)
          - source_diversity   : float — Shannon entropy of source types in top-k
          - avg_freshness      : float — average freshness_score of retrieved set
        """
        query_vec = self.embedder.embed_query(query)   # shape (1, dim)
        distances, indices = self.index.search(query_vec, self.k)

        top_docs: List[Dict]   = []
        similarities: List[float] = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.docs):
                continue
            top_docs.append(self.docs[idx])
            similarities.append(float(dist))

        # Binary relevance labels (keyword overlap with title+text)
        relevance_labels: List[int] = []
        if relevance_keywords:
            kw_lower = {kw.lower() for kw in relevance_keywords if kw}
            for doc in top_docs:
                blob = f"{doc.get('title','').lower()} {doc.get('text','').lower()}"
                hit  = sum(1 for kw in kw_lower if kw in blob)
                relevance_labels.append(1 if hit >= 1 else 0)
        else:
            # Default: treat all retrieved docs as relevant (conservative)
            relevance_labels = [1] * len(top_docs)

        source_types    = [doc.get("source_type", "unknown") for doc in top_docs]
        source_diversity = _shannon_entropy(source_types)
        avg_freshness    = _avg_freshness_score(top_docs)

        return {
            "retrieved_docs":   top_docs,
            "similarities":     similarities,
            "relevance_labels": relevance_labels,
            "source_diversity": source_diversity,
            "avg_freshness":    avg_freshness,
        }
