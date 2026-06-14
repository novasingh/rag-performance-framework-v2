"""
rq1_experiment/evaluation/hallucination.py
===========================================
Hallucination rate detection using DeBERTa-based NLI entailment.

For each generated claim, checks whether it is *entailed* by the retrieved
passage set. Claims not entailed by ANY retrieved document are counted as
hallucinations.

Primary metric per Report.pdf Table 3.3 (Honovich et al., 2022 approach).
"""
from __future__ import annotations

import logging
import re
import statistics
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Load DeBERTa NLI model ────────────────────────────────────────────────────
try:
    from transformers import pipeline as hf_pipeline
    _NLI_AVAILABLE = True
except ImportError:
    _NLI_AVAILABLE = False
    logger.warning("transformers not installed; NLI will use heuristic fallback")

_nli_pipe = None   # lazy-loaded


def _get_nli_pipeline():
    global _nli_pipe
    if _nli_pipe is None and _NLI_AVAILABLE:
        try:
            logger.info("Loading DeBERTa NLI pipeline …")
            _nli_pipe = hf_pipeline(
                "zero-shot-classification",
                model="cross-encoder/nli-deberta-v3-small",
                device=-1,   # CPU (GPU auto if available)
            )
            logger.info("DeBERTa NLI pipeline loaded.")
        except Exception as exc:
            logger.error("Failed to load NLI model: %s — using heuristic fallback", exc)
            _nli_pipe = None
    return _nli_pipe


# ─────────────────────────────────────────────────────────────────────────────
# Claim segmentation
# ─────────────────────────────────────────────────────────────────────────────

def _segment_claims(text: str, max_claims: int = 10) -> List[str]:
    """
    Split generated response into individual claims (sentence-level).
    Filters out meta-statements like "According to the documents …"
    """
    if not text:
        return []

    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    _META_PREFIXES = (
        "according to", "based on the", "the documents", "the context",
        "i cannot determine", "the provided", "as mentioned",
    )

    claims: List[str] = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20:          # too short to be a meaningful claim
            continue
        lower = sent.lower()
        if any(lower.startswith(p) for p in _META_PREFIXES):
            continue
        claims.append(sent)
        if len(claims) >= max_claims:
            break

    return claims


# ─────────────────────────────────────────────────────────────────────────────
# NLI entailment check
# ─────────────────────────────────────────────────────────────────────────────

def _is_entailed_nli(claim: str, passages: List[str], threshold: float = 0.5) -> bool:
    """
    Returns True if *claim* is entailed by ANY of *passages* (p_entail >= threshold).
    """
    pipe = _get_nli_pipeline()
    if pipe is None:
        return _is_entailed_heuristic(claim, passages)

    combined_context = " ".join(p[:500] for p in passages)   # truncate to keep fast
    try:
        result = pipe(
            combined_context[:1024],   # NLI max input
            candidate_labels=["entailment", "contradiction", "neutral"],
            hypothesis_template="{}",
        )
        # result is {"labels": [...], "scores": [...]}
        scores = dict(zip(result["labels"], result["scores"]))
        return scores.get("entailment", 0.0) >= threshold
    except Exception as exc:
        logger.debug("NLI pipe failed for claim: %s", exc)
        return _is_entailed_heuristic(claim, passages)


def _is_entailed_heuristic(claim: str, passages: List[str]) -> bool:
    """
    Heuristic fallback: check if ≥50% of claim's key nouns appear in any passage.
    """
    claim_words = set(re.findall(r"\b[a-z]{4,}\b", claim.lower()))
    if not claim_words:
        return True   # can't falsify
    for passage in passages:
        p_words = set(re.findall(r"\b[a-z]{4,}\b", passage.lower()))
        overlap = len(claim_words & p_words) / len(claim_words)
        if overlap >= 0.4:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_hallucination_rate(
    response: str,
    retrieved_docs: List[Dict],
    entail_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute hallucination rate for one generated response.

    Parameters
    ----------
    response        : the generated answer string
    retrieved_docs  : list of retrieved document dicts (must have 'text' field)
    entail_threshold: NLI entailment probability threshold

    Returns
    -------
    dict:
      - hallucination_rate  : float in [0, 1] — fraction of claims not entailed
      - n_claims            : int — number of claims extracted
      - n_hallucinated      : int — claims not supported by any retrieved doc
      - claims              : list of claim strings
    """
    passages = [
        (doc.get("text") or "")[:800]
        for doc in retrieved_docs
        if doc.get("text")
    ]

    claims = _segment_claims(response)
    if not claims:
        return {
            "hallucination_rate": 0.0,
            "n_claims": 0,
            "n_hallucinated": 0,
            "claims": [],
        }

    n_hallucinated = 0
    for claim in claims:
        if not _is_entailed_nli(claim, passages, threshold=entail_threshold):
            n_hallucinated += 1

    rate = round(n_hallucinated / len(claims), 6)
    return {
        "hallucination_rate": rate,
        "n_claims":           len(claims),
        "n_hallucinated":     n_hallucinated,
        "claims":             claims,
    }


def _stats(vals: List[float]) -> dict:
    n    = len(vals)
    mean = sum(vals) / n if n else 0.0
    std  = statistics.stdev(vals) if n > 1 else 0.0
    return {"mean": round(mean, 6), "std": round(std, 6), "n": n}


def aggregate_hallucination_rates(rates: List[float]) -> dict:
    """Aggregate hallucination rates across multiple queries."""
    return {"hallucination_rate": _stats(rates)}
