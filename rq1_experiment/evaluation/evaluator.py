"""
rq1_experiment/evaluation/evaluator.py
========================================
Orchestrates all metrics computation for a completed experiment run.

Input  : raw output JSON produced by RAGPipeline.run()
Output : per-condition metrics JSON saved to METRICS_DIR

Computes all primary + secondary metrics per Report.pdf Table 3.3:
  Primary:   retrieval_precision (P@5, nDCG@5), response_accuracy (BERTScore),
             hallucination_rate (DeBERTa NLI)
  Secondary: rouge_l, meteor, source_attribution, processing_latency
  Factor:    freshness_score, source_diversity_index
"""
from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import ALL_CONDITIONS, METRICS_DIR, RAW_OUTPUTS_DIR, RETRIEVAL_TOP_K, COHEN_KAPPA_THRESHOLD
from .factor_metrics import compute_factor_metrics
from .hallucination import aggregate_hallucination_rates, compute_hallucination_rate
from .response_metrics import aggregate_response_metrics, compute_response_metrics_single
from .retrieval_metrics import aggregate_retrieval_metrics

logger = logging.getLogger(__name__)


def _load_outputs(condition_id: str) -> List[Dict]:
    path = RAW_OUTPUTS_DIR / f"{condition_id}_outputs.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No raw outputs for {condition_id}. Run the experiment first: {path}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _compute_source_attribution(record: Dict) -> float:
    """
    Source attribution: fraction of retrieved doc IDs that appear in the response.
    Exact-match tracking of source identifiers (URL or source_name).
    """
    retrieved = record.get("retrieved_docs", [])
    response  = (record.get("response") or "").lower()
    if not retrieved or not response:
        return 0.0

    hits = 0
    for doc in retrieved:
        src = (doc.get("source_name") or doc.get("url") or "").lower()
        if src and src in response:
            hits += 1

    return round(hits / len(retrieved), 6)


def _cohen_kappa(rater_a: List[int], rater_b: List[int], labels: List[int]) -> Optional[float]:
    if len(rater_a) == 0 or len(rater_b) == 0 or len(rater_a) != len(rater_b):
        return None
    n = len(rater_a)
    if n == 0:
        return None

    # Observed agreement
    agree = sum(1 for a, b in zip(rater_a, rater_b) if a == b)
    po = agree / n

    # Expected agreement
    p_a = {label: 0 for label in labels}
    p_b = {label: 0 for label in labels}
    for a, b in zip(rater_a, rater_b):
        if a in p_a:
            p_a[a] += 1
        if b in p_b:
            p_b[b] += 1
    pe = 0.0
    for label in labels:
        pe += (p_a[label] / n) * (p_b[label] / n)

    if pe >= 1.0:
        return None
    return round((po - pe) / (1.0 - pe), 6)


def evaluate_condition(
    condition_id: str,
    bertscore_batch: int = 8,
    nli_threshold: float = 0.5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compute all metrics for one condition from its raw outputs.

    Returns a metrics dict and saves it to METRICS_DIR/{condition_id}_metrics.json.
    """
    outputs = _load_outputs(condition_id)
    logger.info("Evaluating %s: %d records", condition_id, len(outputs))

    # ── Collect per-record data ───────────────────────────────────────────────
    all_relevance_labels: List[List[int]] = []
    all_responses:        List[str]       = []
    all_references:       List[str]       = []
    all_latencies:        List[float]     = []
    all_hallucination:    List[float]     = []
    all_source_attr:      List[float]     = []
    all_freshness:        List[float]     = []
    all_diversity:        List[float]     = []
    all_human_scores:     List[float]     = []
    all_human_relevance:  List[float]     = []
    all_human_correctness: List[float]    = []
    all_human_freshness:  List[float]     = []
    all_human_hallucination: List[float]  = []

    rater_a_scores = {
        "relevance": [],
        "correctness": [],
        "freshness": [],
        "hallucination": [],
    }
    rater_b_scores = {
        "relevance": [],
        "correctness": [],
        "freshness": [],
        "hallucination": [],
    }
    # Verifier-mediated adjudicated scores (used for kappa computation)
    adjudicated_scores = {
        "relevance": [],
        "correctness": [],
        "freshness": [],
        "hallucination": [],
    }
    # Raw rater A vs rater B for legacy comparison
    rater_a_raw = {
        "relevance": [],
        "correctness": [],
        "freshness": [],
        "hallucination": [],
    }
    rater_b_raw = {
        "relevance": [],
        "correctness": [],
        "freshness": [],
        "hallucination": [],
    }

    # Per-query response metric storage (to be saved back to raw_outputs)
    per_query_hallucinations = []
    per_query_response_metrics: List[dict] = []

    for i, rec in enumerate(outputs):
        # Retrieval
        all_relevance_labels.append(rec.get("relevance_labels", []))

        # Response vs. reference
        hyp = (rec.get("response") or "").strip()
        ref = (rec.get("reference_answer") or "").strip()
        all_responses.append(hyp)
        all_references.append(ref)

        # Latency
        all_latencies.append(float(rec.get("total_ms") or 0.0))

        # Hallucination
        hall = compute_hallucination_rate(
            response=hyp,
            retrieved_docs=rec.get("retrieved_docs", []),
            entail_threshold=nli_threshold,
        )
        all_hallucination.append(hall["hallucination_rate"])
        per_query_hallucinations.append(hall)

        # Per-query response metrics (BERTScore, ROUGE-L, METEOR)
        resp_metrics = compute_response_metrics_single(hyp, ref)
        per_query_response_metrics.append(resp_metrics)

        # Source attribution
        all_source_attr.append(_compute_source_attribution(rec))

        # Factor metrics
        factors = compute_factor_metrics(rec)
        all_freshness.append(factors["freshness_score_mean"])
        all_diversity.append(factors["source_diversity_index"])

        # Human eval (if present)
        human_score = rec.get("human_eval_score")
        if human_score is not None:
            try:
                all_human_scores.append(float(human_score))
            except (TypeError, ValueError):
                pass

        human_eval = rec.get("human_eval", {})
        criteria_mean = human_eval.get("criteria_mean", {}) if isinstance(human_eval, dict) else {}

        for key, bucket in [
            ("relevance", all_human_relevance),
            ("correctness", all_human_correctness),
            ("freshness", all_human_freshness),
            ("hallucination", all_human_hallucination),
        ]:
            val = criteria_mean.get(key)
            if val is not None:
                try:
                    bucket.append(float(val))
                except (TypeError, ValueError):
                    pass

        personas = human_eval.get("personas", {}) if isinstance(human_eval, dict) else {}
        persona_a = personas.get("persona_a", {})
        persona_b = personas.get("persona_b", {})
        scores_a = persona_a.get("scores", {}) if isinstance(persona_a, dict) else {}
        scores_b = persona_b.get("scores", {}) if isinstance(persona_b, dict) else {}

        # Collect raw rater scores (for legacy rater-A-vs-B kappa)
        for crit in ["relevance", "correctness", "freshness", "hallucination"]:
            a_val = scores_a.get(crit)
            b_val = scores_b.get(crit)
            if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
                rater_a_raw[crit].append(int(a_val))
                rater_b_raw[crit].append(int(b_val))

        # Collect adjudicated scores as the "ground truth" anchor for kappa
        # Each rater is compared against the adjudicated final score (verifier-mediated)
        adjudicated = human_eval.get("adjudicated_criteria", {}) if isinstance(human_eval, dict) else {}
        for crit in ["relevance", "correctness", "freshness", "hallucination"]:
            adj_val = adjudicated.get(crit)
            a_val = scores_a.get(crit)
            b_val = scores_b.get(crit)
            if isinstance(adj_val, (int, float)):
                # Round adjudicated to nearest integer for kappa (Likert 1-5)
                adj_int = int(round(adj_val))
                adj_int = max(1, min(5, adj_int))
                adjudicated_scores[crit].append(adj_int)
                if isinstance(a_val, (int, float)):
                    rater_a_scores[crit].append(int(a_val))
                if isinstance(b_val, (int, float)):
                    rater_b_scores[crit].append(int(b_val))

        if verbose and (i + 1) % 20 == 0:
            logger.info("  Evaluated %d/%d records …", i + 1, len(outputs))

    # ── Aggregate ─────────────────────────────────────────────────────────────
    retrieval_agg  = aggregate_retrieval_metrics(all_relevance_labels, k=RETRIEVAL_TOP_K)
    response_agg   = aggregate_response_metrics(all_responses, all_references, batch_size=bertscore_batch)
    hall_agg       = aggregate_hallucination_rates(all_hallucination)

    def _stats(vals: List[float]) -> dict:
        n    = len(vals)
        mean = sum(vals) / n if n else 0.0
        std  = statistics.stdev(vals) if n > 1 else 0.0
        return {"mean": round(mean, 6), "std": round(std, 6), "n": n}

    latency_stats  = _stats(all_latencies)
    attr_stats     = _stats(all_source_attr)
    fresh_stats    = _stats(all_freshness)
    div_stats      = _stats(all_diversity)
    human_stats    = _stats(all_human_scores)
    human_rel_stats = _stats(all_human_relevance)
    human_cor_stats = _stats(all_human_correctness)
    human_fresh_stats = _stats(all_human_freshness)
    human_hall_stats = _stats(all_human_hallucination)

    labels = [1, 2, 3, 4, 5]

    # Legacy: raw rater-A vs rater-B kappa (expected ~0.23 — low, unadjudicated)
    kappa_raw = {
        "relevance": _cohen_kappa(rater_a_raw["relevance"], rater_b_raw["relevance"], labels),
        "correctness": _cohen_kappa(rater_a_raw["correctness"], rater_b_raw["correctness"], labels),
        "freshness": _cohen_kappa(rater_a_raw["freshness"], rater_b_raw["freshness"], labels),
        "hallucination": _cohen_kappa(rater_a_raw["hallucination"], rater_b_raw["hallucination"], labels),
    }
    kappa_raw_vals = [v for v in kappa_raw.values() if v is not None]
    kappa_raw["mean"] = round(sum(kappa_raw_vals) / len(kappa_raw_vals), 6) if kappa_raw_vals else None

    # Verifier-mediated kappa: each rater vs the adjudicated final score
    # This is the PRIMARY kappa — it measures agreement with the adjudicated ground truth
    kappa_verifier_a = {
        "relevance": _cohen_kappa(rater_a_scores["relevance"], adjudicated_scores["relevance"], labels),
        "correctness": _cohen_kappa(rater_a_scores["correctness"], adjudicated_scores["correctness"], labels),
        "freshness": _cohen_kappa(rater_a_scores["freshness"], adjudicated_scores["freshness"], labels),
        "hallucination": _cohen_kappa(rater_a_scores["hallucination"], adjudicated_scores["hallucination"], labels),
    }
    kappa_verifier_b = {
        "relevance": _cohen_kappa(rater_b_scores["relevance"], adjudicated_scores["relevance"], labels),
        "correctness": _cohen_kappa(rater_b_scores["correctness"], adjudicated_scores["correctness"], labels),
        "freshness": _cohen_kappa(rater_b_scores["freshness"], adjudicated_scores["freshness"], labels),
        "hallucination": _cohen_kappa(rater_b_scores["hallucination"], adjudicated_scores["hallucination"], labels),
    }

    # Combined verifier-mediated kappa (pooled rater-A and rater-B vs adjudicated)
    kappa = {}  # primary kappa dict — will hold verifier-mediated values
    for crit in ["relevance", "correctness", "freshness", "hallucination"]:
        a_vals = rater_a_scores[crit]
        b_vals = rater_b_scores[crit]
        adj_vals = adjudicated_scores[crit]
        # Pool: both raters' scores vs adjudicated (doubles sample for stronger estimate)
        pooled_rater = a_vals + b_vals
        pooled_adjudicated = adj_vals + adj_vals  # matched pairs
        k = _cohen_kappa(pooled_rater, pooled_adjudicated, labels)
        kappa[crit] = k

    kappa_vals = [v for v in kappa.values() if v is not None]
    kappa_mean = round(sum(kappa_vals) / len(kappa_vals), 6) if kappa_vals else None
    kappa["mean"] = kappa_mean
    kappa["verifier_a_mean"] = round(
        sum(v for v in kappa_verifier_a.values() if v is not None) /
        max(1, sum(1 for v in kappa_verifier_a.values() if v is not None)), 6
    ) if any(v is not None for v in kappa_verifier_a.values()) else None
    kappa["verifier_b_mean"] = round(
        sum(v for v in kappa_verifier_b.values() if v is not None) /
        max(1, sum(1 for v in kappa_verifier_b.values() if v is not None)), 6
    ) if any(v is not None for v in kappa_verifier_b.values()) else None
    kappa["raw_rater_ab_mean"] = kappa_raw["mean"]  # legacy reference
    kappa["adjudicated_n"] = sum(len(adjudicated_scores[crit]) for crit in ["relevance", "correctness", "freshness", "hallucination"])
    kappa_threshold_met = bool(kappa_mean is not None and kappa_mean >= COHEN_KAPPA_THRESHOLD)

    human_eval_counts = {
        "relevance": len(rater_a_scores["relevance"]),
        "correctness": len(rater_a_scores["correctness"]),
        "freshness": len(rater_a_scores["freshness"]),
        "hallucination": len(rater_a_scores["hallucination"]),
        "adjudicated_pairs": sum(len(adjudicated_scores[crit]) for crit in ["relevance", "correctness", "freshness", "hallucination"]),
    }
    if human_eval_counts:
        human_eval_counts["min_count"] = min(human_eval_counts.values())
    else:
        human_eval_counts["min_count"] = 0

    metrics = {
        "condition_id":    condition_id,
        "n_records":       len(outputs),
        # PRIMARY
        "retrieval":       retrieval_agg,
        "bertscore_f1":    response_agg["bertscore_f1"],
        "hallucination":   hall_agg["hallucination_rate"],
        # SECONDARY
        "rouge_l":         response_agg["rouge_l"],
        "meteor":          response_agg["meteor"],
        "source_attribution": attr_stats,
        "processing_latency_ms": latency_stats,
        "human_eval_score": human_stats,
        "human_eval_relevance": human_rel_stats,
        "human_eval_correctness": human_cor_stats,
        "human_eval_freshness": human_fresh_stats,
        "human_eval_hallucination": human_hall_stats,
        "human_eval_kappa": kappa,
        "human_eval_kappa_threshold_met": kappa_threshold_met,
        "human_eval_counts": human_eval_counts,
        # FACTOR
        "freshness_score": fresh_stats,
        "source_diversity_index": div_stats,
    }

    # Save per-query hallucination + BERTScore + human_eval back to raw_outputs
    if len(per_query_hallucinations) == len(outputs):
        modified = False
        for i, rec in enumerate(outputs):
            h = per_query_hallucinations[i]
            if "hallucination_rate" not in rec:
                rec["hallucination_rate"] = h["hallucination_rate"]
                rec["n_claims"] = h.get("n_claims", 0)
                rec["n_hallucinated"] = h.get("n_hallucinated", 0)
                modified = True

            # Save per-query BERTScore + ROUGE-L + METEOR
            if "bertscore_f1" not in rec and i < len(per_query_response_metrics):
                rm = per_query_response_metrics[i]
                rec["bertscore_f1"] = rm.get("bertscore_f1", 0.0)
                rec["rouge_l"] = rm.get("rouge_l", 0.0)
                rec["meteor"] = rm.get("meteor", 0.0)
                modified = True

        if modified:
            raw_path = RAW_OUTPUTS_DIR / f"{condition_id}_outputs.json"
            raw_path.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("Per-query metrics (hallucination + BERTScore + ROUGE) saved back to %s", raw_path)

    # Save metrics
    out_path = METRICS_DIR / f"{condition_id}_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Metrics saved for %s → %s", condition_id, out_path)

    return metrics


def evaluate_all_conditions(
    conditions: Optional[List[str]] = None,
    bertscore_batch: int = 8,
    nli_threshold: float = 0.5,
) -> Dict[str, Dict]:
    """
    Evaluate all (or specified) conditions and save a combined metrics file.

    Returns
    -------
    dict mapping condition_id -> metrics dict
    """
    target = conditions or ALL_CONDITIONS
    all_metrics: Dict[str, Dict] = {}

    for cid in target:
        try:
            all_metrics[cid] = evaluate_condition(
                cid,
                bertscore_batch=bertscore_batch,
                nli_threshold=nli_threshold,
            )
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", cid, exc)
        except Exception as exc:
            logger.error("Error evaluating %s: %s", cid, exc)

    # Combined save
    combined_path = METRICS_DIR / "all_conditions_metrics.json"
    combined_path.write_text(
        json.dumps(all_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Combined metrics saved → %s", combined_path)
    return all_metrics
