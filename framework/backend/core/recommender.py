"""
core/recommender.py
===================
Generates decision-support recommendations **derived from the data and the
RQ3 model** — no hardcoded performance numbers.

For a requested (domain, freshness, source) configuration it computes:
- source-impact deltas from the real per-condition RQ1 metrics,
- recommended update cadence from model-derived freshness sensitivity,
- a deployment-readiness score from the model's predicted metrics,
- risks and a key finding generated from those computed values,
- an expected performance range from the model prediction's MAE band.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import (
    DOMAIN_VOLATILITY_SCORE,
    PRIMARY_METRICS,
    READINESS_WEIGHTS,
    UPDATE_CADENCE_BY_VOLATILITY,
    load_condition_meta,
    load_condition_metrics,
)
from .predictor import RAGPerformancePredictor

# Generic source levels exposed by the UI, ordered from least to most diverse.
SOURCE_LEVELS = ["single", "mix", "full"]
SOURCE_LEVEL_LABELS = {
    "single": "Single-Source",
    "mix": "Two-Source Mix",
    "full": "Full Diversity",
}


def _metric_mean(cond_metrics: dict, metric: str) -> Optional[float]:
    """Pull a metric mean from a condition's metrics block (handles nesting)."""
    if metric in ("precision_at_5", "ndcg_at_5"):
        node = cond_metrics.get("retrieval", {}).get(metric, {})
    elif metric == "hallucination_rate":
        node = cond_metrics.get("hallucination", {})
    else:
        node = cond_metrics.get(metric, {})
    val = node.get("mean")
    return float(val) if val is not None else None


def _conditions_for_domain(domain: str) -> List[dict]:
    """Return condition rows for a domain, enriched with SDI and metric means."""
    meta = load_condition_meta()
    metrics = load_condition_metrics()
    rows: List[dict] = []
    for cid, m in meta.items():
        if m.get("domain") != domain:
            continue
        cm = metrics.get(cid, {})
        sdi = cm.get("source_diversity_index", {}).get("mean")
        row = {
            "condition_id": cid,
            "source_config": m.get("source_config"),
            "freshness": m.get("freshness"),
            "source_diversity_index": float(sdi) if sdi is not None else 0.0,
        }
        for metric in PRIMARY_METRICS:
            row[metric] = _metric_mean(cm, metric)
        rows.append(row)
    return rows


def source_levels_for_domain(domain: str) -> List[dict]:
    """
    Aggregate a domain's conditions into ordered source levels by SDI.

    Conditions sharing the same source-diversity tier are averaged so that the
    comparison reflects source type, not freshness.
    """
    rows = _conditions_for_domain(domain)
    if not rows:
        return []

    distinct_sdi = sorted({round(r["source_diversity_index"], 4) for r in rows})
    tier_of = {sdi: i for i, sdi in enumerate(distinct_sdi)}

    buckets: Dict[int, List[dict]] = {}
    for r in rows:
        tier = tier_of[round(r["source_diversity_index"], 4)]
        buckets.setdefault(tier, []).append(r)

    levels: List[dict] = []
    for tier in sorted(buckets):
        group = buckets[tier]
        level: Dict[str, Any] = {
            "tier": tier,
            "source_diversity_index": round(
                sum(g["source_diversity_index"] for g in group) / len(group), 4
            ),
            "source_configs": sorted({g["source_config"] for g in group}),
            "n_conditions": len(group),
        }
        for metric in PRIMARY_METRICS:
            vals = [g[metric] for g in group if g[metric] is not None]
            level[metric] = round(sum(vals) / len(vals), 4) if vals else None
        levels.append(level)
    return levels


def _select_source_index(num_levels: int, source_config: str) -> int:
    """Map a generic UI source id onto a domain's available source tiers."""
    order = {"academic": 0, "single": 0,
             "academic+news": 1, "mix": 1,
             "full": 2}
    idx = order.get(source_config, 0)
    return min(idx, num_levels - 1)


def _freshness_sensitivity(predictor: RAGPerformancePredictor, domain: str,
                           sdi: float) -> float:
    """
    Model-derived freshness sensitivity: the absolute change in predicted
    Precision@5 when sweeping document age from very fresh to stale. Reported
    for transparency alongside the volatility-based cadence.
    """
    fresh = predictor.predict(domain, avg_age_days=3.0, source_diversity_index=sdi)
    stale = predictor.predict(domain, avg_age_days=365.0, source_diversity_index=sdi)
    p_fresh = fresh["predictions"].get("precision_at_5", {}).get("expected", 0.0)
    p_stale = stale["predictions"].get("precision_at_5", {}).get("expected", 0.0)
    return abs(p_fresh - p_stale)


def _cadence_for_domain(domain: str, sensitivity: float | None = None) -> Dict[str, Any]:
    """
    Recommended update cadence from domain volatility (the documented driver of
    staleness). The model-derived freshness sensitivity is attached for context.
    """
    volatility = DOMAIN_VOLATILITY_SCORE.get(domain, 0.5)
    cadence, rationale = UPDATE_CADENCE_BY_VOLATILITY[-1][1], UPDATE_CADENCE_BY_VOLATILITY[-1][2]
    for threshold, label, why in UPDATE_CADENCE_BY_VOLATILITY:
        if volatility >= threshold:
            cadence, rationale = label, why
            break
    out: Dict[str, Any] = {"cadence": cadence, "rationale": rationale, "volatility": volatility}
    if sensitivity is not None:
        out["freshness_sensitivity"] = round(sensitivity, 4)
    return out


def _normalize(metric: str, value: float) -> float:
    """Scale a metric to [0,1] where 1 = best, respecting direction."""
    lo, hi = PRIMARY_METRICS[metric]["scale"]
    norm = (value - lo) / (hi - lo) if hi > lo else 0.0
    norm = max(0.0, min(1.0, norm))
    if PRIMARY_METRICS[metric]["direction"] == "lower":
        norm = 1.0 - norm
    return norm


def _deployment_readiness(predictions: Dict[str, dict]) -> int:
    """Weighted readiness score (0-100) from predicted metrics."""
    score = 0.0
    total_w = 0.0
    for metric, weight in READINESS_WEIGHTS.items():
        pred = predictions.get(metric, {}).get("expected")
        if pred is None:
            continue
        score += weight * _normalize(metric, pred)
        total_w += weight
    return round(100 * score / total_w) if total_w else 0


def domain_guideline(predictor: RAGPerformancePredictor, domain: str,
                     boundary_counts: Optional[dict] = None) -> Dict[str, Any]:
    """
    Per-domain deployment guidance derived from the data and the RQ3 model.

    Combines: model-derived freshness sensitivity -> update cadence, the best
    measured source level, a source-pollution flag, a representative predicted
    hallucination rate, and RQ4 reliability (boundary-condition count).
    """
    levels = source_levels_for_domain(domain)
    scored = [lv for lv in levels if lv.get("precision_at_5") is not None]

    best = max(scored, key=lambda x: x["precision_at_5"]) if scored else None
    most_diverse = max(scored, key=lambda x: x["tier"]) if scored else None
    leaner = [lv for lv in scored if most_diverse and lv["tier"] < most_diverse["tier"]]
    pollution = bool(
        most_diverse and leaner
        and most_diverse["precision_at_5"] < max(l["precision_at_5"] for l in leaner)
    )

    # Representative configuration: best source level, moderately stale knowledge base.
    rep_sdi = best["source_diversity_index"] if best else 0.0
    rep = predictor.predict(domain, avg_age_days=90.0, source_diversity_index=rep_sdi)
    rep_preds = rep["predictions"]

    cadence = _cadence_for_domain(domain, _freshness_sensitivity(predictor, domain, rep_sdi))
    boundary = (boundary_counts or {}).get(domain)

    def _label(lv):
        return SOURCE_LEVEL_LABELS.get(
            SOURCE_LEVELS[lv["tier"]] if lv["tier"] < len(SOURCE_LEVELS) else "full",
            "Full Diversity",
        )

    return {
        "domain": domain,
        "volatility": DOMAIN_VOLATILITY_SCORE.get(domain, 0.5),
        "update_cadence": cadence,
        "best_source_level": _label(best) if best else None,
        "best_source_configs": best["source_configs"] if best else [],
        "source_pollution": pollution,
        "predicted": {
            "precision_at_5": rep_preds.get("precision_at_5", {}).get("expected"),
            "hallucination_rate": rep_preds.get("hallucination_rate", {}).get("expected"),
            "bertscore_f1": rep_preds.get("bertscore_f1", {}).get("expected"),
        },
        "boundary_conditions": boundary,
        "reliable": boundary == 0 if boundary is not None else None,
    }


def build_recommendations(predictor: RAGPerformancePredictor, domain: str,
                          source_config: str, prediction: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble fully data-derived recommendations for one configuration."""
    predictions = prediction.get("predictions", {})
    levels = source_levels_for_domain(domain)

    # ── Source impact: best vs worst source tier (real RQ1 numbers) ──────────
    source_impact: Dict[str, Any] = {}
    chosen_level: Optional[dict] = None
    if levels:
        idx = _select_source_index(len(levels), source_config)
        chosen_level = levels[idx]

        scored = [lv for lv in levels if lv.get("precision_at_5") is not None]
        if scored:
            best = max(scored, key=lambda lv: lv["precision_at_5"])
            worst = min(scored, key=lambda lv: lv["precision_at_5"])
            delta_pct = None
            if worst["precision_at_5"] > 0:
                delta_pct = round(
                    100 * (best["precision_at_5"] - worst["precision_at_5"])
                    / worst["precision_at_5"], 1
                )
            source_impact = {
                "best_level": SOURCE_LEVEL_LABELS.get(SOURCE_LEVELS[best["tier"]] if best["tier"] < len(SOURCE_LEVELS) else "full", "Full Diversity"),
                "best_precision": best["precision_at_5"],
                "worst_level": SOURCE_LEVEL_LABELS.get(SOURCE_LEVELS[worst["tier"]] if worst["tier"] < len(SOURCE_LEVELS) else "single", "Single-Source"),
                "worst_precision": worst["precision_at_5"],
                "delta_pct": delta_pct,
                "full_diversity_hurts": False,
            }
            # Source-pollution check: does the most diverse tier underperform a leaner tier?
            most_diverse = max(scored, key=lambda lv: lv["tier"])
            leaner = [lv for lv in scored if lv["tier"] < most_diverse["tier"]]
            if leaner and most_diverse["precision_at_5"] < max(l["precision_at_5"] for l in leaner):
                source_impact["full_diversity_hurts"] = True

    # ── Update cadence (volatility-based; the documented driver) ─────────────
    sdi = chosen_level["source_diversity_index"] if chosen_level else 0.0
    cadence = _cadence_for_domain(domain, _freshness_sensitivity(predictor, domain, sdi))

    # ── Deployment readiness ─────────────────────────────────────────────────
    readiness = _deployment_readiness(predictions)

    # ── Performance range (from model MAE band on precision) ─────────────────
    p5 = predictions.get("precision_at_5", {})
    performance_range = ""
    if p5:
        performance_range = (
            f"Expected Precision@5: {p5.get('expected', 0):.3f} "
            f"(95% band {p5.get('lower_bound', 0):.3f}-{p5.get('upper_bound', 0):.3f})"
        )

    # ── Risks + key finding (generated from computed values) ─────────────────
    risks: List[str] = []
    if source_impact.get("full_diversity_hurts") and source_config == "full":
        risks.append(
            "Source pollution: in this domain the most diverse source mix measured "
            "lower retrieval precision than a leaner mix. Consider dropping noisier sources."
        )
    if chosen_level and source_impact.get("best_precision") is not None:
        if chosen_level.get("precision_at_5") is not None and \
                chosen_level["precision_at_5"] < source_impact["best_precision"]:
            risks.append(
                f"The selected source mix underperforms the best measured mix "
                f"({source_impact['best_level']}) for this domain."
            )
    if "warnings" in prediction:
        risks.extend(prediction["warnings"])
    if not risks:
        risks.append("No significant risks detected for this configuration.")

    key_finding = ""
    if source_impact.get("delta_pct") is not None:
        key_finding = (
            f"Source mix matters here: moving from {source_impact['worst_level']} "
            f"({source_impact['worst_precision']:.3f}) to {source_impact['best_level']} "
            f"({source_impact['best_precision']:.3f}) changes Precision@5 by "
            f"{source_impact['delta_pct']:+.1f}%."
        )

    return {
        "update_frequency": f"{cadence['cadence']} - {cadence['rationale']}",
        "update_cadence": cadence,
        "source_mix": (
            ", ".join(chosen_level["source_configs"]) if chosen_level else source_config
        ),
        "source_level": (
            SOURCE_LEVEL_LABELS.get(SOURCE_LEVELS[chosen_level["tier"]], "Full Diversity")
            if chosen_level and chosen_level["tier"] < len(SOURCE_LEVELS) else "Full Diversity"
        ),
        "source_impact": source_impact,
        "performance_range": performance_range,
        "deployment_readiness": readiness,
        "risks": risks,
        "key_finding": key_finding,
    }
