"""
rq2_experiment/analysis/source_type_anova.py
============================================
One-way and two-way ANOVA with source type as the primary factor.

Answers RQ2:
  "What is the individual contribution of source type configurations to RAG
   system performance across multiple evaluation metrics, and does this
   contribution vary by domain volatility?"

Statistical approach (Report.txt §3.5 / Table 3.6):
  1. One-way ANOVA: source_config_level → metric  (global)
  2. Two-way ANOVA: source_config_level × domain_volatility → metric  (interaction)
  3. Post-hoc Tukey HSD with Bonferroni correction
  4. Effect sizes: η² (eta-squared) per factor
"""
from __future__ import annotations

import json
import logging
import math
import statistics
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, levene, shapiro

from ..config import (
    ANALYSIS_DIR_RQ2,
    ALL_CONDITIONS,
    CONDITION_META,
    METRICS_DIR,
    PRIMARY_METRICS,
    SOURCE_LEVEL_LABELS,
    SOURCE_LEVEL_MAP,
    VOLATILITY_SCORE,
)
from rq1_experiment.analysis.anova import expand_df_for_anova

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_all_metrics() -> Dict[str, Dict]:
    path = METRICS_DIR / "all_conditions_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Run RQ1 evaluation first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_rq2_df(all_metrics: Dict[str, Dict]) -> pd.DataFrame:
    """
    Build a flat DataFrame enriched with RQ2-specific columns:
    source_level (1/2/3), source_level_label, domain_volatility_score.
    """
    rows = []
    for cid, m in all_metrics.items():
        meta = CONDITION_META.get(cid, {})
        domain = meta.get("domain", "")
        row = {
            "condition_id":          cid,
            "domain":                domain,
            "volatility":            meta.get("volatility", ""),
            "volatility_score":      VOLATILITY_SCORE.get(meta.get("volatility", ""), 0),
            "freshness_window":      meta.get("freshness", ""),
            "source_config":         meta.get("source_config", ""),
            "source_level":          SOURCE_LEVEL_MAP.get(cid, 0),
            "source_level_label":    SOURCE_LEVEL_LABELS.get(SOURCE_LEVEL_MAP.get(cid, 0), ""),
            "n_records":             m.get("n_records", 0),
            # Primary metrics
            "bertscore_f1":          m.get("bertscore_f1", {}).get("mean", 0.0),
            "bertscore_f1_std":      m.get("bertscore_f1", {}).get("std", 0.0),
            "hallucination_rate":    m.get("hallucination", {}).get("mean", 0.0),
            "hallucination_rate_std":m.get("hallucination", {}).get("std", 0.0),
            "precision_at_5":        m.get("retrieval", {}).get("precision_at_5", {}).get("mean", 0.0),
            "precision_at_5_std":    m.get("retrieval", {}).get("precision_at_5", {}).get("std", 0.0),
            "ndcg_at_5":             m.get("retrieval", {}).get("ndcg_at_5", {}).get("mean", 0.0),
            "ndcg_at_5_std":         m.get("retrieval", {}).get("ndcg_at_5", {}).get("std", 0.0),
            "human_eval_score":      m.get("human_eval_score", {}).get("mean", 0.0),
            "human_eval_score_std":  m.get("human_eval_score", {}).get("std", 0.0),
            # Secondary
            "rouge_l":               m.get("rouge_l", {}).get("mean", 0.0),
            "meteor":                m.get("meteor", {}).get("mean", 0.0),
            # Factor metrics
            "freshness_score":       m.get("freshness_score", {}).get("mean", 0.0),
            "source_diversity_index": m.get("source_diversity_index", {}).get("mean", 0.0),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Effect size helpers
# ─────────────────────────────────────────────────────────────────────────────

def eta_squared(ss_effect: float, ss_total: float) -> float:
    return round(ss_effect / ss_total, 6) if ss_total > 0 else 0.0


def interpret_eta2(eta2: float) -> str:
    if eta2 >= 0.14: return "large"
    if eta2 >= 0.06: return "medium"
    if eta2 >= 0.01: return "small"
    return "negligible"


def cohens_d(g1: List[float], g2: List[float]) -> float:
    if len(g1) < 2 or len(g2) < 2:
        return 0.0
    n1, n2 = len(g1), len(g2)
    v1, v2 = statistics.variance(g1), statistics.variance(g2)
    pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return round((statistics.mean(g1) - statistics.mean(g2)) / pooled, 6) if pooled else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# One-way ANOVA: source level → metric
# ─────────────────────────────────────────────────────────────────────────────

def one_way_anova_source(df: pd.DataFrame, metric: str, alpha: float = 0.05) -> Dict[str, Any]:
    """One-way ANOVA: does source_level significantly affect metric?"""
    expanded_df = expand_df_for_anova(df, metric)
    groups = []
    group_labels = []
    for level in [1, 2, 3]:
        vals = expanded_df[expanded_df["source_level"] == level][metric].dropna().tolist()
        if vals:
            groups.append(vals)
            group_labels.append(SOURCE_LEVEL_LABELS[level])

    if len(groups) < 2:
        return {"error": "Not enough groups for ANOVA"}

    f_stat, p_val = f_oneway(*groups)

    # Manual SS for η²
    all_vals = [v for g in groups for v in g]
    grand_mean = statistics.mean(all_vals)
    ss_total   = sum((v - grand_mean) ** 2 for v in all_vals)
    ss_between = sum(len(g) * (statistics.mean(g) - grand_mean) ** 2 for g in groups)
    eta2 = eta_squared(ss_between, ss_total)

    # Pairwise Cohen's d
    pairwise = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            pairwise.append({
                "group1": group_labels[i],
                "group2": group_labels[j],
                "cohens_d": cohens_d(groups[i], groups[j]),
                "mean_diff": round(statistics.mean(groups[i]) - statistics.mean(groups[j]), 6),
            })

    return {
        "metric":       metric,
        "factor":       "source_config_level",
        "n":            len(all_vals),
        "group_means":  {label: round(statistics.mean(g), 6) for label, g in zip(group_labels, groups)},
        "F":            round(float(f_stat), 6),
        "p_value":      round(float(p_val), 6),
        "significant":  bool(p_val < alpha),
        "eta_squared":  eta2,
        "effect_size_label": interpret_eta2(eta2),
        "pairwise_cohens_d": pairwise,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Two-way ANOVA: source_level × domain_volatility → metric
# ─────────────────────────────────────────────────────────────────────────────

def two_way_anova_source_x_volatility(df: pd.DataFrame, metric: str, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Two-way ANOVA with statsmodels: source_level × domain_volatility → metric.
    This answers whether source type effects vary by domain volatility.
    """
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
    except ImportError:
        return {"error": "statsmodels not available"}

    expanded_df = expand_df_for_anova(df, metric)
    data = expanded_df[["source_level", "volatility", metric]].copy().dropna()
    data["source_level"] = data["source_level"].astype(str)
    data["volatility"]   = pd.Categorical(data["volatility"], categories=["low", "medium", "high"], ordered=True)

    formula = f"{metric} ~ C(source_level) + C(volatility) + C(source_level):C(volatility)"
    model   = smf.ols(formula, data=data).fit()
    table   = anova_lm(model, typ=3)

    ss_total = float(model.ess + model.ssr)
    result = {"metric": metric, "n": len(data), "r_squared": round(float(model.rsquared), 6), "factors": {}}

    for term in table.index:
        if term in ("Intercept", "Residual"):
            continue
        row      = table.loc[term]
        ss       = float(row.get("sum_sq", 0.0))
        f_val    = float(row.get("F", 0.0))
        p_val    = float(row.get("PR(>F)", 1.0))
        eta2     = eta_squared(ss, ss_total)

        if "source_level" in term and "volatility" not in term:
            label = "source_type"
        elif "volatility" in term and "source_level" not in term:
            label = "domain_volatility"
        else:
            label = "interaction"

        result["factors"][label] = {
            "term":              term,
            "F":                 round(f_val, 6),
            "p_value":           round(p_val, 6),
            "significant":       bool(p_val < alpha),
            "eta_squared":       eta2,
            "effect_size_label": interpret_eta2(eta2),
        }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Tukey HSD post-hoc by source level
# ─────────────────────────────────────────────────────────────────────────────

def tukey_hsd_source(df: pd.DataFrame, metric: str, alpha: float = 0.05) -> Dict[str, Any]:
    """Post-hoc Tukey HSD comparing the 3 source config levels."""
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
    except ImportError:
        return {"error": "statsmodels required"}

    expanded_df = expand_df_for_anova(df, metric)
    sub = expanded_df[["source_level_label", metric]].dropna()
    tukey = pairwise_tukeyhsd(
        endog=sub[metric].values,
        groups=sub["source_level_label"].values,
        alpha=alpha,
    )

    comparisons = []
    for row in tukey.summary().data[1:]:
        g1, g2, mean_diff, p_adj, lower, upper, reject = row
        g1_vals = sub[sub["source_level_label"] == g1][metric].tolist()
        g2_vals = sub[sub["source_level_label"] == g2][metric].tolist()
        comparisons.append({
            "group1":      g1,
            "group2":      g2,
            "mean_diff":   round(float(mean_diff), 6),
            "p_adjusted":  round(float(p_adj), 6),
            "ci_lower":    round(float(lower), 6),
            "ci_upper":    round(float(upper), 6),
            "reject_h0":   bool(reject),
            "cohens_d":    cohens_d(g1_vals, g2_vals),
        })
    return {"metric": metric, "comparisons": comparisons}


# ─────────────────────────────────────────────────────────────────────────────
# Per-domain stratified analysis
# ─────────────────────────────────────────────────────────────────────────────

def anova_by_domain(df: pd.DataFrame, metric: str, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Runs one-way ANOVA (source level → metric) separately for each domain.
    Answers: does the source type effect size vary by domain volatility?
    """
    results = {}
    for domain in df["domain"].unique():
        sub = df[df["domain"] == domain]
        results[domain] = one_way_anova_source(sub, metric, alpha)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Full runner
# ─────────────────────────────────────────────────────────────────────────────

def run_source_type_anova(alpha: float = 0.05) -> Dict[str, Any]:
    """
    Run full source-type ANOVA analysis. Saves rq2_anova_results.json.
    """
    all_metrics = load_all_metrics()
    df = build_rq2_df(all_metrics)
    logger.info("Running RQ2 source-type ANOVA on %d conditions …", len(df))

    results: Dict[str, Any] = {
        "descriptive_by_source_level": {},
        "one_way_anova":               {},
        "two_way_anova":               {},
        "posthoc_tukey":               {},
        "per_domain_anova":            {},
    }

    # Descriptive stats by source level
    for level, label in SOURCE_LEVEL_LABELS.items():
        sub = df[df["source_level"] == level]
        results["descriptive_by_source_level"][label] = {
            m: round(float(sub[m].mean()), 6)
            for m in PRIMARY_METRICS + ["rouge_l", "meteor", "source_diversity_index"]
            if m in sub.columns
        }

    for metric in PRIMARY_METRICS:
        if metric not in df.columns:
            continue
        logger.info("  ANOVA for %s …", metric)
        results["one_way_anova"][metric]    = one_way_anova_source(df, metric, alpha)
        results["two_way_anova"][metric]    = two_way_anova_source_x_volatility(df, metric, alpha)
        results["posthoc_tukey"][metric]    = tukey_hsd_source(df, metric, alpha)
        results["per_domain_anova"][metric] = anova_by_domain(df, metric, alpha)

    out_path = ANALYSIS_DIR_RQ2 / "rq2_anova_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("RQ2 ANOVA results saved → %s", out_path)
    return results
