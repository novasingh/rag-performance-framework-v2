"""
rq2_experiment/analysis/source_contribution.py
================================================
Computes the INDIVIDUAL CONTRIBUTION of each source type addition to performance.

Analyses:
  1. Marginal gain analysis: Academic → +News, +News → +Tech (per domain)
  2. Source Diversity Index (SDI) regression against each metric
  3. Domain-moderated SDI regression (SDI × volatility interaction)

These directly answer the "individual contribution" part of RQ2.
"""
from __future__ import annotations

import json
import logging
import statistics
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .source_type_anova import build_rq2_df, load_all_metrics
from ..config import (
    ANALYSIS_DIR_RQ2,
    PRIMARY_METRICS,
    SOURCE_LEVEL_LABELS,
    VOLATILITY_SCORE,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Marginal gain analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_marginal_gains(df: pd.DataFrame) -> Dict[str, Any]:
    """
    For each domain and each metric, compute the marginal gain from:
      Step A: Academic Only → Academic + News           (Level 1 → 2)
      Step B: Academic + News → Academic + News + Tech  (Level 2 → 3)
    """
    results: Dict[str, Any] = {}

    for metric in PRIMARY_METRICS:
        if metric not in df.columns:
            continue
        results[metric] = {}

        for domain in df["domain"].unique():
            sub = df[df["domain"] == domain]

            l1 = sub[sub["source_level"] == 1][metric].tolist()
            l2 = sub[sub["source_level"] == 2][metric].tolist()
            l3 = sub[sub["source_level"] == 3][metric].tolist()

            mean_l1 = statistics.mean(l1) if l1 else None
            mean_l2 = statistics.mean(l2) if l2 else None
            mean_l3 = statistics.mean(l3) if l3 else None

            step_a = round(mean_l2 - mean_l1, 6) if (mean_l2 is not None and mean_l1 is not None) else None
            step_b = round(mean_l3 - mean_l2, 6) if (mean_l3 is not None and mean_l2 is not None) else None
            total  = round(mean_l3 - mean_l1, 6) if (mean_l3 is not None and mean_l1 is not None) else None

            # Percentage share of each step in total gain
            step_a_pct = round(step_a / total * 100, 2) if (step_a is not None and total and total != 0) else None
            step_b_pct = round(step_b / total * 100, 2) if (step_b is not None and total and total != 0) else None

            results[metric][domain] = {
                "mean_academic_only":        round(mean_l1, 6) if mean_l1 is not None else None,
                "mean_academic_plus_news":   round(mean_l2, 6) if mean_l2 is not None else None,
                "mean_full_diversity":       round(mean_l3, 6) if mean_l3 is not None else None,
                "step_a_gain_adding_news":   step_a,
                "step_b_gain_adding_tech":   step_b,
                "total_gain_l1_to_l3":       total,
                "step_a_pct_of_total":       step_a_pct,
                "step_b_pct_of_total":       step_b_pct,
                "dominant_step":             (
                    "adding_news" if (step_a is not None and step_b is not None and abs(step_a) >= abs(step_b))
                    else "adding_tech" if (step_a is not None and step_b is not None)
                    else None
                ),
            }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. SDI (Source Diversity Index) regression
# ─────────────────────────────────────────────────────────────────────────────

def sdi_regression(df: pd.DataFrame) -> Dict[str, Any]:
    """
    OLS regression: metric ~ SDI  (simple linear)
    and:            metric ~ SDI + volatility_score + SDI*volatility_score  (moderated)
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return {"error": "statsmodels required"}

    results: Dict[str, Any] = {}

    for metric in PRIMARY_METRICS:
        if metric not in df.columns:
            continue

        data = df[["source_diversity_index", "volatility_score", metric]].dropna().copy()
        data.columns = ["sdi", "vol", "y"]

        # Simple regression: y ~ sdi
        simple = smf.ols("y ~ sdi", data=data).fit()
        # Moderated: y ~ sdi + vol + sdi:vol
        moderated = smf.ols("y ~ sdi + vol + sdi:vol", data=data).fit()

        results[metric] = {
            "simple_regression": {
                "r_squared":     round(float(simple.rsquared), 6),
                "adj_r_squared": round(float(simple.rsquared_adj), 6),
                "sdi_coef":      round(float(simple.params.get("sdi", 0.0)), 6),
                "sdi_pvalue":    round(float(simple.pvalues.get("sdi", 1.0)), 6),
                "sdi_significant": bool(simple.pvalues.get("sdi", 1.0) < 0.05),
                "intercept":     round(float(simple.params.get("Intercept", 0.0)), 6),
            },
            "moderated_regression": {
                "r_squared":         round(float(moderated.rsquared), 6),
                "adj_r_squared":     round(float(moderated.rsquared_adj), 6),
                "sdi_coef":          round(float(moderated.params.get("sdi", 0.0)), 6),
                "vol_coef":          round(float(moderated.params.get("vol", 0.0)), 6),
                "interaction_coef":  round(float(moderated.params.get("sdi:vol", 0.0)), 6),
                "interaction_pval":  round(float(moderated.pvalues.get("sdi:vol", 1.0)), 6),
                "interaction_sig":   bool(moderated.pvalues.get("sdi:vol", 1.0) < 0.05),
                "interpretation": (
                    "SDI effect is moderated by domain volatility (interaction significant)"
                    if moderated.pvalues.get("sdi:vol", 1.0) < 0.05
                    else "SDI effect does not significantly vary by domain volatility"
                ),
            },
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Source config performance profiles
# ─────────────────────────────────────────────────────────────────────────────

def compute_performance_profiles(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Full performance profile for each source config level:
    means, std devs, and rank ordering across all metrics.
    Used for radar/spider charts.
    """
    profiles: Dict[str, Any] = {}
    all_metrics = PRIMARY_METRICS + ["rouge_l", "meteor"]

    for level, label in SOURCE_LEVEL_LABELS.items():
        sub = df[df["source_level"] == level]
        profiles[label] = {}
        for metric in all_metrics:
            if metric not in sub.columns:
                continue
            vals = sub[metric].dropna().tolist()
            profiles[label][metric] = {
                "mean": round(float(sub[metric].mean()), 6) if vals else None,
                "std":  round(float(sub[metric].std()),  6) if len(vals) > 1 else None,
                "n":    len(vals),
            }

    # Rank source levels for each metric (1 = best)
    rankings: Dict[str, Any] = {}
    for metric in all_metrics:
        means = {
            label: profiles[label].get(metric, {}).get("mean", None)
            for label in profiles
        }
        valid = {k: v for k, v in means.items() if v is not None}
        # For hallucination_rate, lower is better
        reverse = metric != "hallucination_rate"
        ranked = sorted(valid, key=lambda k: valid[k], reverse=reverse)
        rankings[metric] = {label: (i + 1) for i, label in enumerate(ranked)}

    return {"profiles": profiles, "rankings": rankings}


# ─────────────────────────────────────────────────────────────────────────────
# Full runner
# ─────────────────────────────────────────────────────────────────────────────

def run_source_contribution_analysis() -> Dict[str, Any]:
    """
    Run full source contribution analysis. Saves rq2_contribution_results.json.
    """
    all_metrics = load_all_metrics()
    df = build_rq2_df(all_metrics)
    logger.info("Running RQ2 source contribution analysis on %d conditions …", len(df))

    results = {
        "marginal_gains":        compute_marginal_gains(df),
        "sdi_regression":        sdi_regression(df),
        "performance_profiles":  compute_performance_profiles(df),
    }

    out_path = ANALYSIS_DIR_RQ2 / "rq2_contribution_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("RQ2 contribution results saved → %s", out_path)
    return results


# Expose build_rq2_df for use by other modules
__all__ = ["run_source_contribution_analysis", "build_rq2_df", "load_all_metrics"]
