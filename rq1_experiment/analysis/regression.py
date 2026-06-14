"""
rq1_experiment/analysis/regression.py
=======================================
Multiple regression and non-linear curve fitting.

Per Report.pdf Table 3.6:
  - OLS regression: freshness_score + source_diversity_index + domain_volatility
    + interaction terms → each primary metric
  - Non-linear curve fitting: exponential/polynomial freshness decay by domain
  - Produces interpretable coefficients for the predictive framework
"""
from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from ..config import ANALYSIS_DIR, DOMAIN_VOLATILITY_SCORE, METRICS_DIR
from .anova import build_flat_df, load_all_metrics

logger = logging.getLogger(__name__)

PRIMARY_METRICS = [
    "bertscore_f1",
    "hallucination_rate",
    "precision_at_5",
    "ndcg_at_5",
    "human_eval_score",
]


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def _encode_volatility(domain: str) -> float:
    return DOMAIN_VOLATILITY_SCORE.get(domain, 0.5)


def build_regression_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix for regression:
      - freshness_score      (continuous, 0-1)
      - source_diversity     (continuous, Shannon entropy)
      - domain_volatility    (encoded: high=1.0, medium=0.5, low=0.0)
      - fresh_x_diversity    (interaction term)
      - fresh_x_volatility   (interaction term)
      - source_x_volatility  (interaction term)
    """
    feat = df.copy()
    feat["domain_volatility"]   = feat["domain"].apply(_encode_volatility)
    feat["fresh_x_diversity"]   = feat["freshness_score"] * feat["source_diversity_index"]
    feat["fresh_x_volatility"]  = feat["freshness_score"] * feat["domain_volatility"]
    feat["source_x_volatility"] = feat["source_diversity_index"] * feat["domain_volatility"]
    return feat


FEATURE_COLS = [
    "freshness_score",
    "source_diversity_index",
    "domain_volatility",
    "fresh_x_diversity",
    "fresh_x_volatility",
    "source_x_volatility",
]


# ─────────────────────────────────────────────────────────────────────────────
# OLS Multiple Regression
# ─────────────────────────────────────────────────────────────────────────────

def run_ols_regression(df: pd.DataFrame, metric: str) -> Dict[str, Any]:
    """
    Fit OLS regression: feature_cols → metric.

    Returns coefficients, R², adjusted R², p-values.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        logger.warning("statsmodels not installed — using numpy OLS")
        return _numpy_ols(df, metric)

    feat = build_regression_features(df)
    sub  = feat[FEATURE_COLS + [metric]].dropna()

    X = sm.add_constant(sub[FEATURE_COLS].values)
    y = sub[metric].values

    model  = sm.OLS(y, X).fit()
    coef_names = ["intercept"] + FEATURE_COLS

    coefficients = {}
    for name, coef, se, t, p in zip(
        coef_names,
        model.params,
        model.bse,
        model.tvalues,
        model.pvalues,
    ):
        coefficients[name] = {
            "coef":    round(float(coef), 6),
            "std_err": round(float(se), 6),
            "t_stat":  round(float(t), 6),
            "p_value": round(float(p), 6),
            "significant": bool(float(p) < 0.05),
        }

    return {
        "metric":        metric,
        "n":             int(len(sub)),
        "r_squared":     round(float(model.rsquared), 6),
        "adj_r_squared": round(float(model.rsquared_adj), 6),
        "f_statistic":   round(float(model.fvalue), 6),
        "f_p_value":     round(float(model.f_pvalue), 6),
        "coefficients":  coefficients,
    }


def _numpy_ols(df: pd.DataFrame, metric: str) -> Dict[str, Any]:
    """Pure numpy OLS fallback."""
    feat = build_regression_features(df)
    sub  = feat[FEATURE_COLS + [metric]].dropna()

    X = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in FEATURE_COLS])
    y = sub[metric].values

    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return {"metric": metric, "error": "OLS failed (singular matrix)"}

    y_pred  = X @ beta
    ss_res  = np.sum((y - y_pred) ** 2)
    ss_tot  = np.sum((y - y.mean()) ** 2)
    r2      = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n, k    = X.shape
    adj_r2  = 1 - (1 - r2) * (n - 1) / (n - k) if n > k else r2

    coef_names = ["intercept"] + FEATURE_COLS
    return {
        "metric":        metric,
        "n":             n,
        "r_squared":     round(float(r2), 6),
        "adj_r_squared": round(float(adj_r2), 6),
        "coefficients":  {
            name: {"coef": round(float(b), 6)}
            for name, b in zip(coef_names, beta)
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Non-linear Freshness Decay Curve Fitting
# ─────────────────────────────────────────────────────────────────────────────

def _exponential_decay(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """y = a * exp(-b * x) + c"""
    return a * np.exp(-b * x) + c


def _polynomial_decay(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """y = a * x^2 + b * x + c"""
    return a * x**2 + b * x + c


def fit_freshness_decay(df: pd.DataFrame, metric: str) -> Dict[str, Any]:
    """
    Fit both exponential and polynomial decay curves for freshness_score → metric,
    separately per domain. Returns best-fit parameters and R² for each domain.
    """
    results: Dict[str, Any] = {"metric": metric, "domains": {}}

    for domain, grp in df.groupby("domain"):
        sub = grp[["freshness_score", metric]].dropna()
        if len(sub) < 4:
            continue

        x = sub["freshness_score"].values
        y = sub[metric].values

        domain_result: Dict[str, Any] = {}

        for name, func, p0 in [
            ("exponential", _exponential_decay, [1.0, 1.0, 0.0]),
            ("polynomial",  _polynomial_decay,  [0.0, 1.0, 0.0]),
        ]:
            try:
                popt, _ = curve_fit(func, x, y, p0=p0, maxfev=5000)
                y_pred  = func(x, *popt)
                ss_res  = np.sum((y - y_pred) ** 2)
                ss_tot  = np.sum((y - y.mean()) ** 2)
                r2      = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

                domain_result[name] = {
                    "params":    [round(float(p), 6) for p in popt],
                    "r_squared": round(r2, 6),
                }
            except Exception as exc:
                domain_result[name] = {"error": str(exc)}

        results["domains"][domain] = domain_result

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_regression_analysis() -> Dict[str, Any]:
    """
    Run OLS regression and non-linear decay fitting for all primary metrics.
    Saves to ANALYSIS_DIR/regression_results.json.
    """
    all_metrics = load_all_metrics()
    df          = build_flat_df(all_metrics)
    feat_df     = build_regression_features(df)

    logger.info("Running regression analysis on %d conditions …", len(df))

    # Correlation matrix first
    corr_cols = FEATURE_COLS + PRIMARY_METRICS
    avail_cols = [c for c in corr_cols if c in feat_df.columns]
    corr_matrix = feat_df[avail_cols].corr().round(4).to_dict()

    results = {
        "correlation_matrix": corr_matrix,
        "ols":   {},
        "decay": {},
    }

    for metric in PRIMARY_METRICS:
        if metric not in feat_df.columns:
            continue
        results["ols"][metric]   = run_ols_regression(feat_df, metric)
        results["decay"][metric] = fit_freshness_decay(feat_df, metric)

    out_path = ANALYSIS_DIR / "regression_results.json"
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("Regression results saved → %s", out_path)
    return results
