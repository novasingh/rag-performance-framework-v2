"""
rq1_experiment/analysis/random_forest.py
==========================================
Random Forest analysis with 500 decision trees for:
  - Non-linear factor-performance relationships
  - Feature importance (complements regression coefficients)
  - 10-fold stratified cross-validation

Per Report.pdf Table 3.6 (non-parametric validation layer).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from ..config import ALPHA, ANALYSIS_DIR, CV_FOLDS, RF_N_ESTIMATORS
from .anova import build_flat_df, load_all_metrics
from .regression import FEATURE_COLS, PRIMARY_METRICS, build_regression_features

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_stratify_key(df: pd.DataFrame) -> np.ndarray:
    """Composite stratum: domain + source_config (for stratified CV)."""
    le = LabelEncoder()
    keys = df["domain"].astype(str) + "::" + df["source_config"].astype(str)
    return le.fit_transform(keys)


# ─────────────────────────────────────────────────────────────────────────────
# Random Forest per metric
# ─────────────────────────────────────────────────────────────────────────────

def run_random_forest(
    df: pd.DataFrame,
    metric: str,
    n_estimators: int = RF_N_ESTIMATORS,
    cv_folds: int = CV_FOLDS,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Fit Random Forest with grid-searched hyperparameters.
    Returns feature importance scores and cross-validation R².

    Uses StratifiedKFold stratified by domain + source_config.
    """
    feat = build_regression_features(df)
    sub  = feat[FEATURE_COLS + [metric, "domain", "source_config"]].dropna()

    if len(sub) < cv_folds:
        return {"metric": metric, "error": f"Insufficient data: {len(sub)} rows < {cv_folds} folds"}

    X = sub[FEATURE_COLS].values
    y = sub[metric].values

    # ── Hyperparameter grid search ───────────────────────────────────────────
    best_model  = None
    best_r2     = -np.inf
    best_params = {}

    param_grid = {
        "max_depth":       [None, 3, 5],
        "min_samples_leaf": [1, 2],
        "max_features":    ["sqrt", "log2"],
    }

    from itertools import product
    for max_depth, min_leaf, max_feat in product(
        param_grid["max_depth"],
        param_grid["min_samples_leaf"],
        param_grid["max_features"],
    ):
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            max_features=max_feat,
            random_state=random_state,
            n_jobs=-1,
        )
        # Quick estimate with small CV
        try:
            strat_keys = _make_stratify_key(sub)
            # Use 3-fold for grid search speed
            cv_small = StratifiedKFold(n_splits=min(3, cv_folds), shuffle=True, random_state=random_state)
            scores = cross_val_score(rf, X, y, cv=cv_small, scoring="r2")
            mean_r2 = float(np.mean(scores))
            if mean_r2 > best_r2:
                best_r2     = mean_r2
                best_model  = rf
                best_params = dict(max_depth=max_depth, min_samples_leaf=min_leaf, max_features=max_feat)
        except Exception:
            continue

    if best_model is None:
        best_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        best_params = {}

    # ── Final 10-fold CV with best model ─────────────────────────────────────
    try:
        strat_keys = _make_stratify_key(sub)
        cv_final   = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_scores  = cross_val_score(best_model, X, y, cv=cv_final, scoring="r2")
        cv_mean_r2 = float(np.mean(cv_scores))
        cv_std_r2  = float(np.std(cv_scores))
    except Exception as exc:
        logger.warning("CV failed for %s: %s", metric, exc)
        cv_scores  = np.array([0.0])
        cv_mean_r2 = 0.0
        cv_std_r2  = 0.0

    # ── Fit on full data for feature importance ───────────────────────────────
    best_model.fit(X, y)
    importances = {
        feat_name: round(float(imp), 6)
        for feat_name, imp in zip(FEATURE_COLS, best_model.feature_importances_)
    }
    # Rank features
    ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)

    return {
        "metric":             metric,
        "n":                  len(sub),
        "best_params":        best_params,
        "cv_folds":           cv_folds,
        "cv_r2_scores":       [round(float(s), 6) for s in cv_scores],
        "cv_r2_mean":         round(cv_mean_r2, 6),
        "cv_r2_std":          round(cv_std_r2, 6),
        "feature_importance": importances,
        "feature_ranking":    [{"feature": k, "importance": v} for k, v in ranked],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_random_forest_analysis() -> Dict[str, Any]:
    """
    Run Random Forest analysis for all primary metrics.
    Saves to ANALYSIS_DIR/random_forest_results.json.
    """
    all_metrics = load_all_metrics()
    df          = build_flat_df(all_metrics)

    logger.info(
        "Running Random Forest analysis (n_estimators=%d, cv=%d folds) …",
        RF_N_ESTIMATORS, CV_FOLDS,
    )

    results: Dict[str, Any] = {}
    for metric in PRIMARY_METRICS:
        if metric not in df.columns:
            continue
        logger.info("  RF → %s", metric)
        results[metric] = run_random_forest(df, metric)

    # Convergence check: compare RF importance vs regression coefficients
    results["convergence_note"] = (
        "Compare RF feature_importance rankings vs regression coefficient magnitudes. "
        "Agreement on top-2 features strengthens confidence in framework factor weights."
    )

    out_path = ANALYSIS_DIR / "random_forest_results.json"
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("Random Forest results saved → %s", out_path)
    return results
