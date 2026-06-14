"""
rq3_experiment/models/random_forest.py
======================================
Random Forest validation layer for the Predictive Framework.
Provides non-parametric validation of the OLS results and generates
robust feature importance rankings.

Uses metric-specific feature sets:
- human_eval_score: FEATURE_COLS_HUMAN_EVAL (includes per-query objective metrics)
- All other metrics: FEATURE_COLS (base + retrieval + interactions)
"""
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from ..config import (
    CV_FOLDS,
    FEATURE_COLS,
    FEATURE_COLS_HUMAN_EVAL,
    RANDOM_SEED,
    RF_N_ESTIMATORS,
)

logger = logging.getLogger(__name__)


def _get_features(metric: str) -> list[str]:
    """Return the appropriate feature set for the given metric."""
    if metric == "human_eval_score":
        return FEATURE_COLS_HUMAN_EVAL
    return FEATURE_COLS


def run_rf_with_cv(df: pd.DataFrame, metric: str) -> Dict[str, Any]:
    """
    Fits Random Forest with GridSearchCV, extracts feature importance,
    and runs stratified 10-fold CV for performance bounds.
    """
    logger.info("  Training Random Forest for %s...", metric)
    features = _get_features(metric)
    sub = df[features + [metric, "condition_id"]].dropna()
    X = sub[features].values
    y = sub[metric].values
    conditions = sub["condition_id"].values

    # 1. Hyperparameter Tuning (Grid Search)
    rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_SEED, n_jobs=-1)
    param_grid = {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv_stratified.split(X, conditions),
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    grid_search.fit(X, y)
    best_rf = grid_search.best_estimator_

    # 2. Feature Importance
    importances = best_rf.feature_importances_
    feat_importances = {
        name: round(float(imp), 6)
        for name, imp in zip(features, importances)
    }

    # 3. 10-Fold Stratified Cross-Validation for Error Bounds
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_mae = []
    cv_rmse = []

    for train_idx, test_idx in skf.split(X, conditions):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        best_rf.fit(X_train, y_train)
        y_pred = best_rf.predict(X_test)

        cv_mae.append(mean_absolute_error(y_test, y_pred))
        cv_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))

    return {
        "metric": metric,
        "n": len(sub),
        "n_features": len(features),
        "features": features,
        "best_params": grid_search.best_params_,
        "feature_importances": feat_importances,
        "cv_bounds": {
            "mae_mean": round(float(np.mean(cv_mae)), 6),
            "mae_std": round(float(np.std(cv_mae)), 6),
            "rmse_mean": round(float(np.mean(cv_rmse)), 6),
            "rmse_std": round(float(np.std(cv_rmse)), 6),
        }
    }


def train_all_rf(df: pd.DataFrame, primary_metrics: list[str]) -> Dict[str, Any]:
    results = {}
    for metric in primary_metrics:
        results[metric] = run_rf_with_cv(df, metric)
    return results