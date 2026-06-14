"""
rq3_experiment/models/xgboost_model.py
=======================================
XGBoost regression layer for the Predictive Framework.
Provides a stronger non-linear alternative to Random Forest with
built-in regularization, early stopping, and feature importance.

Requires: pip install xgboost
"""
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedKFold

from ..config import (
    CV_FOLDS,
    FEATURE_COLS,
    RANDOM_SEED,
    XGB_N_ESTIMATORS,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_EARLY_STOPPING_ROUNDS,
)

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not installed. Install with: pip install xgboost")


def run_xgb_with_cv(df: pd.DataFrame, metric: str) -> Dict[str, Any]:
    """
    Fits XGBoost with cross-validation, extracts feature importance,
    and returns performance bounds. Falls back gracefully if xgboost is not installed.
    """
    if not XGB_AVAILABLE:
        return {
            "metric": metric,
            "n": 0,
            "error": "XGBoost not installed. Run: pip install xgboost",
            "cv_bounds": {
                "mae_mean": 0.0,
                "mae_std": 0.0,
                "rmse_mean": 0.0,
                "rmse_std": 0.0,
            }
        }

    logger.info("  Training XGBoost for %s...", metric)

    sub = df[FEATURE_COLS + [metric, "condition_id"]].dropna()
    X = sub[FEATURE_COLS].values
    y = sub[metric].values
    conditions = sub["condition_id"].values

    # 1. Train with best parameters
    xgb_model = XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS,
        learning_rate=XGB_LEARNING_RATE,
        max_depth=XGB_MAX_DEPTH,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )

    # 2. 10-Fold Stratified Cross-Validation for Error Bounds
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_mae = []
    cv_rmse = []
    all_importances = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, conditions)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train with early stopping on a hold-out validation split
        split_idx = int(len(X_train) * 0.8)
        xgb_model.fit(
            X_train[:split_idx], y_train[:split_idx],
            eval_set=[(X_train[split_idx:], y_train[split_idx:])],
            verbose=False,
        )
        y_pred = xgb_model.predict(X_test)

        cv_mae.append(mean_absolute_error(y_test, y_pred))
        cv_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        all_importances.append(xgb_model.feature_importances_)

    # Aggregate feature importances across folds
    avg_importances = np.mean(all_importances, axis=0)
    feat_importances = {
        name: round(float(imp), 6)
        for name, imp in zip(FEATURE_COLS, avg_importances)
    }

    return {
        "metric": metric,
        "n": len(sub),
        "best_params": {
            "n_estimators": XGB_N_ESTIMATORS,
            "learning_rate": XGB_LEARNING_RATE,
            "max_depth": XGB_MAX_DEPTH,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        "feature_importances": feat_importances,
        "cv_bounds": {
            "mae_mean": round(float(np.mean(cv_mae)), 6),
            "mae_std": round(float(np.std(cv_mae)), 6),
            "rmse_mean": round(float(np.mean(cv_rmse)), 6),
            "rmse_std": round(float(np.std(cv_rmse)), 6),
        }
    }


def train_all_xgb(df: pd.DataFrame, primary_metrics: list[str]) -> Dict[str, Any]:
    results = {}
    for metric in primary_metrics:
        results[metric] = run_xgb_with_cv(df, metric)
    return results