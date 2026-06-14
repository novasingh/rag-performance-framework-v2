"""
rq3_experiment/models/ols_regression.py
=======================================
Multiple OLS regression modeling for the Predictive Framework.
Fits linear equations mapping freshness and source factors to expected
RAG performance, evaluated with 10-fold cross-validation.

NOTE: Uses FEATURE_COLS_BASE (6 features: 3 base + 3 interactions) to avoid
multicollinearity from polynomial terms. Tree-based models (RF, XGBoost) 
use the full feature set with polynomial terms for non-linear capture.
"""
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

from ..config import CV_FOLDS, FEATURE_COLS_BASE, RANDOM_SEED

logger = logging.getLogger(__name__)


def run_ols_with_cv(df: pd.DataFrame, metric: str) -> Dict[str, Any]:
    """
    Fits full OLS model to get equations, and runs 10-fold CV to get 
    error bounds (MAE/RMSE).
    Uses FEATURE_COLS_BASE (no polynomial terms) to avoid multicollinearity.
    """
    logger.info("  Training OLS for %s...", metric)
    sub = df[FEATURE_COLS_BASE + [metric]].dropna()
    X = sm.add_constant(sub[FEATURE_COLS_BASE])
    y = sub[metric]

    # 1. Full Fit for Equations
    model = sm.OLS(y, X).fit()
    
    coef_names = ["intercept"] + FEATURE_COLS_BASE
    coefficients = {}
    for name, coef, p in zip(coef_names, model.params, model.pvalues):
        coefficients[name] = {
            "coef": round(float(coef), 6),
            "p_value": round(float(p), 6),
            "significant": bool(float(p) < 0.05)
        }

    # 2. 10-Fold Cross-Validation for Error Bounds
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_mae = []
    cv_rmse = []

    X_np = X.values
    y_np = y.values

    for train_idx, test_idx in kf.split(X_np):
        X_train, X_test = X_np[train_idx], X_np[test_idx]
        y_train, y_test = y_np[train_idx], y_np[test_idx]

        # Use exact OLS solution for speed in CV
        beta, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
        y_pred = X_test @ beta
        
        cv_mae.append(mean_absolute_error(y_test, y_pred))
        cv_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))

    return {
        "metric": metric,
        "n": len(sub),
        "r_squared": round(float(model.rsquared), 6),
        "adj_r_squared": round(float(model.rsquared_adj), 6),
        "f_p_value": round(float(model.f_pvalue), 6),
        "coefficients": coefficients,
        "cv_bounds": {
            "mae_mean": round(float(np.mean(cv_mae)), 6),
            "mae_std": round(float(np.std(cv_mae)), 6),
            "rmse_mean": round(float(np.mean(cv_rmse)), 6),
            "rmse_std": round(float(np.std(cv_rmse)), 6),
        }
    }


def train_all_ols(df: pd.DataFrame, primary_metrics: list[str]) -> Dict[str, Any]:
    results = {}
    for metric in primary_metrics:
        results[metric] = run_ols_with_cv(df, metric)
    return results