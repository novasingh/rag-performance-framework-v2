"""
rq4_experiment/validation/loco_validator.py
===========================================
Implements Leave-One-Condition-Out (LOCO) Cross-Validation.
Establishes 95% Confidence Intervals for predictions on held-out data
and automatically flags boundary conditions.
"""
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy import stats

from rq3_experiment.data import load_query_level_df
from ..config import BOUNDARY_MAE_THRESHOLD, FEATURE_COLS

logger = logging.getLogger(__name__)


def _calculate_95_ci(data: np.ndarray) -> tuple[float, float]:
    """Calculates 95% CI bounds for an array of errors/predictions."""
    if len(data) < 2:
        return 0.0, 0.0
    mean = np.mean(data)
    sem = stats.sem(data)
    margin = sem * stats.t.ppf((1 + 0.95) / 2., len(data)-1)
    return float(mean - margin), float(mean + margin)


def run_loco_validation(metric: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    For a given metric, loop through all 12 conditions.
    Train on 11, test on the 1 held out.
    Compare predicted mean vs actual mean.
    """
    logger.info("  Running LOCO Validation for %s...", metric)
    
    conditions = sorted(df["condition_id"].unique())
    results_by_condition = {}
    
    for held_out_cid in conditions:
        train_df = df[df["condition_id"] != held_out_cid]
        test_df  = df[df["condition_id"] == held_out_cid]
        
        # Train Random Forest
        X_train = train_df[FEATURE_COLS].values
        y_train = train_df[metric].values
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Test (predict)
        X_test = test_df[FEATURE_COLS].values
        y_pred = model.predict(X_test)
        y_actual = test_df[metric].values
        
        # Compare means
        actual_mean = float(y_actual.mean())
        pred_mean = float(y_pred.mean())
        abs_error = abs(actual_mean - pred_mean)
        
        # Calculate 95% CI of the errors in this fold
        errors = y_pred - y_actual
        ci_lower, ci_upper = _calculate_95_ci(errors)
        
        domain = test_df["domain"].iloc[0]
        
        is_boundary = abs_error > BOUNDARY_MAE_THRESHOLD
        
        results_by_condition[held_out_cid] = {
            "domain": domain,
            "actual_mean": round(actual_mean, 4),
            "predicted_mean": round(pred_mean, 4),
            "absolute_error": round(abs_error, 4),
            "ci_95_lower": round(pred_mean + ci_lower, 4),
            "ci_95_upper": round(pred_mean + ci_upper, 4),
            "is_boundary_condition": is_boundary,
        }
        
    return results_by_condition


def run_all_metrics_loco(primary_metrics: List[str]) -> Dict[str, Any]:
    df = load_query_level_df()
    
    all_results = {}
    boundary_summary = {
        "technology": 0,
        "healthcare": 0,
        "history": 0,
    }
    
    for metric in primary_metrics:
        res = run_loco_validation(metric, df)
        all_results[metric] = res
        
        for cid, data in res.items():
            if data["is_boundary_condition"]:
                boundary_summary[data["domain"]] += 1
                
    return {
        "n_samples": len(df),
        "metrics": all_results,
        "boundary_condition_count_by_domain": boundary_summary
    }
