"""
rq3_experiment/models/decay_curves.py
=====================================
Fits and extracts decay curve parameters for freshness decay across
different domains. Supports exponential, polynomial, and logistic models.

All three models implement the same signature: y = f(x, *params)
- exponential: a * exp(-b * x) + c
- polynomial:  a * x^2 + b * x + c
- logistic: L / (1 + exp(-k * (x - x0)))   [uses -k for stability]
"""
import logging
import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


def _exponential_decay(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """y = a * exp(-b * x) + c"""
    return a * np.exp(-b * x) + c


def _polynomial_decay(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """y = a * x^2 + b * x + c"""
    return a * x**2 + b * x + c


def _logistic_decay(x: np.ndarray, L: float, k: float, x0: float) -> np.ndarray:
    """
    Sigmoid/logistic decay: y = L / (1 + exp(-k * (x - x0)))
    
    Using -k in the exponent prevents overflow: when x >> x0 and k > 0,
    exp(-k * (x - x0)) -> 0 instead of inf.
    Better for bounded metrics (e.g., precision [0,1]) that have a 
    plateau then drop-off.
    """
    exponent = -k * (x - x0)
    # Clip exponent to avoid overflow: exp(710) ~= inf in float64
    exponent = np.clip(exponent, -700.0, 700.0)
    return L / (1.0 + np.exp(exponent))


def fit_freshness_decay(df: pd.DataFrame, metric: str) -> Dict[str, Any]:
    """
    Fits decay curves per domain and returns the optimal parameters.
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
            ("logistic",    _logistic_decay,    [1.0, 1.0, 0.5]),
        ]:
            try:
                # Suppress the covariance warning (not fatal — the fit is still valid)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=Warning)
                    popt, _ = curve_fit(func, x, y, p0=p0, maxfev=5000)

                y_pred = func(x, *popt)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

                domain_result[name] = {
                    "params": [round(float(p), 6) for p in popt],
                    "r_squared": round(r2, 6),
                }
            except Exception as exc:
                domain_result[name] = {"error": str(exc)}

        results["domains"][domain] = domain_result

    return results


def fit_all_decay_curves(df: pd.DataFrame, primary_metrics: list[str]) -> Dict[str, Any]:
    results = {}
    for metric in primary_metrics:
        results[metric] = fit_freshness_decay(df, metric)
    return results