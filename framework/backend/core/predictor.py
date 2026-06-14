"""
core/predictor.py
=================
Self-contained version of the RQ3 predictive framework.

Loads the trained model (``rq3_models.json``) and predicts RAG performance
metrics from dataset-level factors: freshness, source diversity, and domain
volatility. Identical model maths to ``rq3_experiment/framework/predictor.py``
but with no dependency on the experiment packages.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List

from .config import (
    DOMAIN_VOLATILITY_SCORE,
    FRESHNESS_HALF_LIFE_DAYS,
    load_models,
)

logger = logging.getLogger(__name__)

# Maximum source-diversity index seen during training; used for extrapolation warnings.
_SDI_TRAINING_MAX = 0.45


class RAGPerformancePredictor:
    """Estimates RAG effectiveness from dataset-level factors using the RQ3 model."""

    def __init__(self, models: Dict[str, Any] | None = None):
        self.models = models if models is not None else load_models()
        self.ols_models = self.models.get("ols", {})
        if not self.ols_models:
            raise ValueError("No OLS models found in rq3_models.json.")

        first_metric = next(iter(self.ols_models))
        coefs = self.ols_models[first_metric].get("coefficients", {})
        self.has_poly_features = "freshness_score_sq" in coefs
        logger.info(
            "Predictor loaded (%s features)",
            "polynomial" if self.has_poly_features else "base",
        )

    # ── feature engineering ────────────────────────────────────────────────
    def compute_freshness_score(self, age_days: float) -> float:
        """Half-life decay: 1.0 at age 0, 0.5 at the half-life."""
        return math.exp(-math.log(2) * (age_days / FRESHNESS_HALF_LIFE_DAYS))

    def _build_features(self, domain: str, avg_age_days: float,
                        source_diversity_index: float) -> Dict[str, float]:
        volatility = DOMAIN_VOLATILITY_SCORE.get(domain.lower(), 0.5)
        freshness = self.compute_freshness_score(avg_age_days)
        sdi = source_diversity_index

        features = {
            "intercept": 1.0,
            "freshness_score": freshness,
            "source_diversity_index": sdi,
            "domain_volatility": volatility,
            "fresh_x_diversity": freshness * sdi,
            "fresh_x_volatility": freshness * volatility,
            "source_x_volatility": sdi * volatility,
        }
        if self.has_poly_features:
            features["freshness_score_sq"] = freshness ** 2
            features["source_diversity_index_sq"] = sdi ** 2
            features["domain_volatility_sq"] = volatility ** 2
        return features

    @staticmethod
    def _clamp(metric: str, value: float) -> float:
        if metric == "human_eval_score":
            return max(1.0, min(5.0, value))
        return max(0.0, min(1.0, value))

    # ── prediction ─────────────────────────────────────────────────────────
    def predict(self, domain: str, avg_age_days: float,
                source_diversity_index: float) -> Dict[str, Any]:
        features = self._build_features(domain, avg_age_days, source_diversity_index)
        warnings: List[str] = []

        if source_diversity_index > _SDI_TRAINING_MAX:
            warnings.append(
                f"Source diversity index {source_diversity_index:.2f} exceeds the "
                f"training range (~{_SDI_TRAINING_MAX}). Predictions may be unreliable."
            )

        predictions: Dict[str, Any] = {}
        for metric, model in self.ols_models.items():
            coefs = model["coefficients"]
            pred = sum(c["coef"] * features[name]
                       for name, c in coefs.items() if name in features)
            pred = self._clamp(metric, pred)

            mae = model.get("cv_bounds", {}).get("mae_mean", 0.0)
            lower = self._clamp(metric, pred - mae)
            upper = self._clamp(metric, pred + mae)

            predictions[metric] = {
                "expected": round(pred, 4),
                "lower_bound": round(lower, 4),
                "upper_bound": round(upper, 4),
                "mae_margin": round(mae, 4),
            }

        result: Dict[str, Any] = {
            "inputs": {
                "domain": domain,
                "avg_age_days": avg_age_days,
                "freshness_score": round(features["freshness_score"], 4),
                "source_diversity_index": source_diversity_index,
                "domain_volatility": DOMAIN_VOLATILITY_SCORE.get(domain.lower(), 0.5),
            },
            "predictions": predictions,
        }
        if warnings:
            result["warnings"] = warnings
        return result
