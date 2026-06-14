"""
rq3_experiment/framework/predictor.py
=====================================
The final Predictive Framework API (IMPROVED).
Loads trained models and predicts RAG performance metrics based on dataset factors.
Supports polynomial features and detects model version (with/without polynomials).
"""
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List

from ..config import DOMAIN_VOLATILITY_SCORE, RESULTS_DIR_RQ3

logger = logging.getLogger(__name__)


class RAGPerformancePredictor:
    """
    Estimates RAG effectiveness based on dataset-level factors:
    freshness, source diversity, and domain volatility.

    Supports both old models (6 features) and new models with polynomial features (9 features).
    """
    def __init__(self, models_path: Path = RESULTS_DIR_RQ3 / "rq3_models.json"):
        if not models_path.exists():
            raise FileNotFoundError(f"Trained models not found at {models_path}. Run rq3_experiment first.")

        self.models = json.loads(models_path.read_text(encoding="utf-8"))
        self.ols_models = self.models.get("ols", {})

        # Detect model version by checking coefficient names
        first_metric = list(self.ols_models.keys())[0] if self.ols_models else ""
        self.has_poly_features = False
        if first_metric:
            coefs = self.ols_models[first_metric].get("coefficients", {})
            self.has_poly_features = "freshness_score_sq" in coefs
            logger.info("Detected OLS model with %s features", "polynomial" if self.has_poly_features else "base 6")

    def _compute_freshness_score(self, age_days: float) -> float:
        """Standard half-life decay (180 days) used in the experiment."""
        return math.exp(-math.log(2) * (age_days / 180.0))

    def _build_features(self, domain: str, avg_age_days: float,
                        source_diversity_index: float) -> Dict[str, float]:
        """Build feature dict with or without polynomial terms based on model."""
        volatility = DOMAIN_VOLATILITY_SCORE.get(domain.lower(), 0.5)
        freshness = self._compute_freshness_score(avg_age_days)
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

        # Add polynomial features if the model expects them
        if self.has_poly_features:
            features["freshness_score_sq"] = freshness ** 2
            features["source_diversity_index_sq"] = sdi ** 2
            features["domain_volatility_sq"] = volatility ** 2

        return features

    def predict(
        self,
        domain: str,
        avg_age_days: float,
        source_diversity_index: float
    ) -> Dict[str, Any]:
        """
        Predicts primary metrics with error bounds (MAE).

        Args:
            domain: "technology", "healthcare", or "history"
            avg_age_days: Average age of retrieved documents in days
            source_diversity_index: Shannon entropy of source types (0.0 to ~0.4)

        Returns:
            Dictionary of predicted metrics and confidence intervals.
        """
        features = self._build_features(domain, avg_age_days, source_diversity_index)
        warnings: List[str] = []

        # Check for potential domain extrapolation
        if source_diversity_index > 0.45:
            warnings.append(
                f"Source diversity index {source_diversity_index:.2f} exceeds "
                f"training range (~0.45 max). Predictions may be unreliable."
            )

        predictions = {}

        for metric, model in self.ols_models.items():
            coefs = model["coefficients"]

            pred_value = 0.0
            matched_coefs = 0
            total_coefs = len(coefs)

            for feat_name, feat_val in features.items():
                if feat_name in coefs:
                    pred_value += coefs[feat_name]["coef"] * feat_val
                    matched_coefs += 1

            # Warn if many coefficients are missing (model mismatch)
            if matched_coefs < total_coefs - 1:  # -1 for intercept, OK if missing
                missing = set(coefs.keys()) - set(features.keys()) - {"intercept"}
                if missing:
                    warnings.append(
                        f"Metric '{metric}': {len(missing)} coefficient(s) "
                        f"not matched in input features: {missing}. "
                        f"Prediction may be incomplete."
                    )

            # Bound the metrics to their logical ranges
            if metric in ["bertscore_f1", "ndcg_at_5", "precision_at_5"]:
                pred_value = max(0.0, min(1.0, pred_value))
            elif metric == "hallucination_rate":
                pred_value = max(0.0, min(1.0, pred_value))
            elif metric == "human_eval_score":
                pred_value = max(1.0, min(5.0, pred_value))

            mae = model["cv_bounds"]["mae_mean"]
            lower_bound = pred_value - mae
            upper_bound = pred_value + mae

            # Apply metric-specific bounds
            if metric in ["bertscore_f1", "ndcg_at_5", "precision_at_5", "hallucination_rate"]:
                lower_bound = max(0.0, lower_bound)
                upper_bound = min(1.0, upper_bound)
            elif metric == "human_eval_score":
                lower_bound = max(1.0, lower_bound)
                upper_bound = min(5.0, upper_bound)

            predictions[metric] = {
                "expected": round(pred_value, 4),
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
                "mae_margin": round(mae, 4),
                "n_features_matched": f"{matched_coefs}/{total_coefs}",
            }

        result: Dict[str, Any] = {
            "inputs": {
                "domain": domain,
                "avg_age_days": avg_age_days,
                "freshness_score": round(features["freshness_score"], 4),
                "source_diversity_index": source_diversity_index,
            },
            "predictions": predictions,
        }

        if warnings:
            result["warnings"] = warnings

        return result