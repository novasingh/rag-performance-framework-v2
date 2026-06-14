"""
rq3_experiment/run_rq3.py
=========================
Main entry point for the RQ3 Predictive Framework module (IMPROVED).

Orchestrates data preparation, cross-validated OLS, RF, and XGBoost modeling,
decay curve fitting (exponential, polynomial, logistic), and saving the final models.

Improvements over original:
- Polynomial features (freshness^2, SDI^2, volatility^2) for non-linearity
- XGBoost model as a stronger non-linear alternative to RF
- Logistic decay curves for better bounded-metric fitting
- Enhanced predictor with 95% CI bounds and domain-specific warnings
"""
import json
import logging
import sys

from rq3_experiment.config import PRIMARY_METRICS, RESULTS_DIR_RQ3
from rq3_experiment.data import load_query_level_df
from rq3_experiment.models.decay_curves import fit_all_decay_curves
from rq3_experiment.models.ols_regression import train_all_ols
from rq3_experiment.models.random_forest import train_all_rf
from rq3_experiment.models.xgboost_model import train_all_xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR_RQ3 / "rq3_training.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("rq3_experiment.run_rq3")


def main() -> None:
    logger.info("=== Phase 3: Training Predictive Framework (IMPROVED) ===")

    # 1. Load Data (includes polynomial features now)
    try:
        df = load_query_level_df()
    except FileNotFoundError as exc:
        logger.error(exc)
        sys.exit(1)

    # 2. Train Models
    logger.info("--- Step 1: OLS Regression (Cross-Validated) ---")
    ols_results = train_all_ols(df, PRIMARY_METRICS)

    logger.info("--- Step 2: Random Forest (Cross-Validated) ---")
    rf_results = train_all_rf(df, PRIMARY_METRICS)

    logger.info("--- Step 3: XGBoost (Cross-Validated) ---")
    xgb_results = train_all_xgb(df, PRIMARY_METRICS)

    logger.info("--- Step 4: Decay Curve Fitting ---")
    decay_results = fit_all_decay_curves(df, PRIMARY_METRICS)

    # 3. Save Results
    final_models = {
        "n_samples": len(df),
        "n_features": len(df.columns),
        "ols": ols_results,
        "random_forest": rf_results,
        "xgboost": xgb_results,
        "decay_curves": decay_results,
    }

    out_path = RESULTS_DIR_RQ3 / "rq3_models.json"
    out_path.write_text(json.dumps(final_models, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("RQ3 models saved -> %s", out_path)

    # 4. Print Summary Comparison
    logger.info("--- Step 5: Model Comparison Summary ---")
    _print_comparison(ols_results, rf_results, xgb_results)

    # 5. Verification Test
    logger.info("--- Step 6: Testing Framework API ---")
    try:
        from rq3_experiment.framework.predictor import RAGPerformancePredictor
        predictor = RAGPerformancePredictor(out_path)

        test_configs = [
            ("technology", 14.0, 0.35, "Recent Tech, High Diversity"),
            ("healthcare", 60.0, 0.25, "Mid Healthcare, Mid Diversity"),
            ("history", 365.0, 0.15, "Old History, Low Diversity"),
        ]

        for domain, age, sdi, label in test_configs:
            result = predictor.predict(domain=domain, avg_age_days=age, source_diversity_index=sdi)
            logger.info("Test: %s", label)
            for metric, vals in result["predictions"].items():
                logger.info(
                    "  %s: %.4f [%.4f, %.4f] (MAE +/-%.4f)",
                    metric.ljust(20), vals["expected"],
                    vals["lower_bound"], vals["upper_bound"],
                    vals["mae_margin"],
                )
            if "warnings" in result and result["warnings"]:
                for w in result["warnings"]:
                    logger.warning("  Warning: %s", w)

    except Exception as exc:
        logger.error("Framework API test failed: %s", exc)

    logger.info("Phase 3 pipeline complete.")


def _print_comparison(ols: dict, rf: dict, xgb: dict) -> None:
    """Print a comparison table of OLS vs RF vs XGBoost performance."""
    metrics = list(ols.keys())
    logger.info("Model Comparison (MAE, lower is better):")
    logger.info("  %-22s %-15s %-15s %s", "Metric", "OLS (MAE)", "RF (MAE)", "XGB (MAE)")
    logger.info("  " + "-" * 70)
    for m in metrics:
        ols_mae = ols.get(m, {}).get("cv_bounds", {}).get("mae_mean", 0)
        rf_mae = rf.get(m, {}).get("cv_bounds", {}).get("mae_mean", 0)
        xgb_mae = xgb.get(m, {}).get("cv_bounds", {}).get("mae_mean", 0)
        best = min(ols_mae, rf_mae, xgb_mae)
        markers = []
        for v in [ols_mae, rf_mae, xgb_mae]:
            markers.append(" <- best" if v == best else "")
        logger.info(
            "  %-22s %-8.4f%s %-8.4f%s %-8.4f%s",
            m, ols_mae, markers[0], rf_mae, markers[1], xgb_mae, markers[2],
        )


if __name__ == "__main__":
    main()