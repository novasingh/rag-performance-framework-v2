"""
rq1_experiment/run_analysis.py
================================
Statistical analysis and visualization runner for RQ1.

Runs AFTER run_experiment.py has produced raw outputs.

Steps:
  1. Compute all metrics for all conditions (evaluator)
  2. Two-way ANOVA + post-hoc Tukey HSD
  3. Multiple regression + decay curve fitting
  4. Random Forest feature importance + 10-fold CV
  5. Generate all plots

Usage:
    python -m rq1_experiment.run_analysis [--conditions C1 C2 ...] [--step all|metrics|anova|regression|rf|plots]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(__file__).parent / "results" / "analysis.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("rq1_experiment.run_analysis")


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ1 Statistical Analysis Runner")
    parser.add_argument(
        "--step",
        choices=["all", "metrics", "anova", "regression", "rf", "plots"],
        default="all",
        help="Which analysis step to run (default: all)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        metavar="C",
        help="Limit metric evaluation to specific conditions.",
    )
    parser.add_argument(
        "--bertscore-batch",
        type=int,
        default=8,
        help="BERTScore batch size (default 8).",
    )
    parser.add_argument(
        "--nli-threshold",
        type=float,
        default=0.5,
        help="NLI entailment threshold for hallucination detection (default 0.5).",
    )
    parser.add_argument(
        "--min-human-sample",
        type=int,
        default=480,
        help="Minimum paired human-eval sample size per condition (default 480).",
    )
    args = parser.parse_args()

    # Ensure results dirs exist
    from rq1_experiment.config import ANALYSIS_DIR, METRICS_DIR, PLOTS_DIR
    for d in [ANALYSIS_DIR, METRICS_DIR, PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Compute metrics ──────────────────────────────────────────────
    if args.step in ("all", "metrics"):
        logger.info("=== Step 1: Computing all evaluation metrics ===")
        from rq1_experiment.evaluation.evaluator import evaluate_all_conditions
        metrics = evaluate_all_conditions(
            conditions=args.conditions,
            bertscore_batch=args.bertscore_batch,
            nli_threshold=args.nli_threshold,
        )
        logger.info(
            "Metrics computed for %d conditions.", len(metrics)
        )
        _validate_human_eval(metrics, min_sample=args.min_human_sample)

    # ── Step 2: ANOVA ────────────────────────────────────────────────────────
    if args.step in ("all", "anova"):
        logger.info("=== Step 2: Two-way ANOVA + post-hoc analysis ===")
        try:
            from rq1_experiment.analysis.anova import run_anova_analysis
            anova_results = run_anova_analysis()
            _print_anova_summary(anova_results)
        except FileNotFoundError as exc:
            logger.error("ANOVA failed: %s — run --step metrics first", exc)

    # ── Step 3: Regression ───────────────────────────────────────────────────
    if args.step in ("all", "regression"):
        logger.info("=== Step 3: Multiple regression + decay curve fitting ===")
        try:
            from rq1_experiment.analysis.regression import run_regression_analysis
            reg_results = run_regression_analysis()
            _print_regression_summary(reg_results)
        except FileNotFoundError as exc:
            logger.error("Regression failed: %s — run --step metrics first", exc)

    # ── Step 4: Random Forest ────────────────────────────────────────────────
    if args.step in ("all", "rf"):
        logger.info("=== Step 4: Random Forest analysis (500 trees, 10-fold CV) ===")
        try:
            from rq1_experiment.analysis.random_forest import run_random_forest_analysis
            rf_results = run_random_forest_analysis()
            _print_rf_summary(rf_results)
        except FileNotFoundError as exc:
            logger.error("RF failed: %s — run --step metrics first", exc)

    # ── Step 5: Plots ────────────────────────────────────────────────────────
    if args.step in ("all", "plots"):
        logger.info("=== Step 5: Generating all plots ===")
        try:
            from rq1_experiment.analysis.visualizer import generate_all_plots
            generate_all_plots()
        except FileNotFoundError as exc:
            logger.error("Plots failed: %s — run metrics + analysis steps first", exc)

    logger.info("Analysis complete.")


def _print_anova_summary(results: dict) -> None:
    anova = results.get("anova", {})
    logger.info("\n--- ANOVA SUMMARY ---")
    for metric, data in anova.items():
        factors = data.get("factors", {})
        logger.info("  %s:", metric)
        for factor, info in factors.items():
            sig = "[SIG]" if info.get("significant") else "not sig."
            eta2 = info.get("eta_squared", "-")
            label = info.get("effect_size_label", "")
            logger.info(
                "    %-15s F=%-8s p=%-8s eta2=%-6s (%s) %s",
                factor,
                f"{info.get('F', '—'):.4f}" if isinstance(info.get('F'), float) else "—",
                f"{info.get('p_value', '—'):.4f}" if isinstance(info.get('p_value'), float) else "—",
                f"{eta2:.4f}" if isinstance(eta2, float) else "—",
                label,
                sig,
            )


def _print_regression_summary(results: dict) -> None:
    ols = results.get("ols", {})
    logger.info("\n--- OLS REGRESSION SUMMARY ---")
    for metric, data in ols.items():
        r2     = data.get("r_squared", "—")
        adj_r2 = data.get("adj_r_squared", "—")
        logger.info("  %s: R²=%.4f, adj-R²=%.4f", metric, r2 or 0, adj_r2 or 0)
        for name, info in data.get("coefficients", {}).items():
            sig = "✓" if info.get("significant") else " "
            logger.info("    %s %-25s β=%.4f  p=%.4f", sig, name, info.get("coef", 0), info.get("p_value", 1))


def _print_rf_summary(results: dict) -> None:
    logger.info("\n--- RANDOM FOREST SUMMARY ---")
    for metric, data in results.items():
        if not isinstance(data, dict) or "cv_r2_mean" not in data:
            continue
        logger.info(
            "  %s: CV-R²=%.4f ± %.4f",
            metric,
            data.get("cv_r2_mean", 0),
            data.get("cv_r2_std", 0),
        )
        for rank in data.get("feature_ranking", [])[:3]:
            logger.info("    top: %-28s importance=%.4f", rank["feature"], rank["importance"])


def _validate_human_eval(metrics: dict, min_sample: int) -> None:
    from rq1_experiment.config import COHEN_KAPPA_THRESHOLD

    kappa_fail = []
    sample_fail = []

    for cid, m in metrics.items():
        kappa_ok = bool(m.get("human_eval_kappa_threshold_met"))
        counts = m.get("human_eval_counts", {})
        min_count = counts.get("min_count", 0) if isinstance(counts, dict) else 0

        if not kappa_ok:
            kappa_fail.append(cid)
        if min_count < min_sample:
            sample_fail.append((cid, min_count))

    if kappa_fail or sample_fail:
        if sample_fail:
            details = ", ".join([f"{cid} (n={n})" for cid, n in sample_fail])
            logger.error("Human eval sample too small (<%d): %s", min_sample, details)
        if kappa_fail:
            logger.error("Human eval kappa below threshold (%.2f): %s", COHEN_KAPPA_THRESHOLD, ", ".join(kappa_fail))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
