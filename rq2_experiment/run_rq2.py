"""
rq2_experiment/run_rq2.py
==========================
Main entry point for the RQ2 analysis pipeline.

RQ2: "What is the individual contribution of source type configurations to
      RAG system performance across multiple evaluation metrics, and does this
      contribution vary by domain volatility?"

This script REUSES all raw data from RQ1. No new RAG pipeline runs needed.

Usage:
    python -m rq2_experiment.run_rq2
    python -m rq2_experiment.run_rq2 --step anova
    python -m rq2_experiment.run_rq2 --step contribution
    python -m rq2_experiment.run_rq2 --step plots
    python -m rq2_experiment.run_rq2 --step all   (default)
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
            Path(__file__).parent.parent / "rq1_experiment" / "results" / "rq2_analysis.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("rq2_experiment.run_rq2")


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ2 Statistical Analysis Runner")
    parser.add_argument(
        "--step",
        choices=["all", "anova", "contribution", "plots"],
        default="all",
        help="Which analysis step to run (default: all)",
    )
    args = parser.parse_args()

    # Ensure output dirs exist
    from rq2_experiment.config import ANALYSIS_DIR_RQ2, PLOTS_DIR_RQ2
    ANALYSIS_DIR_RQ2.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR_RQ2.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Source-type ANOVA ────────────────────────────────────────────
    if args.step in ("all", "anova"):
        logger.info("=== Step 1: Source-Type ANOVA (one-way, two-way, Tukey HSD) ===")
        try:
            from rq2_experiment.analysis.source_type_anova import run_source_type_anova
            anova_results = run_source_type_anova()
            _print_anova_summary(anova_results)
        except FileNotFoundError as exc:
            logger.error("ANOVA failed: %s — ensure RQ1 metrics exist first.", exc)
            sys.exit(1)

    # ── Step 2: Marginal contribution & SDI regression ───────────────────────
    if args.step in ("all", "contribution"):
        logger.info("=== Step 2: Source Contribution Analysis (marginal gains + SDI regression) ===")
        try:
            from rq2_experiment.analysis.source_contribution import run_source_contribution_analysis
            contrib_results = run_source_contribution_analysis()
            _print_contribution_summary(contrib_results)
        except FileNotFoundError as exc:
            logger.error("Contribution analysis failed: %s", exc)
            sys.exit(1)

    # ── Step 3: Plots ────────────────────────────────────────────────────────
    if args.step in ("all", "plots"):
        logger.info("=== Step 3: Generating all RQ2 plots ===")
        try:
            from rq2_experiment.analysis.visualizer_rq2 import generate_all_rq2_plots
            generate_all_rq2_plots()
        except FileNotFoundError as exc:
            logger.error("Plot generation failed: %s — run --step anova and --step contribution first.", exc)
            sys.exit(1)

    logger.info("RQ2 analysis complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Summary printers
# ─────────────────────────────────────────────────────────────────────────────

def _print_anova_summary(results: dict) -> None:
    logger.info("\n" + "=" * 65)
    logger.info("  RQ2 — SOURCE TYPE ANOVA SUMMARY")
    logger.info("=" * 65)

    one_way = results.get("one_way_anova", {})
    two_way = results.get("two_way_anova", {})

    for metric, data in one_way.items():
        sig  = "[SIG]" if data.get("significant") else "not sig."
        eta2 = data.get("eta_squared", 0)
        label = data.get("effect_size_label", "")
        f_val = data.get("F", 0)
        p_val = data.get("p_value", 1)
        logger.info(
            "  [ONE-WAY] %-22s F=%8.4f  p=%8.5f  eta2=%6.4f (%s)  %s",
            metric, f_val, p_val, eta2, label, sig
        )

        # Two-way interaction check
        tw = two_way.get(metric, {}).get("factors", {})
        interaction = tw.get("interaction", {})
        if interaction:
            int_sig = "[INTERACTION SIG]" if interaction.get("significant") else "interaction n.s."
            logger.info(
                "  [TWO-WAY interaction] %-14s p=%8.5f  eta2=%6.4f  %s",
                metric, interaction.get("p_value", 1), interaction.get("eta_squared", 0), int_sig
            )

    logger.info("")
    logger.info("  Group means by source level:")
    desc = results.get("descriptive_by_source_level", {})
    for label, vals in desc.items():
        p5  = vals.get("precision_at_5", 0)
        ndcg= vals.get("ndcg_at_5", 0)
        hall= vals.get("hallucination_rate", 0)
        bs  = vals.get("bertscore_f1", 0)
        logger.info("    %-30s  P@5=%.4f  nDCG=%.4f  Hall=%.4f  BERTs=%.4f",
                    label, p5, ndcg, hall, bs)


def _print_contribution_summary(results: dict) -> None:
    logger.info("\n" + "=" * 65)
    logger.info("  RQ2 — SOURCE CONTRIBUTION SUMMARY")
    logger.info("=" * 65)

    gains = results.get("marginal_gains", {})
    for metric in ["ndcg_at_5", "precision_at_5", "hallucination_rate", "bertscore_f1"]:
        if metric not in gains:
            continue
        logger.info("\n  %s:", metric.upper())
        for domain, d in gains[metric].items():
            sa = d.get("step_a_gain_adding_news") or 0
            sb = d.get("step_b_gain_adding_tech") or 0
            logger.info(
                "    %-12s  +News=%+.4f  +Tech=%+.4f  dominant=%s",
                domain, sa, sb, d.get("dominant_step", "-")
            )

    logger.info("")
    sdi = results.get("sdi_regression", {})
    logger.info("  SDI Regression (moderated by volatility):")
    for metric, d in sdi.items():
        mod = d.get("moderated_regression", {})
        logger.info(
            "    %-22s  interaction_coef=%+.4f  p=%.5f  %s",
            metric,
            mod.get("interaction_coef", 0),
            mod.get("interaction_pval", 1),
            "[SIG]" if mod.get("interaction_sig") else " ",
        )


if __name__ == "__main__":
    main()
