"""
rq4_experiment/run_rq4.py
=========================
Main entry point for the RQ4 Framework Validation module.
Executes LOCO cross-validation, identifies boundary conditions,
and generates validation plots.
"""
import json
import logging
import sys

from rq4_experiment.config import PRIMARY_METRICS, RESULTS_DIR_RQ4
from rq4_experiment.validation.loco_validator import run_all_metrics_loco
from rq4_experiment.validation.visualizer_rq4 import generate_loco_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR_RQ4 / "rq4_validation.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("rq4_experiment.run_rq4")


def main() -> None:
    logger.info("=== Phase 4: Framework Validation & Benchmarking (RQ4) ===")
    
    # 1. Run LOCO Validation
    logger.info("--- Step 1: Leave-One-Condition-Out (LOCO) Cross-Validation ---")
    try:
        validation_results = run_all_metrics_loco(PRIMARY_METRICS)
    except Exception as exc:
        logger.error("Validation failed: %s", exc)
        sys.exit(1)
        
    # 2. Save JSON
    out_json = RESULTS_DIR_RQ4 / "rq4_validation.json"
    out_json.write_text(json.dumps(validation_results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved numerical validation results -> %s", out_json)
    
    # Print Boundary Condition Summary
    logger.info("\n--- Step 2: Boundary Condition Analysis ---")
    bc_count = validation_results["boundary_condition_count_by_domain"]
    total_boundaries = sum(bc_count.values())
    
    logger.info("Total Boundary Conditions Identified: %d", total_boundaries)
    for domain, count in bc_count.items():
        logger.info("  %s: %d", domain.capitalize(), count)
        
    if total_boundaries > 0:
        logger.warning("The framework struggles with certain configurations (check JSON for details).")
    else:
        logger.info("The framework generalized well to all 12 experimental conditions.")

    # 3. Generate Plots
    logger.info("\n--- Step 3: Generating Validation Plots ---")
    generate_loco_plots(validation_results)
    
    logger.info("Phase 4 validation complete.")


if __name__ == "__main__":
    main()
