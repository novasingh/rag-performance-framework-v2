"""
rq4_experiment/validation/visualizer_rq4.py
===========================================
Generates visualizations for RQ4 Framework Validation.
Creates Predicted vs Actual scatter plots with 95% CI error bars,
highlighting boundary conditions.
"""
import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from ..config import PLOTS_DIR_RQ4

logger = logging.getLogger(__name__)

# Colors by domain
DOMAIN_COLORS = {
    "technology": "#06b6d4", # cyan
    "healthcare": "#a855f7", # purple
    "history":    "#10b981", # emerald
}

def generate_loco_plots(loco_results: Dict[str, Any]) -> None:
    """
    Plots Predicted (y) vs Actual (x) for each condition, 
    with error bars (95% CI).
    """
    metrics = loco_results["metrics"]
    
    for metric, conditions_data in metrics.items():
        plt.figure(figsize=(8, 8))
        
        # Plot y=x reference line with dynamic bounds
        all_actual = [d["actual_mean"] for d in conditions_data.values()]
        all_pred = [d["predicted_mean"] for d in conditions_data.values()]
        max_val = max(all_actual + all_pred) if (all_actual + all_pred) else 1.0
        axis_max = max_val * 1.1 if max_val > 1.0 else 1.0
        plt.plot([0, axis_max], [0, axis_max], 'k--', alpha=0.3, label="Perfect Prediction")
        
        for cid, data in conditions_data.items():
            actual = data["actual_mean"]
            predicted = data["predicted_mean"]
            lower = data["ci_95_lower"]
            upper = data["ci_95_upper"]
            domain = data["domain"]
            is_boundary = data["is_boundary_condition"]
            
            yerr_lower = abs(predicted - lower)
            yerr_upper = abs(upper - predicted)
            
            # Use red X for boundary conditions
            marker = 'X' if is_boundary else 'o'
            edgecolor = 'red' if is_boundary else 'white'
            s = 150 if is_boundary else 100
            
            plt.errorbar(
                actual, predicted, 
                yerr=[[yerr_lower], [yerr_upper]],
                fmt='none', ecolor=DOMAIN_COLORS.get(domain, "gray"), 
                alpha=0.5, capsize=3
            )
            
            plt.scatter(
                actual, predicted, 
                color=DOMAIN_COLORS.get(domain, "gray"), 
                edgecolors=edgecolor,
                s=s, marker=marker, zorder=5,
                label=domain.capitalize() if cid in ["C1", "C7", "C10"] else ""
            )
            
            # Annotate C_ID
            plt.annotate(
                cid, (actual, predicted), 
                xytext=(5, 5), textcoords='offset points', 
                fontsize=8, alpha=0.8
            )

        plt.title(f"Framework Validation (LOCO CV): {metric}\nPredicted vs Actual", fontweight="bold")
        plt.xlabel(f"Actual {metric} (Held-out Condition)")
        plt.ylabel(f"Predicted {metric} (Framework Estimate)")
        plt.xlim(0, axis_max)
        plt.ylim(0, axis_max)
        plt.grid(True, alpha=0.2)
        
        # Deduplicate legend labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc="best")
        
        out_path = PLOTS_DIR_RQ4 / f"rq4_loco_validation_{metric}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved plot -> %s", out_path)
