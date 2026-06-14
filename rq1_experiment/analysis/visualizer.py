"""
rq1_experiment/analysis/visualizer.py
=======================================
All publication-quality plots for the RQ1 analysis.

Generates:
1. Condition-level performance bar charts (all metrics)
2. Freshness decay curves per domain
3. Source type comparison plots
4. Interaction heatmaps (freshness × source type)
5. Feature importance bar chart (RF)
6. Regression coefficient plot
7. Correlation matrix heatmap
8. ANOVA effect size summary chart

All saved to PLOTS_DIR as high-resolution PNGs (300 DPI).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from ..config import ANALYSIS_DIR, ALL_CONDITIONS, CONDITION_META, METRICS_DIR, PLOTS_DIR
from .anova import build_flat_df, load_all_metrics

logger = logging.getLogger(__name__)

# ── Aesthetics ────────────────────────────────────────────────────────────────
PALETTE = {
    "technology": "#2196F3",   # blue
    "healthcare":  "#4CAF50",  # green
    "history":     "#FF9800",  # orange
}
SOURCE_PALETTE = {
    "Academic only":      "#7B1FA2",
    "Academic + News":    "#0288D1",
    "Acad + News + Tech": "#00796B",
    "Acad + Archival":    "#558B2F",
    "Acad + Arch + Ref":  "#BF360C",
}

PLT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "#f9f9f9",
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "font.family":      "sans-serif",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
}
plt.rcParams.update(PLT_STYLE)

DPI = 300


def _save(fig: plt.Figure, name: str) -> None:
    path = PLOTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Condition-level performance bar charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_condition_performance(df: pd.DataFrame, metric: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))

    conditions = df["condition_id"].tolist()
    values     = df[metric].tolist()
    colors     = [PALETTE.get(CONDITION_META[c]["domain"], "#999") for c in conditions]

    bars = ax.bar(conditions, values, color=colors, edgecolor="white", linewidth=0.8, alpha=0.88)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xlabel("Condition")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{title}\n({metric})")
    max_val = max(values) if values else 1.0
    y_max = (max_val * 1.2 + 0.05) if max_val > 1.0 else min(1.0, max_val * 1.2 + 0.05)
    ax.set_ylim(0, y_max)

    # Domain legend
    patches = [mpatches.Patch(color=c, label=d.title()) for d, c in PALETTE.items()]
    ax.legend(handles=patches, loc="upper right", framealpha=0.9)

    _save(fig, f"condition_performance_{metric}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Freshness decay curves per domain
# ─────────────────────────────────────────────────────────────────────────────

def plot_freshness_decay(df: pd.DataFrame, metric: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    domains = ["technology", "healthcare", "history"]
    freshness_order = {
        "technology": ["<= 1 week", "1 wk - 1 mo"],
        "healthcare":  ["1-6 months", "6-12 months"],
        "history":     [">= 6 mo (adj.)"],
    }

    for ax, domain in zip(axes, domains):
        sub = df[df["domain"] == domain].copy()
        color = PALETTE[domain]

        # Sort by mean freshness score descending (fresher → lower age)
        sub = sub.sort_values("freshness_score", ascending=False)
        ax.plot(
            sub["freshness_score"].tolist(),
            sub[metric].tolist(),
            "o-", color=color, linewidth=2, markersize=7, alpha=0.85,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row["condition_id"],
                (row["freshness_score"], row[metric]),
                textcoords="offset points", xytext=(5, 3), fontsize=7,
            )

        ax.set_title(f"{domain.title()}")
        ax.set_xlabel("Freshness Score (0-1)")
        if ax == axes[0]:
            ax.set_ylabel(metric.replace("_", " ").title())

    fig.suptitle(f"Freshness Decay: {metric}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, f"freshness_decay_{metric}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Source type comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_source_type_comparison(df: pd.DataFrame, metric: str) -> None:
    source_configs = df["source_config"].unique().tolist()
    fig, ax = plt.subplots(figsize=(10, 5))

    x      = np.arange(len(source_configs))
    vals   = [df[df["source_config"] == sc][metric].mean() for sc in source_configs]
    colors = [SOURCE_PALETTE.get(sc, "#888") for sc in source_configs]

    bars = ax.bar(x, vals, color=colors, edgecolor="white", alpha=0.88)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(source_configs, rotation=20, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Source Type Configuration vs {metric}")
    max_val = max(vals) if vals else 1.0
    y_max = (max_val * 1.25) if max_val > 1.0 else min(1.0, max_val * 1.25)
    ax.set_ylim(0, y_max)

    _save(fig, f"source_type_comparison_{metric}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Interaction heatmap (freshness × source type)
# ─────────────────────────────────────────────────────────────────────────────

def plot_interaction_heatmap(df: pd.DataFrame, metric: str) -> None:
    pivot = df.pivot_table(values=metric, index="freshness_window", columns="source_config", aggfunc="mean")
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label=metric)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if val > pivot.values.max() * 0.7 else "black")

    ax.set_title(f"Interaction Heatmap: Freshness × Source Type\n({metric})")
    ax.set_xlabel("Source Configuration")
    ax.set_ylabel("Freshness Window")
    plt.tight_layout()
    _save(fig, f"interaction_heatmap_{metric}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Feature importance (Random Forest)
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(rf_results: Dict[str, Any]) -> None:
    metrics = list(rf_results.keys())
    metrics = [m for m in metrics if isinstance(rf_results[m], dict) and "feature_ranking" in rf_results[m]]

    if not metrics:
        return

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        ranking = rf_results[metric]["feature_ranking"]
        features = [r["feature"].replace("_", "\n") for r in ranking]
        importances = [r["importance"] for r in ranking]

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
        ax.barh(features[::-1], importances[::-1], color=colors[::-1], alpha=0.85)
        ax.set_xlabel("Importance")
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlim(0, max(importances) * 1.2)

        for i, (feat, val) in enumerate(zip(features[::-1], importances[::-1])):
            ax.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=8)

    fig.suptitle("Random Forest Feature Importance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "rf_feature_importance")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Correlation matrix heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_matrix(df: pd.DataFrame) -> None:
    cols = [
        "freshness_score", "source_diversity_index",
        "bertscore_f1", "hallucination_rate", "precision_at_5", "ndcg_at_5",
        "rouge_l", "meteor",
    ]
    avail = [c for c in cols if c in df.columns]
    corr  = df[avail].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(avail)))
    ax.set_xticklabels([c.replace("_", "\n") for c in avail], fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(avail)))
    ax.set_yticklabels([c.replace("_", "\n") for c in avail], fontsize=8)

    for i in range(len(avail)):
        for j in range(len(avail)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)

    ax.set_title("Correlation Matrix: Factor Metrics vs Performance Metrics")
    plt.tight_layout()
    _save(fig, "correlation_matrix")


# ─────────────────────────────────────────────────────────────────────────────
# 7. ANOVA effect size summary
# ─────────────────────────────────────────────────────────────────────────────

def plot_anova_effect_sizes(anova_results: Dict[str, Any]) -> None:
    metrics = list(anova_results.get("anova", {}).keys())
    factors = ["freshness", "source_type", "interaction"]

    data: Dict[str, List[float]] = {f: [] for f in factors}
    for metric in metrics:
        anova_data = anova_results["anova"].get(metric, {}).get("factors", {})
        for factor in factors:
            eta2 = anova_data.get(factor, {}).get("eta_squared", 0.0)
            data[factor].append(float(eta2) if eta2 else 0.0)

    x       = np.arange(len(metrics))
    width   = 0.28
    colors  = ["#3F51B5", "#E91E63", "#FF9800"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (factor, color) in enumerate(zip(factors, colors)):
        ax.bar(x + i * width, data[factor], width, label=factor.title(), color=color, alpha=0.82)

    # Reference lines for effect size thresholds
    ax.axhline(0.01, color="gray",  linestyle="--", linewidth=0.8, alpha=0.7, label="Small (η²=0.01)")
    ax.axhline(0.06, color="gray",  linestyle="-.", linewidth=0.8, alpha=0.7, label="Medium (η²=0.06)")
    ax.axhline(0.14, color="black", linestyle=":",  linewidth=0.8, alpha=0.7, label="Large (η²=0.14)")

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylabel("η² (Eta-Squared)")
    ax.set_title("ANOVA Effect Sizes by Factor and Metric")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    plt.tight_layout()
    _save(fig, "anova_effect_sizes")


# ─────────────────────────────────────────────────────────────────────────────
# Master runner
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots() -> None:
    """Generate all plots from saved analysis results."""
    logger.info("Generating all RQ1 plots …")

    all_metrics = load_all_metrics()
    df          = build_flat_df(all_metrics)

    primary_metrics = ["bertscore_f1", "hallucination_rate", "precision_at_5", "ndcg_at_5", "human_eval_score"]
    available_metrics = [m for m in primary_metrics if m in df.columns]

    # 1. Condition performance bar charts
    for metric in available_metrics:
        plot_condition_performance(df, metric, "RQ1 Experiment")

    # 2. Freshness decay curves
    for metric in available_metrics:
        plot_freshness_decay(df, metric)

    # 3. Source type comparison
    for metric in available_metrics:
        plot_source_type_comparison(df, metric)

    # 4. Interaction heatmaps
    for metric in available_metrics:
        plot_interaction_heatmap(df, metric)

    # 5. Correlation matrix
    plot_correlation_matrix(df)

    # 6. RF feature importance
    rf_path = ANALYSIS_DIR / "random_forest_results.json"
    if rf_path.exists():
        rf_results = json.loads(rf_path.read_text(encoding="utf-8"))
        plot_feature_importance(rf_results)

    # 7. ANOVA effect sizes
    anova_path = ANALYSIS_DIR / "anova_results.json"
    if anova_path.exists():
        anova_results = json.loads(anova_path.read_text(encoding="utf-8"))
        plot_anova_effect_sizes(anova_results)

    logger.info("All plots saved to %s", PLOTS_DIR)
