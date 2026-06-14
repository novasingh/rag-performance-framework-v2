"""
rq2_experiment/analysis/visualizer_rq2.py
==========================================
Generates all RQ2-specific plots.

Plots produced:
  1. source_comparison_<metric>.png    — Grouped bars: 3 source configs × 3 domains
  2. marginal_gain_<metric>.png        — Bar chart: gain from adding News and TechDocs
  3. sdi_scatter_<metric>.png          — Scatter: SDI vs metric, coloured by domain
  4. interaction_heatmap_rq2_<metric>.png — Source × Volatility heatmap
  5. performance_radar.png             — Radar chart of all metrics per source config
  6. source_effect_size.png            — η² effect sizes from one-way ANOVA
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from ..config import ANALYSIS_DIR_RQ2, PLOTS_DIR_RQ2, PRIMARY_METRICS, SOURCE_LEVEL_LABELS

logger = logging.getLogger(__name__)

# ── Aesthetic constants ────────────────────────────────────────────────────────
PALETTE    = ["#3B82F6", "#10B981", "#F59E0B"]   # blue, green, amber
DOMAIN_COLORS = {"technology": "#EF4444", "healthcare": "#8B5CF6", "history": "#F97316"}
FONT_TITLE = 14
FONT_LABEL = 11
FONT_TICK  = 9
DPI        = 150

METRIC_LABELS = {
    "bertscore_f1":       "BERTScore F1",
    "hallucination_rate": "Hallucination Rate",
    "precision_at_5":     "Precision@5",
    "ndcg_at_5":          "nDCG@5",
    "rouge_l":            "ROUGE-L",
    "meteor":             "METEOR",
    "human_eval_score":   "Human Eval (Likert)",
}

SOURCE_SHORT = {
    "Academic Only":            "Acad.",
    "Academic + News":          "Acad.+News",
    "Academic + News + Tech":   "Acad.+News+Tech",
}


def _load(fname: str) -> Dict:
    p = ANALYSIS_DIR_RQ2 / fname
    if not p.exists():
        raise FileNotFoundError(f"Missing {p} — run analysis steps first.")
    return json.loads(p.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, name: str) -> None:
    path = PLOTS_DIR_RQ2 / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Grouped bar chart: source level × domain → metric
# ─────────────────────────────────────────────────────────────────────────────

def plot_source_comparison(anova_results: Dict) -> None:
    descriptive = anova_results.get("descriptive_by_source_level", {})
    domains     = ["technology", "healthcare", "history"]

    # We need per-domain per-source-level means → from per_domain_anova
    per_domain = anova_results.get("per_domain_anova", {})

    for metric in PRIMARY_METRICS:
        if metric not in per_domain:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        x       = np.arange(len(domains))
        width   = 0.25

        for i, (level, label) in enumerate(SOURCE_LEVEL_LABELS.items()):
            heights = []
            for domain in domains:
                domain_data = per_domain[metric].get(domain, {})
                means       = domain_data.get("group_means", {})
                # Look up the label as key
                val = means.get(label, None)
                heights.append(val if val is not None else 0.0)

            bars = ax.bar(x + (i - 1) * width, heights, width,
                          label=SOURCE_SHORT.get(label, label),
                          color=PALETTE[i], alpha=0.88, edgecolor="white", linewidth=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in domains], fontsize=FONT_TICK)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=FONT_LABEL)
        ax.set_title(f"Source Type Contribution: {METRIC_LABELS.get(metric, metric)} by Domain",
                     fontsize=FONT_TITLE, fontweight="bold")
        ax.legend(fontsize=FONT_TICK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        fig.tight_layout()
        _save(fig, f"rq2_source_comparison_{metric}.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Marginal gain bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_marginal_gains(contribution_results: Dict) -> None:
    gains = contribution_results.get("marginal_gains", {})
    domains = ["technology", "healthcare", "history"]

    for metric in PRIMARY_METRICS:
        if metric not in gains:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(domains))
        width = 0.35

        step_a = [gains[metric].get(d, {}).get("step_a_gain_adding_news", 0) or 0 for d in domains]
        step_b = [gains[metric].get(d, {}).get("step_b_gain_adding_tech", 0) or 0 for d in domains]

        bars_a = ax.bar(x - width / 2, step_a, width, label="+ News (Step A)",
                        color=PALETTE[1], alpha=0.88, edgecolor="white")
        bars_b = ax.bar(x + width / 2, step_b, width, label="+ Tech Docs (Step B)",
                        color=PALETTE[2], alpha=0.88, edgecolor="white")

        # Value labels on bars
        for bar in bars_a + bars_b:
            h = bar.get_height()
            if abs(h) > 0.001:
                ax.annotate(f"{h:+.3f}",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 4 if h >= 0 else -12),
                            textcoords="offset points", ha="center", fontsize=8)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in domains], fontsize=FONT_TICK)
        ax.set_ylabel(f"Marginal Gain in {METRIC_LABELS.get(metric, metric)}", fontsize=FONT_LABEL)
        ax.set_title(f"Marginal Source-Type Contribution: {METRIC_LABELS.get(metric, metric)}",
                     fontsize=FONT_TITLE, fontweight="bold")
        ax.legend(fontsize=FONT_TICK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        fig.tight_layout()
        _save(fig, f"rq2_marginal_gain_{metric}.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SDI scatter plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_sdi_scatter(df_dict: Dict) -> None:
    """df_dict: the raw per-condition data we can reconstruct from per_domain_anova."""
    # Load the metrics JSON directly to get SDI values
    from ..config import METRICS_DIR, CONDITION_META, SOURCE_LEVEL_MAP
    import json as _json

    path = METRICS_DIR / "all_conditions_metrics.json"
    if not path.exists():
        logger.warning("all_conditions_metrics.json not found — skipping SDI scatter")
        return

    raw = _json.loads(path.read_text(encoding="utf-8"))

    rows = []
    for cid, m in raw.items():
        meta = CONDITION_META.get(cid, {})
        rows.append({
            "condition_id":   cid,
            "domain":         meta.get("domain", ""),
            "source_level":   SOURCE_LEVEL_MAP.get(cid, 0),
            "sdi":            m.get("source_diversity_index", {}).get("mean", 0.0),
            "bertscore_f1":   m.get("bertscore_f1", {}).get("mean", 0.0),
            "hallucination_rate": m.get("hallucination", {}).get("mean", 0.0),
            "precision_at_5": m.get("retrieval", {}).get("precision_at_5", {}).get("mean", 0.0),
            "ndcg_at_5":      m.get("retrieval", {}).get("ndcg_at_5", {}).get("mean", 0.0),
            "human_eval_score": m.get("human_eval_score", {}).get("mean", 0.0),
        })

    import pandas as pd
    df = pd.DataFrame(rows)

    for metric in PRIMARY_METRICS:
        fig, ax = plt.subplots(figsize=(8, 5))
        for domain, color in DOMAIN_COLORS.items():
            sub = df[df["domain"] == domain]
            ax.scatter(sub["sdi"], sub[metric], c=color, s=90, alpha=0.85,
                       label=domain.capitalize(), zorder=3, edgecolors="white", linewidth=0.6)
            # Condition ID annotations
            for _, row in sub.iterrows():
                ax.annotate(row["condition_id"], (row["sdi"], row[metric]),
                            fontsize=7, ha="left", va="bottom",
                            xytext=(3, 3), textcoords="offset points", color=color)

        # Trendline
        x_all = df["sdi"].values
        y_all = df[metric].values
        if len(x_all) > 2:
            z = np.polyfit(x_all, y_all, 1)
            p = np.poly1d(z)
            xs = np.linspace(x_all.min(), x_all.max(), 100)
            ax.plot(xs, p(xs), "k--", alpha=0.4, linewidth=1.2, label="Trend")

        ax.set_xlabel("Source Diversity Index (SDI)", fontsize=FONT_LABEL)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=FONT_LABEL)
        ax.set_title(f"SDI vs. {METRIC_LABELS.get(metric, metric)}", fontsize=FONT_TITLE, fontweight="bold")
        ax.legend(fontsize=FONT_TICK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, linestyle="--")
        fig.tight_layout()
        _save(fig, f"rq2_sdi_scatter_{metric}.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Source × Volatility interaction heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_interaction_heatmap(anova_results: Dict) -> None:
    per_domain = anova_results.get("per_domain_anova", {})
    domains    = ["technology", "healthcare", "history"]
    labels     = list(SOURCE_LEVEL_LABELS.values())

    for metric in PRIMARY_METRICS:
        if metric not in per_domain:
            continue

        matrix = np.zeros((3, 3))   # source level × domain
        for j, domain in enumerate(domains):
            domain_data = per_domain[metric].get(domain, {})
            means = domain_data.get("group_means", {})
            for i, label in enumerate(labels):
                matrix[i, j] = means.get(label, 0.0)

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(matrix, cmap="YlOrRd" if metric == "hallucination_rate" else "YlGnBu", aspect="auto")

        ax.set_xticks(range(3))
        ax.set_xticklabels([d.capitalize() for d in domains], fontsize=FONT_TICK)
        ax.set_yticks(range(3))
        ax.set_yticklabels([SOURCE_SHORT.get(l, l) for l in labels], fontsize=FONT_TICK)

        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if matrix[i, j] > (matrix.max() * 0.65) else "black")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Source Type × Domain Volatility: {METRIC_LABELS.get(metric, metric)}",
                     fontsize=FONT_TITLE, fontweight="bold")
        ax.set_xlabel("Domain (Volatility ↓ right to left)", fontsize=FONT_LABEL)
        ax.set_ylabel("Source Configuration", fontsize=FONT_LABEL)
        fig.tight_layout()
        _save(fig, f"rq2_interaction_heatmap_{metric}.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Radar / spider chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_performance_radar(contribution_results: Dict) -> None:
    profiles  = contribution_results.get("performance_profiles", {}).get("profiles", {})
    metrics   = ["precision_at_5", "ndcg_at_5", "bertscore_f1", "rouge_l", "meteor"]
    labels    = [METRIC_LABELS.get(m, m) for m in metrics]
    N         = len(metrics)
    angles    = [n / float(N) * 2 * math.pi for n in range(N)]
    angles   += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    for color, (source_label, profile) in zip(PALETTE, profiles.items()):
        values = [profile.get(m, {}).get("mean") or 0.0 for m in metrics]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=SOURCE_SHORT.get(source_label, source_label), color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=FONT_TICK)
    ax.set_ylim(0, 1.05)
    ax.set_title("Performance Profile by Source Configuration",
                 fontsize=FONT_TITLE, fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=FONT_TICK)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "rq2_performance_radar.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Effect size bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_effect_sizes(anova_results: Dict) -> None:
    one_way = anova_results.get("one_way_anova", {})
    two_way = anova_results.get("two_way_anova", {})

    metrics = [m for m in PRIMARY_METRICS if m in one_way]
    if not metrics:
        logger.warning("No one-way ANOVA results to plot — skipping effect size chart")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # One-way η²
    ax = axes[0]
    eta2s   = [one_way[m].get("eta_squared", 0) for m in metrics]
    colors  = [PALETTE[1] if e >= 0.14 else PALETTE[0] if e >= 0.06 else PALETTE[2] for e in eta2s]
    bars    = ax.barh([METRIC_LABELS.get(m, m) for m in metrics], eta2s, color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(0.14, color="red",    linestyle="--", alpha=0.6, linewidth=1, label="Large (0.14)")
    ax.axvline(0.06, color="orange", linestyle="--", alpha=0.6, linewidth=1, label="Medium (0.06)")
    ax.axvline(0.01, color="green",  linestyle="--", alpha=0.6, linewidth=1, label="Small (0.01)")
    ax.set_xlabel("η² (Eta-Squared)", fontsize=FONT_LABEL)
    ax.set_title("Source Type Effect Sizes\n(One-Way ANOVA)", fontsize=FONT_TITLE - 1, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, eta2s):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    # Two-way η² (source_type, domain_volatility, interaction)
    ax2 = axes[1]
    factor_keys  = ["source_type", "domain_volatility", "interaction"]
    factor_names = ["Source Type", "Domain Volatility", "Interaction"]
    x = np.arange(len(metrics))
    width = 0.25

    for i, (fk, fname) in enumerate(zip(factor_keys, factor_names)):
        vals = []
        for m in metrics:
            mdata = two_way.get(m, {})
            vals.append(mdata.get("factors", {}).get(fk, {}).get("eta_squared", 0) or 0)
        ax2.bar(x + (i - 1) * width, vals, width, label=fname, color=PALETTE[i], alpha=0.85, edgecolor="white")

    ax2.set_xticks(x)
    ax2.set_xticklabels([METRIC_LABELS.get(m, m) for m in metrics], fontsize=8, rotation=15, ha="right")
    ax2.set_ylabel("η² (Eta-Squared)", fontsize=FONT_LABEL)
    ax2.set_title("Two-Way ANOVA Effect Sizes\n(Source × Volatility)", fontsize=FONT_TITLE - 1, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    _save(fig, "rq2_effect_sizes.png")


# ─────────────────────────────────────────────────────────────────────────────
# Master runner
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_rq2_plots() -> None:
    logger.info("Generating RQ2 plots …")

    anova_results        = _load("rq2_anova_results.json")
    contribution_results = _load("rq2_contribution_results.json")

    plot_source_comparison(anova_results)
    plot_marginal_gains(contribution_results)
    plot_sdi_scatter({})
    plot_interaction_heatmap(anova_results)
    plot_performance_radar(contribution_results)
    plot_effect_sizes(anova_results)

    logger.info("All RQ2 plots saved to %s", PLOTS_DIR_RQ2)
