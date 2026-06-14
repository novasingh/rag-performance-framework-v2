# RQ2 Experiment: Source Type Contribution Analysis

## Overview
RQ2 examines the individual contribution of source type configurations to RAG system performance across multiple evaluation metrics, and whether this contribution varies by domain volatility. RQ2 analysis runs as part of the RQ1 analysis pipeline.

## Prerequisites
RQ1 experiment must be complete first:
```powershell
python -m rq1_experiment.run_experiment
```

## Execution

Run the RQ2 analysis (this is Step 2 of the RQ1 pipeline):
```powershell
python -m rq1_experiment.run_analysis --step all
```

To run RQ2-specific steps only:
```powershell
# Compute metrics
python -m rq1_experiment.run_analysis --step metrics

# Run ANOVA (one-way + two-way)
python -m rq1_experiment.run_analysis --step anova

# Generate plots (source comparison, interaction heatmaps, marginal gains, etc.)
python -m rq1_experiment.run_analysis --step plots
```

## What RQ2 Measures

### Source-Level Comparison
Three source diversity levels compared across all metrics:
- **Single-Source**: One document type (Academic only or similar)
- **Two-Source Mix**: Two document types combined (e.g., Academic + News)
- **Full Diversity**: Three or more document types (e.g., Academic + News + Technical)

### Statistical Tests
- **One-way ANOVA**: Tests if source level significantly affects each metric
- **Two-way ANOVA**: Tests interaction between source type and domain volatility
- **Effect sizes**: Eta-squared (small = 0.01, medium = 0.06, large = 0.14)

## Output Files

| File | Description |
|------|-------------|
| `rq1_experiment/results/analysis/rq2/source_level_summary.md` | Source-level means table |
| `rq1_experiment/results/analysis/rq2/anova_summary.md` | One-way ANOVA F/p/eta-squared |
| `rq1_experiment/results/analysis/rq2/rq2_anova_results.json` | Full ANOVA results |
| `rq1_experiment/results/analysis/rq2/rq2_contribution_results.json` | Source contribution analysis |
| `rq1_experiment/results/analysis/rq2/rq2_analysis.log` | RQ2 training log |

## Key Results

### Source-Level Means (Across All Domains)

| Source Level | P@5 | nDCG@5 | BERTScore F1 | Hallucination | Human Eval | SDI |
|---|---|---|---|---|---|---|
| Single-Source | 0.290 | 0.604 | 0.126 | 0.518 | 3.232 | 0.000 |
| Two-Source Mix | 0.352 | 0.679 | 0.136 | 0.564 | 3.451 | 0.258 |
| Full Diversity | 0.391 | 0.669 | 0.117 | 0.498 | 3.429 | 0.449 |

### ANOVA Results for Source Level

| Metric | F | p-value | eta-squared | Effect Size |
|---|---|---|---|---|
| precision_at_5 | 22.11 | < 0.001 | 0.019 | small |
| ndcg_at_5 | 9.08 | 0.0001 | 0.008 | negligible |
| bertscore_f1 | 1.61 | 0.200 | 0.001 | negligible |
| hallucination_rate | 1.86 | 0.155 | 0.002 | negligible |
| human_eval_score | 20.14 | < 0.001 | 0.017 | small |

### Domain-Specific Effects
- **Technology (High Volatility)**: Full Diversity raises P@5 from 0.164 to 0.318 (+94%)
- **Healthcare (Medium Volatility)**: Full Diversity **reduces** P@5 from 0.404 to 0.339 (-16%)
- **History (Low Volatility)**: Adding news sources raises P@5 from 0.428 to 0.588 (+37%)

## Plots
All plots are generated in `rq1_experiment/results/plots/rq2/`:
- `rq2_source_comparison_*.png` — Source level bar charts per metric
- `rq2_interaction_heatmap_*.png` — Freshness x Source interaction
- `rq2_marginal_gain_*.png` — Stepwise gains from adding sources
- `rq2_sdi_scatter_*.png` — SDI vs performance scatter plots
- `rq2_performance_radar.png` — Radar chart across all metrics
- `rq2_effect_sizes.png` — ANOVA effect sizes with significance stars