# RQ1 Experiment: Freshness & Source Type Impact Analysis

## Overview
RQ1 evaluates how dataset freshness and source type relate to RAG system performance across high- (Technology), medium- (Healthcare), and low-volatility (History) domains.

## Experimental Conditions (12 total)

| Condition | Domain | Freshness | Source Config | N Queries |
|-----------|--------|-----------|---------------|-----------|
| C1 | technology | <= 1 week | Academic only | 200 |
| C2 | technology | <= 1 week | Academic + News | 200 |
| C3 | technology | <= 1 week | Acad + News + Tech | 200 |
| C4 | technology | 1 wk - 1 mo | Academic only | 200 |
| C5 | technology | 1 wk - 1 mo | Academic + News | 200 |
| C6 | technology | 1 wk - 1 mo | Acad + News + Tech | 200 |
| C7 | healthcare | 1-6 months | Academic only | 200 |
| C8 | healthcare | 1-6 months | Academic + News | 200 |
| C9 | healthcare | 6-12 months | Acad + News + Tech | 200 |
| C10 | history | >= 6 mo (adj.) | Academic only | 200 |
| C11 | history | >= 6 mo (adj.) | Acad + Archival | 200 |
| C12 | history | >= 6 mo (adj.) | Acad + Arch + Ref | 200 |

## Execution Order

### Step 1: Run the full RAG experiment
```powershell
python -m rq1_experiment.run_experiment
```
- Generates raw outputs for all 12 conditions
- Saves to `rq1_experiment/results/raw_outputs/{Condition}_outputs.json`
- Auto-resumes if interrupted (checkpoint every 10 queries)

Options:
```powershell
# Build indexes only
python -m rq1_experiment.run_experiment --step indexes

# Build query banks only
python -m rq1_experiment.run_experiment --step queries

# Run specific conditions only
python -m rq1_experiment.run_experiment --conditions C3 C5

# Force rebuild (overwrite cached)
python -m rq1_experiment.run_experiment --force-rebuild
```

### Step 2: Compute metrics and run statistical analysis
```powershell
python -m rq1_experiment.run_analysis
```
This runs all 5 analysis steps:
1. **metrics** — Compute evaluation metrics (BERTScore, hallucination, P@5, nDCG) for all conditions
2. **anova** — Two-way ANOVA + post-hoc Tukey HSD
3. **regression** — OLS multiple regression + decay curve fitting
4. **rf** — Random Forest with 500 trees, 10-fold CV
5. **plots** — Generate all visualizations

Options:
```powershell
# Run a specific step only
python -m rq1_experiment.run_analysis --step metrics
python -m rq1_experiment.run_analysis --step anova
python -m rq1_experiment.run_analysis --step regression
python -m rq1_experiment.run_analysis --step rf
python -m rq1_experiment.run_analysis --step plots

# Limit to specific conditions
python -m rq1_experiment.run_analysis --conditions C1 C2 C3

# Change BERTScore batch size (default 8)
python -m rq1_experiment.run_analysis --bertscore-batch 16
```

## Output Files

### Metrics (per condition)
- `rq1_experiment/results/metrics/{Condition}_metrics.json`
- `rq1_experiment/results/metrics/all_conditions_metrics.json`
- `rq1_experiment/results/metrics/condition_metrics_summary.md`

### Analysis
- `rq1_experiment/results/analysis/rq2/anova_summary.md` — One-way ANOVA table
- `rq1_experiment/results/analysis/rq2/source_level_summary.md` — Source-level means
- `rq1_experiment/results/analysis/rq2/rq2_anova_results.json`
- `rq1_experiment/results/analysis/rq2/rq2_contribution_results.json`
- `rq1_experiment/results/analysis/rq2/rq2_analysis.log`

### Plots (in rq1_experiment/results/plots/rq2/)
| File | Description |
|------|-------------|
| `rq2_source_comparison_*.png` (5) | Source comparison bar charts |
| `rq2_interaction_heatmap_*.png` (5) | Freshness x Source interaction heatmaps |
| `rq2_marginal_gain_*.png` (5) | Marginal gain from adding source types |
| `rq2_sdi_scatter_*.png` (5) | SDI vs metric scatter plots |
| `rq2_performance_radar.png` | Multi-metric radar chart |
| `rq2_effect_sizes.png` | ANOVA effect size bar chart |

### Logs
- `rq1_experiment/results/experiment.log` — Experiment run log
- `rq1_experiment/results/analysis.log` — Analysis log
- `rq1_experiment/results/rq2_analysis.log` — RQ2-specific analysis log

## Key Results

| Source Level | P@5 | nDCG@5 | Hallucination | Human Eval |
|---|---|---|---|---|
| Single-Source | 0.290 | 0.604 | 0.518 | 3.232 |
| Two-Source Mix | 0.352 | 0.679 | 0.564 | 3.451 |
| Full Diversity | 0.391 | 0.669 | 0.498 | 3.429 |

- **Technology**: Full Diversity increases P@5 from 0.164 to 0.318 (+94%)
- **Healthcare**: Full Diversity **reduces** P@5 from 0.404 to 0.339 (-16%)
- **History**: News sources provide largest gain (+37% P@5)

## Human Evaluation
- Pooled Cohen's kappa: **0.921** (threshold: 0.70)
- Per dimension: relevance=0.978, correctness=0.977, freshness=0.965, hallucination=0.764
- Evaluated on ~20% stratified sample per condition (n=40 per condition)

## Configuration
See `rq1_experiment/config.py` for full parameter list including:
- Embedding model: all-MiniLM-L6-v2
- Retrieval top-k: 5
- Generator backend: digitalocean (llama-4-maverick)
- Temperature: 0.0 (deterministic)