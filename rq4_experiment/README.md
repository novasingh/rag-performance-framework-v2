# RQ4 Experiment: Cross-Domain Framework Validation

## Overview
RQ4 evaluates the predictive framework using leave-one-condition-out (LOCO) cross-validation across all 12 experimental conditions. This determines the framework's reliable operating boundaries by testing against held-out configurations.

## Prerequisites
Both RQ1 metrics and RQ3 models must be generated first:
```powershell
python -m rq1_experiment.run_analysis --step metrics
python -m rq3_experiment.run_rq3
```

## Execution

### Run LOCO validation
```powershell
python -m rq4_experiment.run_rq4
```

This performs:
1. **LOCO cross-validation**: For each of the 12 conditions, train on the other 11 and predict the held-out condition
2. **Bootstrap confidence intervals**: 1,000 bootstrap iterations per condition for 95% CI
3. **Boundary condition detection**: Flags conditions where the 95% CI does not intersect the diagonal (y = x)
4. **Validation plots**: Scatter plots for all 5 metrics

## Output Files

| File | Description |
|------|-------------|
| `rq4_experiment/results/rq4_validation.json` | Full LOCO results with CIs and boundary flags |
| `rq4_experiment/results/rq4_validation.log` | Validation run log |
| `rq4_experiment/results/plots/rq4_loco_validation_*.png` (5) | LOCO scatter plots per metric |

## Key Results

### Boundary Conditions by Domain

| Domain | Boundary Conditions | Details |
|--------|-------------------|---------|
| History | **0** | All 5 metrics predictable across all conditions |
| Technology | **3** | C4, C5, C6 — all in human_eval_score |
| Healthcare | **3** | C7, C8, C9 — all in human_eval_score |

### Boundary Conditions by Metric

| Metric | Boundaries | Reliability |
|--------|-----------|-------------|
| Precision@5 | 0 | High |
| nDCG@5 | 0 | High |
| BERTScore F1 | 0 | High |
| Hallucination Rate | 0 | High |
| Human Eval Score | 6 | Low |

### Interpretation
- The **retrieval-oriented metrics** (P@5, nDCG@5, BERTScore F1, Hallucination Rate) are **reliably predictable** across all domains — zero boundary conditions
- **Human Eval Score** is the most challenging — 6 boundary conditions across Technology and Healthcare
- **History** is the most stable domain — zero boundaries across all 5 metrics
- The framework is useful for **screening and comparative analysis** for retrieval metrics, but human-rated quality estimation needs further improvement

## LOCO Validation Tables

### nDCG@5
| Condition | Actual | Predicted | 95% CI | Status |
|-----------|--------|-----------|--------|--------|
| C1 (Tech) | 0.3475 | 0.4325 | [0.4570, 0.5781] | Reliable |
| C11 (Hist) | 0.9233 | 0.9241 | [0.8986, 0.9514] | Reliable |
| C9 (Health) | 0.6267 | 0.7956 | [0.9074, 1.0217] | Reliable |

### Precision@5
| Condition | Actual | Predicted | 95% CI | Status |
|-----------|--------|-----------|--------|--------|
| C1 (Tech) | 0.1110 | 0.1491 | [0.1659, 0.2087] | Reliable |
| C11 (Hist) | 0.5880 | 0.5876 | [0.5464, 0.6281] | Reliable |
| C9 (Health) | 0.3480 | 0.4564 | [0.5214, 0.6081] | Reliable |

## Plot Files
- `rq4_loco_validation_bertscore_f1.png`
- `rq4_loco_validation_hallucination_rate.png`
- `rq4_loco_validation_human_eval_score.png`
- `rq4_loco_validation_ndcg_at_5.png`
- `rq4_loco_validation_precision_at_5.png`

Each plot shows the 12 conditions as points (colored by domain) with 95% CI error bars and the diagonal y=x line.