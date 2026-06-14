# RQ3 Experiment: Predictive Framework Development

## Overview
RQ3 develops and validates a predictive framework that estimates RAG system performance from dataset-level factors (freshness, source type, domain volatility). The framework uses three model families: OLS (interpretable), Random Forest (non-linear), and XGBoost (gradient-boosted).

## Prerequisites
RQ1 experiment metrics must be generated first:
```powershell
python -m rq1_experiment.run_analysis --step metrics
```

## Execution

### Train all models
```powershell
python -m rq3_experiment.run_rq3
```
This runs the complete improved pipeline:
1. **OLS Regression** — Cross-validated with 6 base features (no polynomial terms to avoid multicollinearity)
2. **Random Forest** — 500 trees with GridSearch hyperparameter tuning, using 9 features (including polynomial terms)
3. **XGBoost** — Gradient boosting with early stopping, using 9 features
4. **Decay Curve Fitting** — Exponential, polynomial, and logistic decay per domain
5. **Model Comparison** — Printed summary of OLS vs RF vs XGBoost MAE
6. **API Test** — Prediction test across all 3 domains

### Use the trained predictor
```python
from rq3_experiment.framework.predictor import RAGPerformancePredictor

predictor = RAGPerformancePredictor()

# Predict for Technology, 14-day old documents, high diversity
result = predictor.predict(
    domain="technology",
    avg_age_days=14.0,
    source_diversity_index=0.35
)

print(result["predictions"])
```

## Features Used

| Feature | Type | Description |
|---------|------|-------------|
| freshness_score | Base | Exponential half-life decay (180-day half-life) |
| source_diversity_index | Base | Shannon entropy of source type distribution |
| domain_volatility | Base | 1.0 (tech), 0.5 (healthcare), 0.0 (history) |
| fresh_x_diversity | Interaction | freshness_score * source_diversity_index |
| fresh_x_volatility | Interaction | freshness_score * domain_volatility |
| source_x_volatility | Interaction | source_diversity_index * domain_volatility |
| freshness_score_sq | Polynomial | freshness_score^2 (tree models only) |
| source_diversity_index_sq | Polynomial | SDI^2 (tree models only) |
| domain_volatility_sq | Polynomial | volatility^2 (tree models only) |

**Note:** OLS uses only the 6 base+interaction features. Tree models (RF, XGBoost) use all 9 features including polynomial terms, which they handle naturally without multicollinearity issues.

## Output Files

| File | Description |
|------|-------------|
| `rq3_experiment/results/rq3_models.json` | All trained models and CV results |
| `rq3_experiment/results/rq3_models_summary.md` | Model summary tables |
| `rq3_experiment/results/rq3_training.log` | Training log |

## Model Performance (10-fold CV)

| Metric | OLS (MAE) | RF (MAE) | XGB (MAE) | Best Model |
|--------|-----------|----------|-----------|------------|
| bertscore_f1 | 0.1068 | **0.1067** | 0.1071 | RF |
| hallucination_rate | **0.3909** | 0.3915 | 0.3931 | OLS |
| precision_at_5 | 0.2193 | **0.2158** | 0.2171 | RF |
| ndcg_at_5 | 0.3035 | **0.3029** | 0.3098 | RF |
| human_eval_score | 0.7126 | **0.7073** | 0.7173 | RF |

## Predictor API

### Inputs
- `domain`: "technology", "healthcare", or "history"
- `avg_age_days`: Average age of documents in days
- `source_diversity_index`: Shannon entropy (0.0 to ~0.45)

### Output
```python
{
  "inputs": {
    "domain": "technology",
    "avg_age_days": 14.0,
    "freshness_score": 0.9484,
    "source_diversity_index": 0.35
  },
  "predictions": {
    "bertscore_f1": {
      "expected": 0.7773,
      "lower_bound": 0.6706,
      "upper_bound": 0.8841,
      "mae_margin": 0.1068,
      "n_features_matched": "6/6"
    },
    "hallucination_rate": { ... },
    "precision_at_5": { ... },
    "ndcg_at_5": { ... },
    "human_eval_score": { ... }
  }
}
```

### Notes
- The predictor auto-detects whether the OLS model has 6 features (base) or 9 features (with polynomials)
- Domain warnings are triggered if `source_diversity_index > 0.45` (exceeds training range)
- Feature matching warnings are issued if coefficient names don't match

## Comparison with Original
The improved pipeline adds:
- Polynomial features for tree models (captures non-linear effects)
- XGBoost as a third model family
- Logistic decay curves (better for bounded metrics)
- Model comparison table on every run
- Multi-domain API tests