"""
Framework Backend API (standalone)
==================================
FastAPI server for the RAG Performance Prediction Framework.

Self-contained: imports only from ``core`` and reads only from ``backend/data``.
Every figure returned is computed from the bundled RQ1 metrics, the RQ3 model,
and the RQ4 validation - nothing is hardcoded.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core import config
from core.predictor import RAGPerformancePredictor
from core.recommender import (
    DOMAIN_VOLATILITY_SCORE,
    SOURCE_LEVELS,
    SOURCE_LEVEL_LABELS,
    build_recommendations,
    domain_guideline,
    source_levels_for_domain,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("framework")

app = FastAPI(
    title="RAG Performance Prediction Framework",
    description="A Framework for Predicting RAG System Performance Based on Relevant Factors",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Predictor (the RQ3 model) ──────────────────────────────────────────────────
predictor: Optional[RAGPerformancePredictor] = None


def _domain_of_condition(cid: str) -> str:
    return config.load_condition_meta().get(cid, {}).get("domain", "unknown")


@app.on_event("startup")
def _startup() -> None:
    global predictor
    try:
        predictor = RAGPerformancePredictor(config.load_models())
        logger.info("Predictor and data loaded.")
    except Exception as exc:  # pragma: no cover - surfaced via /health
        logger.error("Startup load failed: %s", exc)
        predictor = None


# ── Request models ─────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    domain: str                      # technology | healthcare | history
    avg_age_days: Optional[float] = None
    source_config: str = "academic"  # academic | academic+news | full
    source_diversity_index: Optional[float] = None


class BatchPredictRequest(BaseModel):
    predictions: List[PredictionRequest]


# ── Helpers ─────────────────────────────────────────────────────────────────────
def _resolve_sdi(domain: str, source_config: str, override: Optional[float]) -> float:
    """Resolve the source-diversity index from real condition data for the domain."""
    if override is not None:
        return override
    levels = source_levels_for_domain(domain)
    if not levels:
        return 0.0
    order = {"academic": 0, "academic+news": 1, "full": 2}
    idx = min(order.get(source_config, 0), len(levels) - 1)
    return levels[idx]["source_diversity_index"]


def _run_prediction(req: PredictionRequest) -> dict:
    if predictor is None:
        raise HTTPException(500, "Predictor not loaded. Run sync_data.py and restart.")
    if req.domain not in DOMAIN_VOLATILITY_SCORE:
        raise HTTPException(400, f"Invalid domain '{req.domain}'. Use technology, healthcare, or history.")

    age_days = req.avg_age_days if req.avg_age_days is not None else 30.0
    sdi = _resolve_sdi(req.domain, req.source_config, req.source_diversity_index)

    prediction = predictor.predict(req.domain, avg_age_days=age_days, source_diversity_index=sdi)
    recommendations = build_recommendations(predictor, req.domain, req.source_config, prediction)

    return {
        "inputs": {
            "domain": req.domain,
            "domain_label": config.DOMAIN_LABELS.get(req.domain, req.domain),
            "avg_age_days": round(age_days, 1),
            "freshness_score": prediction["inputs"]["freshness_score"],
            "source_config": req.source_config,
            "source_diversity_index": round(sdi, 3),
            "source_level": recommendations["source_level"],
        },
        "predictions": prediction["predictions"],
        "warnings": prediction.get("warnings", []),
        "recommendations": recommendations,
        "deployment_readiness_score": recommendations["deployment_readiness"],
    }


# ── Routes ──────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "RAG Performance Prediction Framework",
        "status": "online",
        "version": app.version,
        "endpoints": ["/summary", "/predict", "/batch-predict", "/conditions",
                      "/freshness", "/sources", "/validation", "/model-info",
                      "/deployment-guidelines", "/metrics-meta"],
    }


@app.get("/metrics-meta")
def metrics_meta():
    """Metric metadata so the frontend renders without hardcoding metric names."""
    return {"metrics": config.PRIMARY_METRICS,
            "domains": config.DOMAIN_LABELS,
            "domain_volatility": config.DOMAIN_VOLATILITY_SCORE}


@app.post("/predict")
def predict(req: PredictionRequest):
    return _run_prediction(req)


@app.post("/batch-predict")
def batch_predict(req: BatchPredictRequest):
    results = []
    for p in req.predictions:
        try:
            results.append(_run_prediction(p))
        except HTTPException as exc:
            results.append({"inputs": p.model_dump(), "error": exc.detail})
    return {"results": results}


@app.get("/summary")
def summary():
    models = config.load_models()
    kappa = config.load_kappa()
    validation = config.load_validation()
    ols = models.get("ols", {})

    return {
        "total_observations": models.get("n_samples", 0),
        "n_features": models.get("n_features", 0),
        "domains": len(config.DOMAIN_LABELS),
        "conditions": len(config.load_condition_meta()),
        "models": ["OLS Regression", "Random Forest", "XGBoost"],
        "human_evaluation": {
            "kappa": kappa.get("mean_kappa"),
            "threshold": kappa.get("threshold"),
            "threshold_met": kappa.get("threshold_met"),
            "kappa_by_dimension": kappa.get("cohen_kappa", {}),
        },
        "boundary_conditions": validation.get("boundary_condition_count_by_domain", {}),
        "metrics_summary": {
            m: {
                "mae": ols.get(m, {}).get("cv_bounds", {}).get("mae_mean", 0),
                "r2": ols.get(m, {}).get("r_squared", 0),
            }
            for m in ols
        },
    }


@app.get("/conditions")
def conditions():
    meta = config.load_condition_meta()
    metrics = config.load_condition_metrics()
    out = {}
    for cid, m in meta.items():
        cm = metrics.get(cid, {})
        retr = cm.get("retrieval", {})
        out[cid] = {
            "domain": m.get("domain"),
            "volatility": m.get("volatility"),
            "freshness": m.get("freshness"),
            "source_config": m.get("source_config"),
            "n_records": cm.get("n_records", 0),
            "precision_at_5": retr.get("precision_at_5", {}).get("mean"),
            "ndcg_at_5": retr.get("ndcg_at_5", {}).get("mean"),
            "bertscore_f1": cm.get("bertscore_f1", {}).get("mean"),
            "hallucination_rate": cm.get("hallucination", {}).get("mean"),
            "human_eval_score": cm.get("human_eval_score", {}).get("mean"),
            "freshness_score": cm.get("freshness_score", {}).get("mean"),
            "source_diversity_index": cm.get("source_diversity_index", {}).get("mean"),
        }
    return {"conditions": out}


@app.get("/sources")
def sources():
    """Source-type impact per domain, computed from real condition metrics."""
    out = {}
    for domain in config.DOMAIN_LABELS:
        levels = source_levels_for_domain(domain)
        for lv in levels:
            lv["level_label"] = SOURCE_LEVEL_LABELS.get(
                SOURCE_LEVELS[lv["tier"]] if lv["tier"] < len(SOURCE_LEVELS) else "full",
                "Full Diversity",
            )
        scored = [lv for lv in levels if lv.get("precision_at_5") is not None]
        delta_pct = None
        best = worst = None
        if scored:
            best = max(scored, key=lambda x: x["precision_at_5"])
            worst = min(scored, key=lambda x: x["precision_at_5"])
            if worst["precision_at_5"] > 0:
                delta_pct = round(
                    100 * (best["precision_at_5"] - worst["precision_at_5"])
                    / worst["precision_at_5"], 1
                )
        out[domain] = {
            "label": config.DOMAIN_LABELS[domain],
            "levels": levels,
            "best_level": best["level_label"] if best else None,
            "worst_level": worst["level_label"] if worst else None,
            "delta_pct": delta_pct,
        }
    return {"sources": out}


@app.get("/freshness")
def freshness():
    """Model-swept freshness decay curves per domain (uses the RQ3 model)."""
    if predictor is None:
        raise HTTPException(500, "Predictor not loaded.")
    models = config.load_models()
    decay = models.get("decay_curves", {})

    ages = [3, 7, 14, 30, 60, 90, 180, 270, 365]
    curves = []
    for age in ages:
        point = {"age_days": age}
        for domain in config.DOMAIN_LABELS:
            sdi = _resolve_sdi(domain, "academic", None)
            pred = predictor.predict(domain, avg_age_days=float(age), source_diversity_index=sdi)
            point[domain] = pred["predictions"].get("precision_at_5", {}).get("expected")
        curves.append(point)

    # Best-fit decay info per domain for Precision@5 (from the trained model).
    sensitivity = {}
    p5_decay = decay.get("precision_at_5", {}).get("domains", {})
    for domain, fits in p5_decay.items():
        best_kind, best_r2 = None, -1.0
        for kind, info in fits.items():
            r2 = info.get("r_squared", 0)
            if r2 > best_r2:
                best_kind, best_r2 = kind, r2
        sensitivity[domain] = {"best_fit": best_kind, "r_squared": round(best_r2, 4)}

    return {"curves": curves, "ages": ages, "decay_fit": sensitivity}


@app.get("/validation")
def validation():
    data = config.load_validation()
    metrics_data = data.get("metrics", {})

    boundaries, loco = [], {}
    for metric, conds in metrics_data.items():
        loco[metric] = []
        for cid, cd in conds.items():
            loco[metric].append({
                "condition": cid,
                "domain": cd.get("domain"),
                "actual": cd.get("actual_mean"),
                "predicted": cd.get("predicted_mean"),
                "ci_lower": cd.get("ci_95_lower"),
                "ci_upper": cd.get("ci_95_upper"),
                "is_boundary": cd.get("is_boundary_condition"),
            })
            if cd.get("is_boundary_condition"):
                boundaries.append({
                    "condition": cid, "domain": cd.get("domain"), "metric": metric,
                    "actual_mean": cd.get("actual_mean"), "predicted_mean": cd.get("predicted_mean"),
                    "ci_lower": cd.get("ci_95_lower"), "ci_upper": cd.get("ci_95_upper"),
                })
    return {
        "n_samples": data.get("n_samples", 0),
        "boundary_conditions": boundaries,
        "boundary_counts_by_domain": data.get("boundary_condition_count_by_domain", {}),
        "loco_data": loco,
    }


@app.get("/model-info")
def model_info():
    models = config.load_models()
    ols = models.get("ols", {})
    rf = models.get("random_forest", {})
    xgb = models.get("xgboost", {})

    comparison = [{
        "metric": m,
        "ols_mae": ols.get(m, {}).get("cv_bounds", {}).get("mae_mean", 0),
        "rf_mae": rf.get(m, {}).get("cv_bounds", {}).get("mae_mean", 0),
        "xgb_mae": xgb.get(m, {}).get("cv_bounds", {}).get("mae_mean", 0),
        "ols_r2": ols.get(m, {}).get("r_squared", 0),
    } for m in ols]

    feature_importance = {}
    for name, data in (("random_forest", rf), ("xgboost", xgb)):
        for metric, d in data.items():
            fi = d.get("feature_importances")
            if fi:
                feature_importance.setdefault(metric, {})[name] = fi

    return {
        "n_samples": models.get("n_samples", 0),
        "n_features": models.get("n_features", 0),
        "comparison": comparison,
        "feature_importance": feature_importance,
    }


@app.get("/deployment-guidelines")
def deployment_guidelines():
    """Per-domain, evidence-based deployment guidance derived from data + model."""
    if predictor is None:
        raise HTTPException(500, "Predictor not loaded.")
    boundary_counts = config.load_validation().get("boundary_condition_count_by_domain", {})
    domains = {
        domain: domain_guideline(predictor, domain, boundary_counts)
        for domain in config.DOMAIN_LABELS
    }
    return {"domains": domains, "domain_labels": config.DOMAIN_LABELS}


@app.get("/health")
def health():
    return {
        "predictor_loaded": predictor is not None,
        "data_dir": str(config.DATA_DIR),
        "models_present": config.MODELS_PATH.exists(),
        "conditions_present": config.CONDITION_METRICS_PATH.exists(),
        "validation_present": config.VALIDATION_PATH.exists(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
