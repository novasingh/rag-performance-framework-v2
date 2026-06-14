import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rq3_experiment.framework.predictor import RAGPerformancePredictor
from rq3_experiment.config import RESULTS_DIR_RQ3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RQ3 Predictive Framework API")

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the predictor
try:
    predictor = RAGPerformancePredictor(RESULTS_DIR_RQ3 / "rq3_models.json")
    models_data = json.loads((RESULTS_DIR_RQ3 / "rq3_models.json").read_text(encoding="utf-8"))
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    predictor = None
    models_data = {}


class PredictionRequest(BaseModel):
    domain: str
    avg_age_days: float
    source_diversity_index: float


@app.get("/")
def read_root():
    return {"status": "online", "message": "RQ3 Predictive Framework API"}


@app.post("/predict")
def predict_performance(req: PredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor models not loaded.")
    
    try:
        result = predictor.predict(
            domain=req.domain,
            avg_age_days=req.avg_age_days,
            source_diversity_index=req.source_diversity_index
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/models")
def get_models():
    """Serves the raw JSON for the frontend model viewer and charts."""
    if not models_data:
        raise HTTPException(status_code=500, detail="Models data not available.")
    return models_data
