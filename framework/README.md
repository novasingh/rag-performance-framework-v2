# RAG Performance Prediction Framework

A standalone decision-support tool built on the dissertation *"A Framework for
Predicting Retrieval-Augmented Generation System Performance Based on Relevant
Factors."* It turns the study's experiment results and trained model into an
interactive web app that estimates RAG performance **before deployment** and
recommends a knowledge-base configuration.

It answers the practical question the dissertation raises: *given a domain, how
fresh does my knowledge base need to be and what mix of sources should I use?*

---

## What it does

Pick three dataset-level factors and get a prediction plus guidance:

| Factor | Options |
|---|---|
| **Domain volatility** | Technology (high), Healthcare (medium), History (low) |
| **Dataset freshness** | < 1 week, 1 week–1 month, 1–6 months, > 6 months |
| **Source-type mix** | Academic only, Academic + News, Full diversity |

For each configuration the app returns:
- Predicted **Precision@5, nDCG@5, BERTScore, hallucination rate, human-eval score**, each with a cross-validated error band (from the RQ3 model).
- A **deployment-readiness** score, recommended **update cadence**, recommended **source mix**, **risks**, and an expected **performance range** — all computed from the data and the model.

It also exposes the underlying evidence: per-condition metrics (RQ1), source/freshness
impact, model comparison (RQ3), and validation / boundary conditions (RQ4).

---

## How it works (standalone)

Everything needed to run lives under `framework/`. The app does **not** import the
`rq1/rq3/rq4` packages at runtime — instead, artifacts are bundled into
`backend/data/`:

```
framework/
├── backend/
│   ├── main.py            # FastAPI app
│   ├── core/
│   │   ├── config.py      # volatility scores, metric metadata, data loaders
│   │   ├── predictor.py   # the RQ3 predictive model (self-contained)
│   │   └── recommender.py # recommendations derived from data + model
│   └── data/              # bundled artifacts (the standalone data layer)
├── frontend/
│   └── src/
│       ├── api/client.js          # API utility layer
│       ├── hooks/useApi.js        # data-fetching hook
│       ├── context/MetaContext.jsx
│       ├── components/            # Sidebar, Callout, cards, charts, UI states
│       └── pages/                 # Info (landing), Predict, Deployment, + evidence views
├── sync_data.py           # refresh bundled data from the experiments
└── start.ps1              # launch backend + frontend
```

**No hardcoded numbers.** Every figure the UI shows is fetched from the API, and
the API computes it from `backend/data/` (the bundled RQ1 metrics, the trained
RQ3 model, RQ4 validation) — not from hand-typed constants.

---

## Running it

Pick the launcher for your OS — each one bundles data (if needed), installs
dependencies, then starts the backend (port 8000) and frontend (port 3000).

### Windows
```bat
cd framework
run-windows.bat          REM double-click also works
```
(or directly: `powershell -ExecutionPolicy Bypass -File start.ps1`)

### Linux
```bash
cd framework
chmod +x run-linux.sh
./run-linux.sh
```

### macOS
```bash
cd framework
chmod +x run-macos.sh
./run-macos.sh
```

- Backend → http://localhost:8000 (interactive docs at `/docs`)
- Frontend → http://localhost:3000

### Manual
```powershell
# 1. (Re)bundle data from the experiments — only needed once or after re-running them
python sync_data.py

# 2. Backend
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --port 8000

# 3. Frontend (new terminal)
cd frontend
npm install
npm run dev
```

### Refreshing results
After re-running an experiment, re-bundle and the app picks up the new numbers:
```powershell
python sync_data.py
```

---

## API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /summary` | Headline stats (observations, κ, conditions, boundary counts). |
| `POST /predict` | Model prediction + derived recommendations for one configuration. |
| `GET /conditions` | All 12 experimental conditions with measured metrics. |
| `GET /sources` | Source-type impact per domain (computed deltas). |
| `GET /freshness` | Model-swept freshness decay curves per domain. |
| `GET /validation` | RQ4 actual-vs-predicted and boundary conditions. |
| `GET /model-info` | OLS / Random Forest / XGBoost comparison + feature importances. |
| `GET /deployment-guidelines` | Per-domain update cadence, source mix, and reliability (derived). |
| `GET /metrics-meta` | Metric labels, scales, and directions (drives the UI). |
| `GET /health` | Data/predictor load status. |

---

## Tech stack
- **Backend:** FastAPI, Python standard library only (the predictor is pure Python).
- **Frontend:** React 19, Vite 6, Recharts.

## Scope note
Predictions are bounded by the study's design: three domains, two manipulated
factors (freshness, source type), and a fixed retrieval/generation pipeline.
Configurations outside the trained range (e.g. very high source diversity) are
flagged with a reliability warning, and RQ4 boundary conditions mark where
estimates are less trustworthy.
