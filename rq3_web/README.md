# RAG Performance Predictive Framework

A sophisticated web application for predicting and analyzing the performance of Retrieval Augmented Generation (RAG) systems across different domains. This framework combines machine learning predictions with an interactive visualization interface.

## 📋 Project Overview

This application allows users to:
- Predict RAG system performance metrics (nDCG@5, Precision@5, Hallucination Rate, BERTScore) 
- Analyze how different parameters affect performance
- Visualize performance patterns across domains
- Save and export prediction scenarios for analysis

## 🏗️ Architecture

### Backend
- **Framework**: FastAPI (Python)
- **Purpose**: Provides RESTful APIs for predictions and model data
- **Key Features**:
  - RAG Performance Prediction endpoint
  - Models data serving
  - CORS enabled for React frontend

### Frontend
- **Framework**: React 19.2 + Vite 8.0
- **Visualization**: Recharts for interactive charts
- **UI Components**: Lucide React icons
- **API Client**: Axios
- **Features**:
  - Interactive prediction parameters
  - Real-time performance charts
  - Scenario management
  - CSV export functionality

## 📦 Prerequisites

### Backend Requirements
- Python 3.8+
- Dependencies listed in `backend/requirements.txt`:
  - FastAPI
  - Uvicorn
  - Pydantic
  - Python-multipart
- The parent project `rq3_experiment` must be accessible in PYTHONPATH

### Frontend Requirements
- Node.js 16+
- npm or yarn

### Quick Setup Checklist
- [ ] Python 3.8+ installed
- [ ] Node.js 16+ installed
- [ ] Parent project (rag-performance-framework-v2) properly set up
- [ ] Python virtual environment (optional but recommended)

## 🚀 Quick Start

### Option 1: Start Everything at Once (Easiest)

**Windows PowerShell:**
```powershell
# From the rq3_web directory
.\start.ps1
```

This will automatically:
- ✅ Start the FastAPI backend on `http://localhost:8000`
- ✅ Install frontend dependencies (if needed)
- ✅ Start the Vite dev server on `http://localhost:5173`

Then open your browser to `http://localhost:5173` and start using the application!

---

### Option 2: Start Backend and Frontend Separately

#### Step 1: Start the Backend

```bash
# Navigate to the backend directory
cd backend

# Install Python dependencies (if needed)
pip install -r requirements.txt

# Run the FastAPI server
python main.py
# OR with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend will start on `http://localhost:8000`

**API Endpoints:**
- `GET /` - Health check
- `GET /models` - Retrieve all model data
- `POST /predict` - Make a prediction (requires domain, avg_age_days, source_diversity_index)

#### Step 2: Start the Frontend (in a new terminal)

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at `http://localhost:5173` (or the port shown in terminal)

---

### Option 3: Development Servers with Live Reload

```bash
# Terminal 1: Backend with auto-reload
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend with hot reload
cd frontend
npm run dev
```

Any code changes will automatically reload in both backend and frontend!

## 📊 Using the Application

### Making Predictions

1. **Select a Domain**: Choose from available knowledge domains (e.g., technology, history, science)
2. **Set Average Age**: Adjust the average age of documents in days (0-365)
3. **Set Source Diversity**: Set the diversity index of sources (0-1)
4. **View Results**: The application automatically shows predictions for:
   - nDCG@5 (Ranking quality metric)
   - Precision@5 (Accuracy of top 5 results)
   - Hallucination Rate (Likelihood of generating incorrect information)
   - BERTScore F1 (Semantic similarity score)

### Saving Scenarios

- Click "Save Scenario" to store current predictions
- Compare multiple scenarios side-by-side
- Export scenarios to CSV for further analysis

### Boundary Conditions

The application flags potential boundary conditions (e.g., history domain with high diversity) that may affect prediction reliability.

## 🔧 Configuration

### Backend Configuration
Edit `backend/main.py` to:
- Adjust CORS settings (modify `allow_origins`)
- Change predictor model paths
- Customize logging

### Frontend Configuration
Edit `frontend/vite.config.js` to:
- Change the API base URL if backend runs on different port
- Modify build settings

## 📈 API Documentation

### POST /predict

**Request:**
```json
{
  "domain": "technology",
  "avg_age_days": 14,
  "source_diversity_index": 0.35
}
```

**Response:**
```json
{
  "predictions": {
    "ndcg_at_5": {
      "expected": 0.75,
      "confidence_interval": [0.72, 0.78]
    },
    "precision_at_5": {
      "expected": 0.80,
      "confidence_interval": [0.77, 0.83]
    },
    "hallucination_rate": {
      "expected": 0.12,
      "confidence_interval": [0.10, 0.14]
    },
    "bertscore_f1": {
      "expected": 0.85,
      "confidence_interval": [0.82, 0.88]
    }
  }
}
```

## 🛠️ Development

### Available Scripts

**Frontend:**
```bash
npm run dev       # Start dev server with hot reload
npm run build     # Build for production
npm run lint      # Run ESLint
npm run preview   # Preview production build
```

**Backend:**
```bash
# From backend directory:
python main.py                                          # Run normally
uvicorn main:app --reload --host 0.0.0.0 --port 8000  # Run with auto-reload
```

### Running with start.ps1 (Windows)

The `start.ps1` script simplifies launching the entire application:

```powershell
# From rq3_web root directory
.\start.ps1
```

**What it does:**
1. Sets Python path for rq3_experiment module access
2. Starts FastAPI backend in a background process
3. Installs frontend dependencies
4. Starts Vite dev server in the foreground

**To stop the application:**
- Press `Ctrl+C` to stop the frontend
- Close the backend window or run `taskkill /F /IM python.exe` in PowerShell

## 📝 Project Structure

```
rq3_web/
├── backend/
│   ├── main.py                 # FastAPI application
│   └── requirements.txt         # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # Main React component
│   │   ├── index.css           # Global styles
│   │   ├── main.jsx            # Entry point
│   │   └── assets/             # Static assets
│   ├── public/                 # Public assets
│   ├── package.json            # NPM dependencies
│   ├── vite.config.js          # Vite configuration
│   └── eslint.config.js        # ESLint configuration
├── start.ps1                   # PowerShell startup script
└── README.md                   # This file
```

## 🐛 Troubleshooting

### start.ps1 Execution Policy Error

If you get: `"start.ps1 cannot be loaded because running scripts is disabled on this system"`

**Solution:** Run PowerShell as Administrator and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try running `.\start.ps1` again.

### Backend Not Loading Models
- Check that `rq3_models.json` exists at the configured path
- Verify the RAG Framework (`rq3_experiment`) is properly installed
- Check backend logs for detailed error messages
- Ensure PYTHONPATH is set correctly (it's handled by start.ps1)

### Frontend Can't Connect to Backend
- Ensure backend is running on `http://localhost:8000`
- Check browser console for CORS errors
- Verify firewall isn't blocking connections
- Try accessing `http://localhost:8000` directly to verify backend is responding

### Port Already in Use
**For Backend (port 8000):**
```powershell
# Find and kill process on port 8000
Get-NetTCPConnection -LocalPort 8000
taskkill /PID <PID> /F
```

**For Frontend (port 5173):**
```bash
# Vite will automatically use next available port
npm run dev -- --host --port 5174
```

### Module Import Errors in Backend
- Ensure the parent project directory is in PYTHONPATH
- Verify `rq3_experiment` package is installed
- Check that you're running from the correct working directory
- The start.ps1 script automatically handles PYTHONPATH setup

### npm Dependencies Issues
```bash
# Clear npm cache and reinstall
cd frontend
npm cache clean --force
rm -r node_modules package-lock.json
npm install
```

## 📋 Performance Metrics Explanation

- **nDCG@5**: Normalized Discounted Cumulative Gain at rank 5 (0-1, higher is better)
- **Precision@5**: Fraction of top 5 results that are relevant (0-1, higher is better)
- **Hallucination Rate**: Probability model generates false information (0-1, lower is better)
- **BERTScore F1**: Semantic similarity between generated and reference text (0-1, higher is better)

---

**Last Updated:** August 2026
