# Start the RQ3 Predictive Framework Web Dashboard

# Start FastAPI Backend in background
$env:PYTHONPATH="f:\rs2"
Start-Process -NoNewWindow -FilePath "..\.venv\Scripts\python.exe" -ArgumentList "-m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

# Start Vite Frontend
cd frontend
npm run dev
