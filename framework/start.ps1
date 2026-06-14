# RAG Performance Prediction Framework - Standalone Launcher
# Runs entirely from the framework/ folder. No dependency on rq1/rq3/rq4 at runtime.

$ErrorActionPreference = "Stop"
$FrameworkDir = $PSScriptRoot
Set-Location $FrameworkDir

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  RAG Performance Prediction Framework" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Locate a Python interpreter (prefer the project venv, fall back to system python).
$python = "python"
if (Test-Path "$FrameworkDir\..\.venv\Scripts\python.exe") {
    $python = "$FrameworkDir\..\.venv\Scripts\python.exe"
}

# Ensure bundled data exists; build it if missing.
if (-not (Test-Path "$FrameworkDir\backend\data\rq3_models.json")) {
    Write-Host "[setup] Bundled data not found. Running sync_data.py..." -ForegroundColor Yellow
    & $python "$FrameworkDir\sync_data.py"
}

# Install backend dependencies.
Write-Host "[1/3] Installing backend dependencies..." -ForegroundColor Green
& $python -m pip install -r "$FrameworkDir\backend\requirements.txt" -q

# Start backend (working dir = backend so 'core' is importable).
Write-Host "[2/3] Starting backend on http://localhost:8000 ..." -ForegroundColor Green
$backend = Start-Process -FilePath $python `
    -ArgumentList "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload" `
    -WorkingDirectory "$FrameworkDir\backend" -NoNewWindow -PassThru
Start-Sleep -Seconds 3

# Install + start frontend.
$frontendDir = "$FrameworkDir\frontend"
if (-not (Test-Path "$frontendDir\node_modules")) {
    Write-Host "[setup] Installing frontend dependencies..." -ForegroundColor Yellow
    Push-Location $frontendDir; npm install --silent; Pop-Location
}
Write-Host "[3/3] Starting frontend on http://localhost:3000 ..." -ForegroundColor Green
# Launch npm via cmd.exe: Start-Process cannot execute npm.cmd/npx.cmd directly
# (that triggers "%1 is not a valid Win32 application").
$frontend = Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/c", "npm run dev" `
    -WorkingDirectory $frontendDir -NoNewWindow -PassThru

Write-Host ""
Write-Host "  Backend:  http://localhost:8000  (docs at /docs)" -ForegroundColor White
Write-Host "  Frontend: http://localhost:3000" -ForegroundColor White
Write-Host "  Press Ctrl+C to stop." -ForegroundColor Yellow

try { $backend.WaitForExit() } catch { Write-Host "Shutting down..." -ForegroundColor Yellow }
