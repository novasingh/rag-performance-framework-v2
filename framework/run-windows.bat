@echo off
REM ===========================================================================
REM  RAG Performance Prediction Framework - Windows launcher
REM  Double-click this file, or run it from a terminal.
REM  It runs start.ps1 with the execution policy bypassed so no extra setup
REM  is needed.
REM ===========================================================================
echo Starting RAG Performance Prediction Framework (Windows)...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
pause
