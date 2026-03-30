@echo off
REM Model Training Script
cd /d "%~dp0model_pipeline" || exit /b 1

echo Model Training Options:
echo [1] Complete Workflow (Recommended)
echo [2] Train Only
echo [3] Evaluate Model
echo [4] Analyze Dataset
echo.
set /p choice="Select (1-4): "

if "%choice%"=="1" python run_complete_training.py
if "%choice%"=="2" python train_enhanced.py
if "%choice%"=="3" python evaluate.py
if "%choice%"=="4" python data_loader.py

echo.
echo Done. Check logs/ for results.
pause
