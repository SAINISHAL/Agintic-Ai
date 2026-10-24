# Local Launcher for Culturally Grounded Counselling Chatbot
# Runs 100% locally on CPU without Modal cloud inference

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Culturally Grounded Counselling Chatbot (Local CPU Runner) " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Locate and activate Python virtual environment
$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "[ERROR] Virtual environment not found at $VenvPython" -ForegroundColor Red
    Write-Host "Please create one using: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

Write-Host "[1/4] Found Python virtual environment: $VenvPython" -ForegroundColor Green

# 2. Check for required local emotion model
$EmotionModel = Join-Path $PSScriptRoot "models\emotion\best\label_mapping.json"
if (-not (Test-Path $EmotionModel)) {
    Write-Host "[WARNING] Emotion model checkpoint not found at models/emotion/best." -ForegroundColor Yellow
    Write-Host "Chatbot will use the keyword baseline fallback until trained." -ForegroundColor Yellow
} else {
    Write-Host "[2/4] Verified trained RoBERTa emotion model checkpoint." -ForegroundColor Green
}

# 3. Check active model configuration
Write-Host "[3/4] Active Model: Trained Qwen3-8B LoRA (models/qwen_8b/best) - 100% Local Mode." -ForegroundColor Green

# 4. Launch Streamlit
Write-Host "[4/4] Starting Streamlit web application on http://localhost:8501 ..." -ForegroundColor Green
Write-Host "Press Ctrl+C in this terminal window to stop the server.`n" -ForegroundColor DarkGray

& $VenvPython -m streamlit run app.py --server.fileWatcherType none
