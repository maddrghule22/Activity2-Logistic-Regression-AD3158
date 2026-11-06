#!/bin/bash
echo "========================================================"
echo "Smart Student Performance & Placement Prediction System"
echo "Logistic Regression - Supervised Machine Learning"
echo "========================================================"

if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv .venv
    echo "[INFO] Installing dependencies..."
    .venv/bin/pip install -r requirements.txt
fi

echo "[INFO] Activating virtual environment..."
source .venv/bin/activate

echo "[INFO] Starting Streamlit ML Dashboard..."
streamlit run app.py
