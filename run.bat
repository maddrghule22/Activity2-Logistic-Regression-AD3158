@echo off
echo ========================================================
echo Smart Student Performance & Placement Prediction System
echo Logistic Regression - Supervised Machine Learning
echo ========================================================
echo.

IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found. Setting up environment...
    py -m venv .venv
    echo [INFO] Installing required dependencies...
    .venv\Scripts\pip install -r requirements.txt
)

echo [INFO] Activating virtual environment...
call .venv\Scripts\activate

echo [INFO] Launching Streamlit ML Dashboard...
streamlit run app.py

pause
