@echo off
REM MMS Fault Classification Dashboard Launcher for Windows

echo ==========================================
echo MMS Fault Classification Dashboard
echo ==========================================
echo.

REM Check if virtual environment exists
if exist venv\ (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Check dependencies
echo Checking dependencies...
python -c "import streamlit, plotly, sklearn, sktime" 2>nul

if errorlevel 1 (
    echo Missing dependencies. Installing...
    pip install -r requirements.txt
)

echo.
echo Launching dashboard...
echo Dashboard will open at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the dashboard
echo.

REM Launch dashboard
streamlit run dashboard/app.py
