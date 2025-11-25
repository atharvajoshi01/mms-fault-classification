#!/bin/bash

# MMS Fault Classification Dashboard Launcher

echo "=========================================="
echo "MMS Fault Classification Dashboard"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check dependencies
echo "Checking dependencies..."
python -c "import streamlit, plotly, sklearn, sktime" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "Missing dependencies. Installing..."
    pip install -r requirements.txt
fi

echo ""
echo "Launching dashboard..."
echo "Dashboard will open at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

# Launch dashboard
streamlit run dashboard/app.py
