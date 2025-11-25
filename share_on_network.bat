@echo off
REM Quick Network Sharing Script for Windows

echo ===========================================
echo MMS Fault Classification Dashboard
echo Network Sharing Mode
echo ===========================================
echo.

REM Get IP address
echo Finding your IP address...
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
    set IP=%%a
    goto :found
)

:found
REM Trim spaces
set IP=%IP: =%

if "%IP%"=="" (
    echo Warning: Could not auto-detect IP address
    echo Please find your IP manually: Control Panel -^> Network
    set /p IP="Enter your IP address: "
)

echo.
echo ===========================================
echo Your IP Address: %IP%
echo ===========================================
echo.
echo Others can access your dashboard at:
echo.
echo   http://%IP%:8501
echo.
echo Share this URL with people on the same WiFi network.
echo.
echo Important:
echo   - Keep this window open
echo   - Keep your computer running
echo   - Everyone must be on the same WiFi
echo.
echo Press Ctrl+C to stop sharing
echo.
echo ===========================================
echo Launching dashboard...
echo ===========================================
echo.

REM Launch Streamlit with network access
streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
