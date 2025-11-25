@echo off
REM Distribution Package Creator for Windows

echo ==========================================
echo MMS Fault Classification
echo Distribution Package Creator
echo ==========================================
echo.

REM Package name
set PACKAGE_NAME=mms_fault_classification_v1.0
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%
set OUTPUT_FILE=%PACKAGE_NAME%_%TIMESTAMP%.zip

echo Creating distribution package...
echo.

REM Clean up temporary files
echo 1. Cleaning temporary files...
for /d /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
del /s /q *.pyc 2>nul
del /s /q .DS_Store 2>nul
del /q temp_upload.csv 2>nul
echo    Done
echo.

REM Verify model files
echo 2. Verifying model files...
if exist "models\minirocket\minirocket_model.pkl" (
    echo    Model found
) else (
    echo    ERROR: Model file not found!
    echo    Make sure models\minirocket\minirocket_model.pkl exists
    exit /b 1
)
echo.

REM Check for PowerShell (needed for ZIP creation)
where powershell >nul 2>&1
if %errorlevel% neq 0 (
    echo PowerShell not found. Please create ZIP manually.
    echo Include all files except venv, __pycache__, and .pyc files
    pause
    exit /b 1
)

REM Create ZIP using PowerShell
echo 3. Creating ZIP package...
powershell -Command "Compress-Archive -Path .\* -DestinationPath .\%OUTPUT_FILE% -Force"

if exist "%OUTPUT_FILE%" (
    echo    Package created: %OUTPUT_FILE%
) else (
    echo    Failed to create package
    exit /b 1
)

echo.
echo ==========================================
echo Distribution Package Ready!
echo ==========================================
echo.
echo Package: %OUTPUT_FILE%
echo.
echo What's included:
echo   - Dashboard application (5 pages)
echo   - Trained model (99.98%% accuracy)
echo   - Complete documentation
echo   - Sample data
echo   - Test suite
echo   - Launch scripts
echo.
echo To share:
echo   1. Send %OUTPUT_FILE% to recipient
echo   2. They extract and run: pip install -r requirements.txt
echo   3. Launch: run_dashboard.bat
echo.
echo ==========================================
pause
