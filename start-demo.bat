@echo off
setlocal
cd /d "%~dp0"

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

set HF_HUB_OFFLINE=1
set GRADIO_ANALYTICS_ENABLED=False
set PYTHONIOENCODING=utf-8

echo ============================================================
echo   AM DEFECT DEMO
echo   Browser will open:  http://127.0.0.1:7860
echo   Close this window to stop the server.
echo ============================================================
echo.

"%PY%" -m analysis.demo_app --offline

echo.
echo [stopped] If there was a red error above, screenshot this window.
pause
