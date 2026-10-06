@echo off
REM One-time setup (needs internet ONCE). After this everything runs offline.
REM Requires Python 3.9 - 3.12 (64-bit) from python.org
cd /d "%~dp0"
if exist wheels (
  py -3 -m venv .venv || python -m venv .venv
  .venv\Scripts\python -m pip install --no-index --find-links wheels -r requirements.txt -r requirements-app.txt
) else (
  py -3 -m venv .venv || python -m venv .venv
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install -r requirements.txt -r requirements-app.txt
)
echo.
echo Setup complete. Desktop app:  run_app_windows.bat
echo Command line:  run_windows.bat sample_plan.png
pause
