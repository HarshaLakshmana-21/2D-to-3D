@echo off
REM Usage: run_windows.bat <image or folder> [more images...]   -> results in .\output
cd /d "%~dp0"
.venv\Scripts\python -m floorplan3d %* -o output
echo Results saved in "%~dp0output"
