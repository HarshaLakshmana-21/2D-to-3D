@echo off
REM Starts the Floorplan3D desktop app. Optional: pass an image to open it straight away.
cd /d "%~dp0"
start "" .venv\Scripts\pythonw -m floorplan3d_app %*
