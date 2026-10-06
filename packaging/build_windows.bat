@echo off
REM Builds dist\Floorplan3D\Floorplan3D.exe (and a zip of the folder). Run install_windows.bat first.
cd /d "%~dp0\.."
.venv\Scripts\python -m pip install -r requirements-build.txt || exit /b 1
.venv\Scripts\python packaging\make_icons.py || exit /b 1
.venv\Scripts\pyinstaller --noconfirm --clean --distpath dist --workpath build packaging\floorplan3d.spec || exit /b 1
powershell -NoProfile -Command "Compress-Archive -Force -Path dist\Floorplan3D -DestinationPath dist\Floorplan3D-windows-x64.zip"
echo.
echo Built: dist\Floorplan3D\Floorplan3D.exe
echo Share: dist\Floorplan3D-windows-x64.zip  (unzip, then double-click Floorplan3D.exe)
