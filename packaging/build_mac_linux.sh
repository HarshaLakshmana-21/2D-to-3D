#!/usr/bin/env bash
# Builds the desktop app for the current OS. Run ./install_mac_linux.sh first.
#   macOS -> dist/Floorplan3D.app   (+ dist/Floorplan3D-macos-<arch>.zip)
#   Linux -> dist/Floorplan3D/Floorplan3D   (+ dist/Floorplan3D-linux-x64.tar.gz)
set -e
cd "$(dirname "$0")/.."
.venv/bin/python -m pip install -r requirements-build.txt
.venv/bin/python packaging/make_icons.py
.venv/bin/pyinstaller --noconfirm --clean --distpath dist --workpath build packaging/floorplan3d.spec
if [ "$(uname)" = "Darwin" ]; then
  # ditto keeps the .app bundle's symlinks and permissions intact
  ditto -c -k --keepParent dist/Floorplan3D.app "dist/Floorplan3D-macos-$(uname -m).zip"
  echo "Built: dist/Floorplan3D.app (double-click to launch)"
else
  cp packaging/floorplan3d.desktop dist/Floorplan3D/
  cp packaging/build_icons/app.png dist/Floorplan3D/floorplan3d.png
  tar -C dist -czf dist/Floorplan3D-linux-x64.tar.gz Floorplan3D
  echo "Built: dist/Floorplan3D/Floorplan3D (double-click or run it)"
fi
