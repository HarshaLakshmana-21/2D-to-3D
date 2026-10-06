#!/usr/bin/env bash
# Starts the Floorplan3D desktop app. Optional: pass an image to open it straight away.
cd "$(dirname "$0")"
.venv/bin/python -m floorplan3d_app "$@"
