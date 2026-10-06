#!/usr/bin/env bash
# Usage: ./run_mac_linux.sh <image or folder> [more images...]   -> results in ./output
cd "$(dirname "$0")"
.venv/bin/python -m floorplan3d "$@" -o output
