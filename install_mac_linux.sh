#!/usr/bin/env bash
# One-time setup (needs internet ONCE). After this everything runs offline.
# Requires Python 3.9 - 3.12
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
if [ -d wheels ]; then
  .venv/bin/python -m pip install --no-index --find-links wheels -r requirements.txt -r requirements-app.txt
else
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt -r requirements-app.txt
fi
echo "Setup complete. Desktop app: ./run_app_mac_linux.sh   Command line: ./run_mac_linux.sh sample_plan.png"
