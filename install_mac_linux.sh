#!/usr/bin/env bash
# One-time setup (needs internet ONCE). After this everything runs offline.
# Requires Python 3.9 - 3.12
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
if [ -d wheels ]; then
  .venv/bin/python -m pip install --no-index --find-links wheels -r requirements.txt
else
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi
echo "Setup complete. Run: ./run_mac_linux.sh sample_plan.png"
