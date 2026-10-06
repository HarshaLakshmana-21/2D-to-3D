"""Optional: build a 'wheels' folder so the tool can be INSTALLED on a machine with
no internet at all. Run this on any internet-connected machine, then copy the whole
folder (incl. 'wheels') to the offline machine and run install_*.

  python make_offline_bundle.py --platform win_amd64 --python 3.11
  python make_offline_bundle.py --platform macosx_11_0_arm64 --python 3.11
  python make_offline_bundle.py --platform manylinux2014_x86_64 --python 3.11
"""
import argparse, subprocess, sys
ap = argparse.ArgumentParser()
ap.add_argument("--platform", required=True); ap.add_argument("--python", default="3.11")
a = ap.parse_args()
subprocess.check_call([sys.executable, "-m", "pip", "download", "-r", "requirements.txt", "-d", "wheels",
                       "--only-binary=:all:", "--platform", a.platform, "--python-version", a.python])
subprocess.check_call([sys.executable, "-m", "pip", "download", "pip", "setuptools", "-d", "wheels"])
print("wheels/ ready - copy the whole folder to the offline machine")
