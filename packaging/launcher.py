"""Entry point PyInstaller freezes into the Floorplan3D executable."""
import os
import sys

# A windowed build has no console: stdout/stderr are None, and any library that prints
# or warns would raise. Send them to the null device instead.
for _name in ("stdout", "stderr"):
    if getattr(sys, _name) is None:
        setattr(sys, _name, open(os.devnull, "w", encoding="utf-8"))

from floorplan3d_app.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
