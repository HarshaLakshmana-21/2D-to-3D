"""Render the app icon (floorplan3d_app.theme.app_icon) to app.png, app.ico and app.icns
in packaging/build_icons/. Run before PyInstaller; the build scripts do this for you."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from PIL import Image  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402

app = QGuiApplication(sys.argv)
from floorplan3d_app.theme import app_icon  # noqa: E402

out = os.path.join(HERE, "build_icons")
os.makedirs(out, exist_ok=True)
png = os.path.join(out, "app.png")
app_icon().pixmap(256, 256).toImage().save(png)
img = Image.open(png).convert("RGBA")
img.save(os.path.join(out, "app.ico"), sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
img.save(os.path.join(out, "app.icns"))
print("icons written to", out)
