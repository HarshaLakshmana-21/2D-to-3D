"""Application entry point: python -m floorplan3d_app"""
import os
import sys
import tempfile

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

# QtWebEngine must be imported before the QApplication exists. It is optional:
# without it the 3D view falls back to opening the viewer in the browser.
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
except Exception:  # missing system libraries on some Linux desktops
    QWebEngineView = None


def main(argv=None):
    argv = sys.argv if argv is None else argv
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(argv)
    app.setApplicationName("Floorplan3D")
    app.setOrganizationName("floorplan3d")
    app.setStyle("Fusion")  # same look on Windows, macOS and Linux

    from .theme import stylesheet, app_icon
    font = QFont(); font.setFamilies(["Segoe UI", "SF Pro Text", "Helvetica Neue", "Inter", "Ubuntu", "Noto Sans", "Arial"])
    font.setPointSizeF(10 if sys.platform == "win32" else 13 if sys.platform == "darwin" else 10.5)
    app.setFont(font)
    app.setStyleSheet(stylesheet(os.path.join(tempfile.gettempdir(), "floorplan3d_app_assets")))
    app.setWindowIcon(app_icon())

    from .main_window import MainWindow
    win = MainWindow(web_engine=QWebEngineView)
    win.show()
    # optional: an image path on the command line is loaded straight away
    if len(argv) > 1 and os.path.isfile(argv[1]):
        win.load_image(os.path.abspath(argv[1]))
    return app.exec()
