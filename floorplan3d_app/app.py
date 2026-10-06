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


def self_test(report_path):
    """`Floorplan3D --self-test report.json`: analyse the bundled sample plan and write a
    JSON report. Used to check a packaged build has every model and library it needs."""
    import json
    import shutil
    import time
    import traceback
    from .main_window import SAMPLE
    rep = dict(ok=False, frozen=bool(getattr(sys, "frozen", False)), platform=sys.platform,
               webengine=QWebEngineView is not None)
    out = tempfile.mkdtemp(prefix="floorplan3d_selftest_")
    t0 = time.time()
    try:
        from .worker import Job, prepare_image
        from floorplan3d.pipeline import analyze
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            res = analyze(prepare_image(Job(image=SAMPLE, out_dir=out)), out_dir=out, make_3d=True)
        b = res["building"]
        rep.update(rooms=b["room_count"], walls=b["wall_count"], doors=b["door_count"], windows=b["window_count"],
                   scale_method=res["scale"]["method"], outputs=sorted(os.path.basename(p) for p in res["outputs"].values()),
                   ocr_texts=len(res["texts"]))
        rep["ok"] = b["room_count"] > 0 and len(res["outputs"]) >= 6 and len(res["texts"]) > 0
    except Exception:
        rep["error"] = traceback.format_exc()
    rep["seconds"] = round(time.time() - t0, 1)
    shutil.rmtree(out, ignore_errors=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    return 0 if rep["ok"] else 1


def main(argv=None):
    argv = sys.argv if argv is None else argv
    if "--self-test" in argv:
        i = argv.index("--self-test")
        return self_test(argv[i + 1] if len(argv) > i + 1 else os.path.join(tempfile.gettempdir(), "floorplan3d_selftest.json"))
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
