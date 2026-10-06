# PyInstaller spec for the Floorplan3D desktop app. Build on the OS you are targeting
# (PyInstaller cannot cross-compile):
#   Windows       packaging\build_windows.bat      -> dist\Floorplan3D\Floorplan3D.exe
#   macOS         ./packaging/build_mac_linux.sh   -> dist/Floorplan3D.app
#   Linux         ./packaging/build_mac_linux.sh   -> dist/Floorplan3D/Floorplan3D
# One-folder build: the app starts much faster than a one-file build, which would
# unpack ~500 MB (Qt WebEngine, ONNX Runtime, OpenCV) to a temp folder on every launch.
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, get_package_paths

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
# RapidOCR appends its own folder to sys.path and imports ch_ppocr_* by name at runtime,
# so PyInstaller has to bundle those sub-packages as top-level modules.
RAPIDOCR_DIR = get_package_paths("rapidocr_onnxruntime")[1]
sys.path.insert(0, RAPIDOCR_DIR)
RAPIDOCR_MODULES = [m for p in ("ch_ppocr_v3_det", "ch_ppocr_v3_rec", "ch_ppocr_v2_cls") for m in collect_submodules(p)]
ICONS = os.path.join(SPECPATH, "build_icons")
NAME = "Floorplan3D"

datas = [
    (os.path.join(ROOT, "floorplan3d", "models"), os.path.join("floorplan3d", "models")),
    (os.path.join(ROOT, "floorplan3d", "assets"), os.path.join("floorplan3d", "assets")),
    (os.path.join(ROOT, "sample_plan.png"), "."),
]
datas += collect_data_files("rapidocr_onnxruntime")  # OCR models + config.yaml
datas += collect_data_files("trimesh", includes=["resources/**"])

a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],
    pathex=[ROOT, RAPIDOCR_DIR],
    datas=datas,
    hiddenimports=["floorplan3d.visualize", "floorplan3d.export3d", "PySide6.QtWebEngineWidgets", "PySide6.QtSvg"] + RAPIDOCR_MODULES,
    excludes=["tkinter", "IPython", "pytest", "torch", "torchvision", "notebook", "jupyter"],
    noarchive=False,
)
pyz = PYZ(a.pure)

if sys.platform == "win32":
    icon = os.path.join(ICONS, "app.ico")
elif sys.platform == "darwin":
    icon = os.path.join(ICONS, "app.icns")
else:
    icon = None

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name=NAME,
    console=False,          # GUI app: no terminal window
    icon=icon,
    upx=False,              # UPX breaks Qt / onnxruntime DLLs
    argv_emulation=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name=NAME, upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{NAME}.app",
        icon=icon,
        bundle_identifier="ai.epiplex.floorplan3d",
        info_plist={
            "CFBundleShortVersionString": "0.1.0",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
            "CFBundleDocumentTypes": [{"CFBundleTypeName": "Image", "CFBundleTypeRole": "Viewer",
                                       "LSItemContentTypes": ["public.image"]}],
        },
    )
