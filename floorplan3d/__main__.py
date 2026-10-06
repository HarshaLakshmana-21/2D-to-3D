import argparse, glob, json, os, sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="floorplan3d", description="Detect walls, doors, windows and rooms with measurements in a 2D floor plan image and export 3D-ready data (fully offline).")
    ap.add_argument("inputs", nargs="+", help="image file(s) or folder(s) (png/jpg/bmp/tif)")
    ap.add_argument("-o", "--out", default="output", help="output folder (default: ./output)")
    ap.add_argument("--scale-px-per-m", type=float, default=None, help="override scale (pixels per metre)")
    ap.add_argument("--scale-px-per-ft", type=float, default=None, help="override scale (pixels per foot)")
    ap.add_argument("--wall-height", type=float, default=None, help="wall height in metres (default 2.75)")
    ap.add_argument("--no-ocr", action="store_true", help="disable OCR (room names / dimension calibration)")
    ap.add_argument("--no-3d", action="store_true", help="skip GLB/OBJ/preview generation (JSON is always written)")
    ap.add_argument("--fast", action="store_true", help="disable test-time augmentation (3x faster)")
    ap.add_argument("--model", default=None, help="path to a custom ONNX model")
    a = ap.parse_args(argv)
    from .pipeline import analyze
    files = []
    for p in a.inputs:
        if os.path.isdir(p):
            for ext in ("png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"):
                files += glob.glob(os.path.join(p, f"*.{ext}")) + glob.glob(os.path.join(p, f"*.{ext.upper()}"))
        else:
            files.append(p)
    if not files:
        print("no input images found"); return 1
    scale = a.scale_px_per_m or (a.scale_px_per_ft / 0.3048 if a.scale_px_per_ft else None)
    failed = 0
    for f in sorted(set(files)):
        try:
            r = analyze(f, out_dir=a.out, model_path=a.model, use_ocr=not a.no_ocr, scale_px_per_m=scale,
                        wall_height_m=a.wall_height, make_3d=not a.no_3d, tta=not a.fast)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[floorplan3d] FAILED on {f}: {e}"); failed += 1; continue
        if "warning" in r:
            print("   WARNING:", r["warning"])
        for k, v in r.get("outputs", {}).items():
            print(f"   {k:15s} {v}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
