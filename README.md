# 2D to 3D: offline floor-plan detection and 3D export

`floorplan3d` reads a 2D floor-plan image and detects **walls, doors, windows and rooms**, including
**room types** and **real-world measurements**. Each image produces two outputs:

1. **`<name>_detected.png`**: the 2D detection image. It shows the input with walls (exterior/interior),
   doors, windows and coloured rooms, each labelled with its name, type, dimensions and area. A side
   panel shows the scale calibration check.
2. **`<name>_3d.json`**: 3D-ready data for building 3D models: wall centre-lines with thickness and height,
   openings attached to walls (offset, width, height, sill), room polygons with type and area, and the scale.
   All values are in metres, with feet-inch strings included.

The JSON is also turned into ready-made 3D files:
`<name>_3d.glb` (Blender, three.js, Unity, Unreal, Windows 3D Viewer), `<name>_3d.obj`,
`<name>_3d_viewer.html` (double-click, opens offline in any browser) and `<name>_3d_preview.png`.

**Fully offline:** no internet connection or cloud API is used. It runs on **Windows, macOS and Linux**
(Python + ONNX Runtime, CPU only, no GPU needed).

---

## Quick start

### Windows
```
install_windows.bat              (one time)
run_windows.bat sample_plan.png  (or a folder of images)
```
### macOS / Linux
```
./install_mac_linux.sh
./run_mac_linux.sh sample_plan.png
```
Results are written to the `output/` folder.

Installation needs Python 3.9–3.12 and internet **once** to install the packages. To install on a
machine with no internet at all, run `python make_offline_bundle.py --platform win_amd64` (or
`macosx_11_0_arm64` / `manylinux2014_x86_64`) on any online PC. Then copy the folder, including the new
`wheels/` folder, to the offline machine and run the installer.

### Desktop app
A desktop app (PySide6 / Qt) is installed by the same installers and runs on Windows, macOS and Linux.
```
run_app_windows.bat              (Windows)
./run_app_mac_linux.sh           (macOS / Linux)
python -m floorplan3d_app        (any OS, from the activated .venv)
```
Drop a plan image on the window or click to browse. Check the settings and click **Analyse plan**.
The 2D detection image appears first. Use the **Output** drop-down to switch to the other views:

| Output | Shows |
|---|---|
| 2D Detection | Walls, doors, windows and coloured rooms with sizes (zoom with the wheel, drag to pan) |
| 3D Model · Interactive | The three.js viewer, embedded (drag to orbit, scroll to zoom) |
| 3D Model · Preview image | Still render of the 3D model |
| Rooms & Measurements / Walls / Doors & Windows | Sortable tables, exportable as CSV |
| Scale & Recognised Text | How the scale was found and every string the OCR read |
| 3D Data (JSON) | The full `floorplan3d/v1` JSON |
| Original Input | The image as loaded |

Settings in the sidebar:
* **Scale**: *Automatic* reads dimension lines. If the plan only has room-size labels, choose
  *Pixels per foot* (or *per metre*) and enter the pixel width of a known wall divided by its length.
  The app shows a warning when the scale had to be estimated.
* **Trim border**: crops frames and title blocks, which would otherwise be detected as walls.
* **Wall height**, **Read text** and **Fast mode** do the same as the command-line options below.

The toolbar buttons open the current output in its default app, save it elsewhere (3D: GLB, OBJ or
HTML), and open the output folder. If the embedded 3D viewer can't start (some Linux desktops lack
the libraries QtWebEngine needs), the app offers to open the viewer in the web browser instead.

### Options
```
python -m floorplan3d plan.png -o out           # default
  --scale-px-per-ft 17.6   | --scale-px-per-m 57.9   # force the scale (if no dimensions are printed)
  --wall-height 3.0        # metres (default 2.75)
  --no-ocr                 # skip text reading
  --no-3d                  # only PNG + JSON
  --fast                   # no test-time augmentation (about 3x faster)
```
Python API:
```python
from floorplan3d.pipeline import analyze
res = analyze("plan.png", out_dir="out")   # returns the same dict as the JSON
```

---

## How it works

| Stage | What it does |
|---|---|
| Segmentation | U-Net (2.2 M params, ONNX, 9 MB) trained from scratch, classifying each pixel as background, wall, door, window, living, kitchen, bedroom, bathroom, balcony, storage or stair. Flip test-time augmentation. When the scale is known, the image is resampled to the training resolution. |
| OCR | RapidOCR (PP-OCRv3 ONNX models bundled in the pip package, offline). Reads horizontal and vertical text and repairs common OCR errors (`18' 3*`, `15°10°`, `123*` → 12'3"). |
| Scale calibration | Each dimension string is matched to its dimension line. The arrow tips, ticks or junctions are located and the pixel length is measured. A consensus over all dimensions gives px/m per axis, and a self-check table lists the error of every annotation. |
| Vectorisation | Wall mask (refined with the actual ink) is broken into Manhattan wall segments with thickness. Segments are snapped at junctions and split at T-junctions. Gaps between collinear walls become door openings (for example bi-fold doors), and tiny breaks are joined. Clutter such as plants, furniture and arrowheads is removed. |
| Rooms | Free space enclosed by walls and openings. Open-plan areas carrying several room names are split along wall corners (for example living / hallway). Balconies are bounded by their drawn railing. |
| Room type | OCR label first (keywords for 15 room types, including living+kitchen, master bedroom, hallway, pooja, office, garage …). The model's room-class vote is the fallback when no label is printed. |
| Exterior / interior walls | Decided by whether a wall faces the outside region. |

### Results on your test image (`sample_plan.png`)
Scale: **17.65 px/ft**. All 9 readable dimension strings agree; measured vs annotated is within ±1.1%.

| Room | Type | Clear size | Area |
|---|---|---|---|
| Living Room and Kitchen | living_kitchen | 17'4" x 22'2" | 381.6 ft² (35.5 m²) |
| Bedroom | bedroom | 11'7" x 8'8" | 100.6 ft² |
| Bathroom | bathroom | 5'3" x 8'8" | 45.7 ft² |
| Hallway | hallway | 11'7" x 7'8" | 64.9 ft² |
| Balcony | balcony | 6'5" x 15'7" (inside railing; drawn 6'8" x 15'10") | 100.7 ft² |

Footprint: 36'9" x 23'4" (the drawn chain is 6'8" + 17'9" + 12'3" = 36'8").
Detected: 14 walls, 4 doors (bathroom 2'9", bedroom 2'9", entrance 2'11", balcony bi-fold 6'6") and 6 windows.

### Accuracy on held-out data
Measured on 60 plans from the ResPlan **test split**, which were never used for training
(`training/eval.py`, results in `training/eval_results.json`):

| Metric | Result |
|---|---|
| Wall pixel IoU | **0.913** |
| Door / window pixel IoU | 0.776 / 0.826 |
| Door detection recall, median width error | **94.4%**, **2.7%** |
| Window detection recall, median width error | **92.1%**, **1.4%** |
| Scale from dimension strings (43 plans that had them): median / p90 error | **0.41% / 0.92%** |
| Room recall (IoU > 0.5) | 77.8% |
| Room type accuracy (OCR label + model) | **91.9%** |
| Room area, median error (all plans, including those where the scale had to be estimated) | 5.5% |

Storage and stair have too few training examples to be learned (IoU 0). They are still recognised when
labelled in text. Inference takes about 4–12 s per image on a laptop CPU.

---

## JSON schema (`floorplan3d/v1`)
```jsonc
{
  "units": {"length": "m", "area": "m2"},
  "coordinate_system": {"origin": "top-left of footprint", "x": "right -> 3D +X", "y": "down -> 3D +Z", "up": "3D +Y"},
  "scale": {"px_per_m_x": 57.9, "px_per_m_y": 57.9, "method": "dimension_annotations",
            "annotations": [{"text": "18' 3\"", "annotated_m": 5.563, "measured_m": 5.54, "error_pct": -0.4}]},
  "defaults": {"wall_height_m": 2.75, "door_height_m": 2.1, "window_sill_m": 0.9, "window_head_m": 2.1},
  "building": {"width_m": 11.19, "depth_m": 7.12, "floor_area_m2": 55.1, "room_count": 5, "...": "..."},
  "walls": [{"id": "W1", "start": [x, y], "end": [x, y], "thickness_m": 0.14, "height_m": 2.75,
             "length_m": 5.52, "length_ft_in": "18' 1\"", "orientation": "horizontal",
             "type": "exterior", "openings": ["N1", "D2"]}],
  "openings": [{"id": "D1", "type": "door", "wall_id": "W4", "offset_from_wall_start_m": 0.81,
                "width_m": 0.84, "height_m": 2.1, "sill_height_m": 0.0, "center": [x, y],
                "connects": ["R3", "R4"], "inferred_from_wall_gap": false},
               {"id": "N1", "type": "window", "wall_id": "W8", "width_m": 1.6, "sill_height_m": 0.9, "height_m": 1.2}],
  "rooms": [{"id": "R2", "name": "Living Room and Kitchen", "type": "living_kitchen",
             "label_source": "ocr_label", "polygon": [[x, y], "..."], "area_m2": 35.46, "area_ft2": 381.6,
             "width_m": 5.28, "depth_m": 6.75, "perimeter_m": 24.1, "centroid": [x, y],
             "floor_elevation_m": 0.0, "ceiling_height_m": 2.75, "doors": ["D2"]}],
  "texts": ["all OCR text with positions"]
}
```
To build a 3D model, extrude each wall as a box (`length × height × thickness`) along its centre-line and
cut each opening at `offset_from_wall_start_m` (width × height, starting at `sill_height_m`). Then add
floor polygons for the rooms. `floorplan3d/export3d.py` is a complete reference implementation of this
(using trimesh).

---

## Training data and licensing
* Trained on **ResPlan** (17,000 real residential plans; **CC BY 4.0, commercial use allowed with
  attribution**). The vector plans were rendered into drawing-style images with pixel-perfect labels.
  The renderer is in `training/`.
* Non-commercial datasets (CubiCasa5K, FloorPlanCAD, RPLAN, Structured3D, ZInD) were **not used**.
  See `THIRD_PARTY_LICENSES.md` for details and the required attribution.
* Every runtime dependency (ONNX Runtime, RapidOCR, OpenCV, trimesh, three.js …) is MIT, BSD or Apache.

## Limitations and tips
* Best on clean architectural drawings, scans or exports where the image is at least about 800 px wide.
  Phone photos with perspective should be straightened first.
* If a plan has **no dimension text**, the scale is estimated from door widths and a warning is printed.
  For exact measurements, pass `--scale-px-per-ft`.
* Walls are vectorised as horizontal/vertical (Manhattan) segments. Curved or diagonal walls are not
  vectorised yet.
* Measured room sizes are **clear interior dimensions**, so they are smaller than outer-face dimension
  chains by the wall thickness.
* To improve on your own drawing style, label a few dozen of your plans and fine-tune (`training/README_TRAINING.md`).
