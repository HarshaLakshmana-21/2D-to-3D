# Data, models and third-party components

All components are licensed for commercial use. Keep this file with any redistribution.

## Training data
**ResPlan: A Large-Scale Vector-Graph Dataset of 17,000 Residential Floor Plans**
https://github.com/m-agour/ResPlan
Data license: **Creative Commons Attribution 4.0 (CC BY 4.0)**. Commercial use is allowed with attribution.
Code license: MIT.
Attribution: "Segmentation model trained on renderings derived from the ResPlan dataset
(The ResPlan Authors, 2025), licensed under CC BY 4.0."
Changes made: the vector plans were rendered into synthetic raster drawings with procedurally
added styles (wall fills, door swings, windows, furniture, text and dimension lines) to produce
training images and pixel labels.

### Datasets reviewed and rejected (no commercial use)
| Dataset | License | Reason |
|---|---|---|
| CubiCasa5K | CC BY-NC 4.0 | non-commercial |
| FloorPlanCAD | CC BY-NC 4.0 | non-commercial |
| RPLAN | research-only agreement | non-commercial |
| Structured3D, Zillow ZInD | research / non-commercial terms | non-commercial |
| ROBIN | GPL-3.0 | copyleft obligations on derived works |
| R2V / LIFULL HOME'S | research-only | non-commercial |

## Runtime components
| Component | License |
|---|---|
| Trained model `floorplan3d/models/floorplan_seg.onnx` | yours (trained from scratch, no pretrained weights) |
| ONNX Runtime | MIT |
| RapidOCR (`rapidocr_onnxruntime`), bundles PaddleOCR PP-OCRv3 models | Apache-2.0 |
| OpenCV | Apache-2.0 |
| NumPy, Shapely | BSD |
| trimesh | MIT |
| Pillow | MIT-CMU (HPND) |
| Matplotlib | PSF-based (Matplotlib License) |
| three.js r147 (embedded in the HTML viewer) | MIT, see `floorplan3d/assets/three/LICENSE` |
| DejaVu fonts (`floorplan3d/assets`) | Bitstream Vera / DejaVu license (free, commercial OK) |
