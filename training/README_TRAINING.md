# Re-training the segmentation model

1. `git clone https://github.com/m-agour/ResPlan` and unzip `ResPlan.zip` (data: CC BY 4.0).
2. `pip install torch shapely onnx` (plus the runtime requirements).
3. Render training data (pixel-perfect labels, drawing styles similar to real plans):
   `RESPLAN_DIR=ResPlan python gen.py 0 1 5300` (val = first 300 images)
4. Train (CPU ok, GPU faster): `ITERS=5000 python train.py`
5. Export (the trained checkpoint `final.pt` is included): `python export_onnx.py final.pt ../floorplan3d/models/floorplan_seg.onnx`
6. Held-out test set & metrics: `python gen_test.py && python eval.py 120`

To specialise on your own drawings, add your images + label PNGs
(0 bg, 1 wall, 2 door, 3 window, 4 living, 5 kitchen, 6 bedroom, 7 bathroom,
8 balcony, 9 storage, 10 stair) to `data/img` / `data/lab` and fine-tune with `RESUME=1`.
