"""Runs floorplan3d.pipeline.analyze() on a background thread so the window stays responsive."""
import io
import json
import os
import tempfile
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal, Slot


@dataclass
class Job:
    image: str
    out_dir: str
    scale_px_per_m: float = None   # None = automatic (dimension lines, then estimate)
    wall_height_m: float = 2.75
    trim_px: int = 0               # crop this many pixels off every edge (frames / borders)
    use_ocr: bool = True
    fast: bool = False


class _LineStream(io.TextIOBase):
    """stdout replacement that forwards each printed line to a callback."""

    def __init__(self, emit):
        self._emit, self._buf = emit, ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._emit(line.strip())
        return len(s)


def prepare_image(job):
    """Apply the border trim. Returns the path the pipeline should read; the trimmed copy
    keeps the original file name so the outputs are named after the user's file."""
    if job.trim_px <= 0:
        return job.image
    import cv2
    import numpy as np
    img = cv2.imdecode(np.fromfile(job.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read the image: {job.image}")
    t = job.trim_px
    h, w = img.shape[:2]
    if 2 * t >= min(h, w) - 50:
        raise ValueError(f"Border trim of {t} px is too large for a {w} x {h} px image.")
    stem = os.path.splitext(os.path.basename(job.image))[0]
    path = os.path.join(tempfile.mkdtemp(prefix="floorplan3d_"), stem + ".png")
    cv2.imencode(".png", img[t:h - t, t:w - t])[1].tofile(path)
    return path


class AnalysisWorker(QObject):
    log = Signal(str)
    finished = Signal(object)   # dict(outputs, data, image, prepared)
    failed = Signal(str)

    def __init__(self, job):
        super().__init__()
        self.job = job

    @Slot()
    def run(self):
        j = self.job
        try:
            self.log.emit("Preparing image…")
            path = prepare_image(j)
            self.log.emit("Loading models and analysing the plan…")
            from floorplan3d.pipeline import analyze
            with redirect_stdout(_LineStream(self.log.emit)):
                res = analyze(path, out_dir=j.out_dir, use_ocr=j.use_ocr, scale_px_per_m=j.scale_px_per_m,
                              wall_height_m=j.wall_height_m, make_3d=True, tta=not j.fast)
            outputs = dict(res.get("outputs", {}))
            with open(outputs["json_3d"], encoding="utf-8") as f:
                data = json.load(f)
            self.finished.emit(dict(outputs=outputs, data=data, image=j.image, prepared=path))
        except Exception as e:  # report every failure to the UI instead of killing the thread
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")
