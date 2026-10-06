"""Main window: sidebar (input + settings) on the left, results with an output-type selector on the right."""
import csv
import html
import json
import os
import shutil
import time

from PySide6.QtCore import QSettings, QSize, Qt, QThread, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
                               QGridLayout, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox, QStackedWidget,
                               QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget)

from floorplan3d.visualize import ROOM_COLORS

from .theme import C, app_icon, icon
from .widgets import Banner, DropZone, ImageView, StatCard, is_image
from .worker import AnalysisWorker, Job

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(PROJECT_DIR, "sample_plan.png")
FT = 0.3048

# key, label shown in the output selector, page kind
VIEWS = [
    ("detected", "2D Detection", "image"),
    ("viewer", "3D Model · Interactive", "web"),
    ("preview", "3D Model · Preview image", "image"),
    ("rooms", "Rooms & Measurements", "table"),
    ("walls", "Walls", "table"),
    ("openings", "Doors & Windows", "table"),
    ("scale", "Scale & Recognised Text", "html"),
    ("json", "3D Data (JSON)", "text"),
    ("input", "Original Input", "image"),
]
KEYS = [k for k, _, _ in VIEWS]


def _num_item(v, fmt="{:.2f}"):
    it = QTableWidgetItem()
    it.setData(Qt.DisplayRole, float(v))  # numeric sort
    it.setText(fmt.format(v))
    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return it


def _swatch(rgb):
    pm = QPixmap(12, 12); pm.fill(QColor(*rgb))
    return QIcon(pm)


class MainWindow(QMainWindow):
    def __init__(self, web_engine=None):
        super().__init__()
        self.settings = QSettings("floorplan3d", "Floorplan3D Desktop")
        self.web_engine = web_engine          # QWebEngineView class, or None when unavailable
        self.image_path = None
        self.result = None
        self._thread = None
        self._started = 0.0
        self.setWindowTitle("Floorplan3D")
        self.setWindowIcon(app_icon())
        self.setAcceptDrops(True)
        self.resize(1440, 900)
        self.setMinimumSize(1080, 680)

        root = QWidget(); root.setObjectName("root"); self.setCentralWidget(root)
        lay = QHBoxLayout(root); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        lay.addWidget(self._build_sidebar())
        lay.addWidget(self._build_main(), 1)

        self.status = QLabel("Ready · runs fully offline on this computer")
        self.statusBar().addWidget(self.status, 1)
        self.statusBar().setSizeGripEnabled(False)

        self._timer = QTimer(self, interval=250, timeout=self._tick)
        self._load_settings()
        self.view_select.currentIndexChanged.connect(self.show_view)
        self._sync_controls()

    # ------------------------------------------------------------------ layout
    def _build_sidebar(self):
        side = QFrame(); side.setObjectName("sidebar"); side.setFixedWidth(330)
        outer = QVBoxLayout(side); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)

        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
        body = QWidget(); v = QVBoxLayout(body); v.setContentsMargins(20, 20, 20, 12); v.setSpacing(10)

        brand = QHBoxLayout(); brand.setSpacing(10)
        logo = QLabel(); logo.setPixmap(app_icon().pixmap(38, 38))
        names = QVBoxLayout(); names.setSpacing(0)
        t = QLabel("Floorplan3D"); t.setObjectName("brand")
        s = QLabel("2D floor plan → measured 3D model"); s.setObjectName("brandSub")
        names.addWidget(t); names.addWidget(s)
        brand.addWidget(logo); brand.addLayout(names, 1)
        v.addLayout(brand); v.addSpacing(10)

        v.addWidget(self._section("INPUT"))
        self.drop = DropZone()
        self.drop.clicked.connect(self.browse)
        self.drop.fileSelected.connect(self.load_image)
        v.addWidget(self.drop)

        v.addSpacing(6)
        v.addWidget(self._section("SETTINGS"))
        grid = QGridLayout(); grid.setHorizontalSpacing(10); grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)

        self.scale_mode = QComboBox()
        self.scale_mode.addItems(["Automatic", "Pixels per foot", "Pixels per metre"])
        self.scale_mode.setToolTip("Automatic reads the plan's dimension lines.\n"
                                   "If the plan has none, enter the scale yourself:\n"
                                   "measure a known wall in pixels and divide by its length.")
        self.scale_mode.currentIndexChanged.connect(self._sync_controls)
        self.scale_val = QDoubleSpinBox(); self.scale_val.setRange(1, 5000); self.scale_val.setDecimals(2)
        self.scale_val.setValue(31.0)
        grid.addWidget(QLabel("Scale"), 0, 0); grid.addWidget(self.scale_mode, 0, 1)
        grid.addWidget(self.scale_val, 1, 1)

        self.wall_h = QDoubleSpinBox(); self.wall_h.setRange(1.8, 10); self.wall_h.setSingleStep(0.05)
        self.wall_h.setValue(2.75); self.wall_h.setSuffix(" m")
        grid.addWidget(QLabel("Wall height"), 2, 0); grid.addWidget(self.wall_h, 2, 1)

        self.trim = QSpinBox(); self.trim.setRange(0, 1000); self.trim.setSuffix(" px")
        self.trim.setToolTip("Crop this many pixels from every edge before analysis.\n"
                             "Use it to remove frames, borders or title blocks,\n"
                             "which would otherwise be detected as walls.")
        grid.addWidget(QLabel("Trim border"), 3, 0); grid.addWidget(self.trim, 3, 1)
        v.addLayout(grid)

        self.ocr = QCheckBox("Read text (room names, dimensions)"); self.ocr.setChecked(True)
        self.fast = QCheckBox("Fast mode (about 3× faster, less accurate)")
        v.addWidget(self.ocr); v.addWidget(self.fast)

        v.addSpacing(6)
        v.addWidget(self._section("OUTPUT FOLDER"))
        row = QHBoxLayout(); row.setSpacing(6)
        self.out_label = QLabel(); self.out_label.setObjectName("muted")
        self.out_label.setMinimumWidth(10)
        change = QPushButton("Change"); change.setObjectName("link"); change.setCursor(Qt.PointingHandCursor)
        change.clicked.connect(self.choose_out_dir)
        row.addWidget(self.out_label, 1); row.addWidget(change)
        v.addLayout(row)
        v.addStretch(1)

        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        foot = QVBoxLayout(); foot.setContentsMargins(20, 10, 20, 18); foot.setSpacing(8)
        self.run_btn = QPushButton("  Analyse plan"); self.run_btn.setObjectName("primary")
        self.run_btn.setIcon(icon("play", "#ffffff", 16)); self.run_btn.setIconSize(QSize(14, 14))
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.clicked.connect(self.run_analysis)
        note = QLabel("Runs offline on the CPU · no data leaves this computer"); note.setObjectName("faint")
        note.setAlignment(Qt.AlignCenter); note.setWordWrap(True)
        foot.addWidget(self.run_btn); foot.addWidget(note)
        outer.addLayout(foot)
        return side

    def _section(self, text):
        l = QLabel(text); l.setObjectName("section")
        return l

    def _build_main(self):
        main = QWidget()
        v = QVBoxLayout(main); v.setContentsMargins(24, 18, 24, 16); v.setSpacing(14)

        top = QHBoxLayout(); top.setSpacing(10)
        titles = QVBoxLayout(); titles.setSpacing(2)
        self.title = QLabel("No plan loaded"); self.title.setObjectName("title")
        self.subtitle = QLabel("Load an image to begin"); self.subtitle.setObjectName("muted")
        titles.addWidget(self.title); titles.addWidget(self.subtitle)
        top.addLayout(titles, 1)

        lbl = QLabel("Output"); lbl.setObjectName("muted")
        self.view_select = QComboBox(); self.view_select.setObjectName("viewSelect")
        self.view_select.setCursor(Qt.PointingHandCursor)
        for key, label, _ in VIEWS:
            self.view_select.addItem(label, key)
        self.fit_btn = self._tool_btn("fit", "Fit to window (or double-click the image)", lambda: self.image_view.fit())
        self.open_btn = self._tool_btn("external", "Open this output in its default app", self.open_external)
        self.export_btn = self._tool_btn("export", "Save this output as…", self.export_current)
        self.folder_btn = self._tool_btn("folder", "Open the output folder", self.open_folder)
        top.addWidget(lbl); top.addWidget(self.view_select)
        for b in (self.fit_btn, self.open_btn, self.export_btn, self.folder_btn):
            top.addWidget(b)
        v.addLayout(top)

        self.banner = Banner()
        v.addWidget(self.banner)

        self.stats_row = QWidget()
        sr = QHBoxLayout(self.stats_row); sr.setContentsMargins(0, 0, 0, 0); sr.setSpacing(10)
        self.stats = {k: StatCard(lbl) for k, lbl in [("rooms", "Rooms"), ("walls", "Walls"), ("doors", "Doors"),
                                                      ("windows", "Windows"), ("area", "Floor area"),
                                                      ("footprint", "Footprint"), ("scale", "Scale")]}
        for c in self.stats.values():
            sr.addWidget(c, 1)
        self.stats_row.hide()
        v.addWidget(self.stats_row)

        self.stack = QStackedWidget()
        self.page_empty = self._build_empty()
        self.page_busy = self._build_busy()
        self.image_view = ImageView()
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        self.table.setSortingEnabled(True)
        self.html_view = QTextBrowser(); self.html_view.setOpenExternalLinks(True)
        self.text_view = QPlainTextEdit(); self.text_view.setReadOnly(True)
        mono = QFont("Consolas"); mono.setStyleHint(QFont.Monospace); mono.setPointSize(10)
        self.text_view.setFont(mono)
        self.web_page = self._build_web()
        for w in (self.page_empty, self.page_busy, self.image_view, self.table, self.html_view,
                  self.text_view, self.web_page):
            self.stack.addWidget(w)
        v.addWidget(self.stack, 1)
        return main

    def _tool_btn(self, name, tip, slot):
        b = QPushButton(); b.setObjectName("iconBtn"); b.setIcon(icon(name, C["text"], 18))
        b.setIconSize(QSize(18, 18)); b.setToolTip(tip); b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(slot)
        return b

    def _build_empty(self):
        w = QWidget()
        v = QVBoxLayout(w); v.setAlignment(Qt.AlignCenter); v.setSpacing(14)
        logo = QLabel(); logo.setPixmap(app_icon().pixmap(72, 72)); logo.setAlignment(Qt.AlignCenter)
        h = QLabel("Turn a 2D floor plan into a 3D model"); h.setObjectName("h1"); h.setAlignment(Qt.AlignCenter)
        p = QLabel("Detects walls, doors, windows and rooms with real-world measurements,\n"
                   "then builds a 3D model you can explore and export.")
        p.setObjectName("muted"); p.setAlignment(Qt.AlignCenter)
        v.addWidget(logo); v.addWidget(h); v.addWidget(p); v.addSpacing(8)

        steps = QHBoxLayout(); steps.setSpacing(12)
        for n, title, text in [("1", "Load a plan", "Drop an image or click Browse"),
                               ("2", "Check settings", "Set the scale if the plan has no dimension lines"),
                               ("3", "Explore outputs", "Switch between 2D, 3D, tables and data")]:
            card = QFrame(); card.setObjectName("card"); card.setFixedWidth(220)
            cl = QVBoxLayout(card); cl.setContentsMargins(16, 14, 16, 14); cl.setSpacing(4)
            num = QLabel(n); num.setStyleSheet(f"color:{C['accent_hi']}; font-weight:700; font-size:15px;")
            t = QLabel(title); t.setStyleSheet("font-weight:600;")
            d = QLabel(text); d.setObjectName("muted"); d.setWordWrap(True)
            cl.addWidget(num); cl.addWidget(t); cl.addWidget(d)
            steps.addWidget(card)
        v.addLayout(steps); v.addSpacing(10)

        row = QHBoxLayout(); row.setAlignment(Qt.AlignCenter); row.setSpacing(12)
        b = QPushButton("  Browse for an image"); b.setObjectName("primary"); b.setIcon(icon("image", "#ffffff", 18))
        b.setCursor(Qt.PointingHandCursor); b.clicked.connect(self.browse)
        row.addWidget(b)
        if os.path.isfile(SAMPLE):
            s = QPushButton("Try the sample plan"); s.setCursor(Qt.PointingHandCursor)
            s.clicked.connect(lambda: self.load_image(SAMPLE))
            row.addWidget(s)
        v.addLayout(row)
        return w

    def _build_busy(self):
        w = QWidget()
        v = QVBoxLayout(w); v.setAlignment(Qt.AlignCenter)
        card = QFrame(); card.setObjectName("card"); card.setFixedWidth(460)
        cl = QVBoxLayout(card); cl.setContentsMargins(26, 24, 26, 24); cl.setSpacing(12)
        t = QLabel("Analysing plan…"); t.setObjectName("title")
        self.busy_elapsed = QLabel("0 s"); self.busy_elapsed.setObjectName("muted")
        head = QHBoxLayout(); head.addWidget(t, 1); head.addWidget(self.busy_elapsed)
        bar = QProgressBar(); bar.setRange(0, 0); bar.setTextVisible(False)
        self.busy_log = QLabel("Starting…"); self.busy_log.setObjectName("muted"); self.busy_log.setWordWrap(True)
        hint = QLabel("Reading text, segmenting the drawing, tracing walls and rooms, then building the 3D model. "
                      "This usually takes 10–60 seconds on a laptop CPU.")
        hint.setObjectName("faint"); hint.setWordWrap(True)
        cl.addLayout(head); cl.addWidget(bar); cl.addWidget(self.busy_log); cl.addWidget(hint)
        v.addWidget(card)
        return w

    def _build_web(self):
        w = QWidget()
        v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0)
        if self.web_engine is not None:
            self.web = self.web_engine()
            self.web.page().setBackgroundColor(QColor(C["bg"]))
            v.addWidget(self.web)
        else:
            self.web = None
            box = QVBoxLayout(); box.setAlignment(Qt.AlignCenter); box.setSpacing(10)
            msg = QLabel("The built-in 3D viewer is not available on this system.\n"
                         "Open the interactive model in your web browser instead.")
            msg.setObjectName("muted"); msg.setAlignment(Qt.AlignCenter)
            b = QPushButton("Open 3D viewer in browser"); b.setObjectName("primary")
            b.clicked.connect(self.open_external)
            box.addWidget(msg); box.addWidget(b, 0, Qt.AlignCenter)
            v.addLayout(box)
        return w

    # ------------------------------------------------------------------ settings
    def _load_settings(self):
        s = self.settings
        self.out_dir = s.value("out_dir", os.path.join(PROJECT_DIR, "output"))
        self.scale_mode.setCurrentIndex(int(s.value("scale_mode", 0)))
        self.scale_val.setValue(float(s.value("scale_val", 31.0)))
        self.wall_h.setValue(float(s.value("wall_h", 2.75)))
        self.trim.setValue(int(s.value("trim", 0)))
        self.ocr.setChecked(s.value("ocr", "true") in (True, "true"))
        self.fast.setChecked(s.value("fast", "false") in (True, "true"))
        self._show_out_dir()

    def _save_settings(self):
        s = self.settings
        for k, v in dict(out_dir=self.out_dir, scale_mode=self.scale_mode.currentIndex(), scale_val=self.scale_val.value(),
                         wall_h=self.wall_h.value(), trim=self.trim.value(), ocr=self.ocr.isChecked(),
                         fast=self.fast.isChecked()).items():
            s.setValue(k, v)

    def _show_out_dir(self):
        fm = self.out_label.fontMetrics()
        self.out_label.setText(fm.elidedText(self.out_dir, Qt.ElideMiddle, 220))
        self.out_label.setToolTip(self.out_dir)

    def choose_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Choose output folder", self.out_dir)
        if d:
            self.out_dir = d; self._show_out_dir(); self._save_settings()

    # ------------------------------------------------------------------ input
    def browse(self):
        if self.busy:
            return
        start = self.settings.value("last_dir", os.path.expanduser("~"))
        path, _ = QFileDialog.getOpenFileName(self, "Choose a floor plan image", start,
                                              "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if path:
            self.load_image(path)

    def load_image(self, path):
        if self.busy:
            return
        if not is_image(path) or not self.drop.set_image(path):
            QMessageBox.warning(self, "Unsupported file", f"This file can't be opened as an image:\n{path}")
            return
        self.image_path = path
        self.settings.setValue("last_dir", os.path.dirname(path))
        self.result = None
        self.title.setText(os.path.basename(path))
        self.subtitle.setText("Ready to analyse · check the settings, then click Analyse plan")
        self.banner.show_text("")
        self.stats_row.hide()
        self.view_select.blockSignals(True); self.view_select.setCurrentIndex(KEYS.index("input"))
        self.view_select.blockSignals(False)
        self.image_view.set_image(path)
        self.stack.setCurrentWidget(self.image_view)
        self.status.setText(f"Loaded {path}")
        self._sync_controls()

    def dragEnterEvent(self, e):
        urls = e.mimeData().urls()
        if urls and is_image(urls[0].toLocalFile()) and not self.busy:
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.load_image(e.mimeData().urls()[0].toLocalFile())

    # ------------------------------------------------------------------ analysis
    @property
    def busy(self):
        return self._thread is not None

    def _sync_controls(self):
        manual = self.scale_mode.currentIndex() > 0
        self.scale_val.setVisible(manual)
        self.scale_val.setEnabled(not self.busy)
        self.scale_val.setSuffix(" px/ft" if self.scale_mode.currentIndex() == 1 else " px/m")
        has = self.result is not None
        self.run_btn.setEnabled(bool(self.image_path) and not self.busy)
        self.run_btn.setText("  Analysing…" if self.busy else ("  Analyse again" if has else "  Analyse plan"))
        for w in (self.scale_mode, self.wall_h, self.trim, self.ocr, self.fast, self.drop):
            w.setEnabled(not self.busy)
        # before a run only the original input can be shown
        model = self.view_select.model()
        for i, (key, _, _) in enumerate(VIEWS):
            model.item(i).setEnabled(has or key == "input")
        self.view_select.setEnabled(bool(self.image_path) and not self.busy)
        key = self.current_key()
        self.fit_btn.setVisible(VIEWS[self.view_select.currentIndex()][2] == "image")
        self.open_btn.setEnabled(bool(self._primary_file(key)) and not self.busy)
        self.export_btn.setEnabled(has and not self.busy or (key == "input" and bool(self.image_path)))
        self.folder_btn.setEnabled(os.path.isdir(self.out_dir))

    def run_analysis(self):
        if not self.image_path or self.busy:
            return
        self._save_settings()
        mode = self.scale_mode.currentIndex()
        scale = None if mode == 0 else (self.scale_val.value() / FT if mode == 1 else self.scale_val.value())
        job = Job(image=self.image_path, out_dir=self.out_dir, scale_px_per_m=scale, wall_height_m=self.wall_h.value(),
                  trim_px=self.trim.value(), use_ocr=self.ocr.isChecked(), fast=self.fast.isChecked())
        self._thread = QThread(self)
        self._worker = AnalysisWorker(job)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._started = time.time()
        self.busy_log.setText("Starting…")
        self.stack.setCurrentWidget(self.page_busy)
        self.banner.show_text("")
        self._thread.start()
        self._timer.start()
        self._sync_controls()

    def _tick(self):
        self.busy_elapsed.setText(f"{time.time() - self._started:.0f} s")

    def _on_log(self, line):
        if line.startswith("[floorplan3d]"):
            line = line.split("|", 1)[0].replace("[floorplan3d]", "").strip()
        self.busy_log.setText(line)

    def _finish_thread(self):
        self._timer.stop()
        self._thread.quit(); self._thread.wait()
        self._thread = None; self._worker = None

    def _on_failed(self, msg):
        self._finish_thread()
        self.stack.setCurrentWidget(self.image_view if self.image_path else self.page_empty)
        self.status.setText("Analysis failed")
        box = QMessageBox(self); box.setIcon(QMessageBox.Critical); box.setWindowTitle("Analysis failed")
        first = msg.split("\n", 1)[0]
        box.setText("The plan could not be analysed."); box.setInformativeText(first)
        box.setDetailedText(msg); box.exec()
        self._sync_controls()

    def _on_finished(self, result):
        took = time.time() - self._started
        self._finish_thread()
        self.result = result
        d = result["data"]; b = d["building"]; sc = d["scale"]
        est = sc["method"].startswith("estimated")
        self.stats["rooms"].set(b["room_count"])
        self.stats["walls"].set(b["wall_count"])
        self.stats["doors"].set(b["door_count"])
        self.stats["windows"].set(b["window_count"])
        self.stats["area"].set(f"{b['floor_area_m2'] * 10.7639:,.0f} ft²", f"{b['floor_area_m2']:.1f} m²")
        self.stats["footprint"].set(f"{b['width_ft_in']} × {b['depth_ft_in']}".replace("' ", "'"),
                                    f"{b['width_m']:.2f} × {b['depth_m']:.2f} m")
        method = {"user_supplied": "set manually", "dimension_annotations": "from dimension lines"}.get(sc["method"], "estimated – verify")
        self.stats["scale"].set(f"{sc['px_per_ft_x']:.2f} px/ft", method, C["warn"] if est else None)
        self.stats_row.show()
        warn = d.get("warning", "")
        if est:
            warn = ("The scale was estimated because the plan has no readable dimension lines, so sizes may be wrong. "
                    "Set the scale manually in Settings (pixels ÷ feet of a known wall) and analyse again.")
        self.banner.show_text(warn)
        self.subtitle.setText(f"Analysed in {took:.0f} s · files saved to the “{os.path.basename(self.out_dir)}” folder")
        self.subtitle.setToolTip(self.out_dir)
        self.status.setText(f"Done in {took:.1f} s · {os.path.basename(result['outputs']['json_3d'])} and 3D files written")
        self._sync_controls()
        if self.view_select.currentIndex() == 0:
            self.show_view(0)
        else:
            self.view_select.setCurrentIndex(0)

    # ------------------------------------------------------------------ views
    def current_key(self):
        return VIEWS[self.view_select.currentIndex()][0]

    def show_view(self, index):
        key, _, kind = VIEWS[index]
        self._sync_controls()
        if self.busy:
            return
        if key == "input":
            if self.image_path:
                self.image_view.set_image(self.image_path); self.stack.setCurrentWidget(self.image_view)
            return
        if not self.result:
            return
        out, d = self.result["outputs"], self.result["data"]
        if kind == "image":
            self.image_view.set_image(out["detected_image"] if key == "detected" else out.get("preview", ""))
            self.stack.setCurrentWidget(self.image_view)
        elif kind == "web":
            if self.web is not None:
                self.web.load(QUrl.fromLocalFile(os.path.abspath(out["viewer_html"])))
            self.stack.setCurrentWidget(self.web_page)
        elif kind == "table":
            getattr(self, f"_fill_{key}")(d)
            self.stack.setCurrentWidget(self.table)
        elif kind == "html":
            self.html_view.setHtml(self._scale_html(d)); self.stack.setCurrentWidget(self.html_view)
        elif kind == "text":
            self.text_view.setPlainText(json.dumps(d, indent=2, ensure_ascii=False))
            self.stack.setCurrentWidget(self.text_view)

    def _set_table(self, headers, rows):
        t = self.table
        t.setSortingEnabled(False)
        t.clear(); t.setColumnCount(len(headers)); t.setRowCount(len(rows))
        t.setHorizontalHeaderLabels(headers)
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                it = val if isinstance(val, QTableWidgetItem) else QTableWidgetItem(str(val))
                t.setItem(r, c, it)
        hh = t.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setStretchLastSection(True)
        hh.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        t.verticalHeader().setDefaultSectionSize(34)
        t.setSortingEnabled(True)

    def _fill_rooms(self, d):
        rows = []
        for r in d["rooms"]:
            name = QTableWidgetItem(r["name"]); name.setIcon(_swatch(ROOM_COLORS.get(r["type"], (220, 220, 220))))
            src = "label on plan" if r["label_source"] == "ocr_label" else ("model" if r["label_source"] == "segmentation_model" else "default")
            rows.append([r["id"], name, r["type"].replace("_", " "), f"{r['width_ft_in']} × {r['depth_ft_in']}",
                         f"{r['width_m']:.2f} × {r['depth_m']:.2f}", _num_item(r["area_ft2"], "{:,.1f}"),
                         _num_item(r["area_m2"]), ", ".join(r["doors"]) or "–", src])
        self._set_table(["ID", "Room", "Type", "Size (ft-in)", "Size (m)", "Area ft²", "Area m²", "Doors", "Named from"], rows)

    def _fill_walls(self, d):
        rows = []
        for w in d["walls"]:
            typ = QTableWidgetItem(w["type"])
            typ.setForeground(QColor("#ef7a66" if w["type"] == "exterior" else "#f0a160"))
            rows.append([w["id"], typ, w["orientation"], w["length_ft_in"], _num_item(w["length_m"]),
                         _num_item(w["thickness_m"]), _num_item(w["height_m"]), ", ".join(w["openings"]) or "–"])
        self._set_table(["ID", "Type", "Orientation", "Length (ft-in)", "Length m", "Thickness m", "Height m", "Openings"], rows)

    def _fill_openings(self, d):
        rows = []
        for o in d["openings"]:
            typ = QTableWidgetItem(o["type"])
            typ.setForeground(QColor("#5cc485" if o["type"] == "door" else "#6ea3f2"))
            note = "inferred from wall gap" if o.get("inferred_from_wall_gap") else ""
            rows.append([o["id"], typ, o.get("wall_id") or "–", o["width_ft_in"], _num_item(o["width_m"]),
                         _num_item(o["sill_height_m"]), _num_item(o["height_m"]),
                         " ↔ ".join(o.get("connects", [])) or "–", note])
        self._set_table(["ID", "Type", "Wall", "Width (ft-in)", "Width m", "Sill m", "Height m", "Connects", "Note"], rows)

    def _scale_html(self, d):
        sc = d["scale"]
        esc = html.escape
        css = (f"<style>body{{color:{C['text']};}} h3{{margin:14px 0 6px;}} td,th{{padding:5px 14px 5px 0;}}"
               f"th{{color:{C['muted']};text-align:left;font-weight:600;}} .m{{color:{C['muted']};}}</style>")
        method = {"user_supplied": "Set manually in Settings", "dimension_annotations": "Measured from the plan's dimension lines"}.get(
            sc["method"], "Estimated (" + sc["method"] + ") – verify")
        h = [css, "<h3>Scale</h3><table>",
             f"<tr><th>Method</th><td>{esc(method)}</td></tr>",
             f"<tr><th>Horizontal</th><td>{sc['px_per_ft_x']:.2f} px/ft · {sc['px_per_m_x']:.2f} px/m</td></tr>",
             f"<tr><th>Vertical</th><td>{sc['px_per_ft_y']:.2f} px/ft · {sc['px_per_m_y']:.2f} px/m</td></tr></table>"]
        if sc["annotations"]:
            h.append("<h3>Dimension strings checked</h3><table><tr><th>Text</th><th>Axis</th><th>Printed</th>"
                     "<th>Measured</th><th>Error</th><th>Used</th></tr>")
            for a in sc["annotations"]:
                h.append(f"<tr><td>{esc(a['text'])}</td><td>{a['axis']}</td><td>{a['annotated_ft_in']}</td>"
                         f"<td>{a['measured_m']:.2f} m</td><td>{a['error_pct']:+.1f}%</td><td>{'yes' if a['used'] else 'no'}</td></tr>")
            h.append("</table>")
        h.append("<h3>Recognised text</h3><table><tr><th>Text</th><th>Confidence</th><th>Read as</th></tr>")
        from floorplan3d.ocr import room_type_from_text
        for t in d["texts"]:
            kind = (f"dimension {t['dimension_m']:.2f} m" if t.get("dimension_m") else
                    (room_type_from_text(t["text"]) or "").replace("_", " ") or "other")
            h.append(f"<tr><td>{esc(t['text'])}</td><td>{t['score']:.2f}</td><td class='m'>{esc(kind)}</td></tr>")
        h.append("</table>")
        return "".join(h)

    # ------------------------------------------------------------------ actions
    def _primary_file(self, key):
        if key == "input":
            return self.image_path
        if not self.result:
            return None
        out = self.result["outputs"]
        return {"detected": out.get("detected_image"), "viewer": out.get("viewer_html"), "preview": out.get("preview"),
                "json": out.get("json_3d"), "scale": out.get("json_3d")}.get(key)

    def open_external(self):
        f = self._primary_file(self.current_key())
        if f and os.path.exists(f):
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(f)))

    def open_folder(self):
        os.makedirs(self.out_dir, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(self.out_dir)))

    def export_current(self):
        key = self.current_key()
        start = self.settings.value("export_dir", os.path.expanduser("~"))
        if VIEWS[self.view_select.currentIndex()][2] == "table":
            path, _ = QFileDialog.getSaveFileName(self, "Export table", os.path.join(start, f"{self._stem()}_{key}.csv"), "CSV (*.csv)")
            if path:
                self._export_csv(path)
        elif key == "viewer":
            out = self.result["outputs"]
            opts = {"Interactive 3D viewer (*.html)": out["viewer_html"], "glTF binary model (*.glb)": out["glb"],
                    "Wavefront OBJ model (*.obj)": out["obj"]}
            path, flt = QFileDialog.getSaveFileName(self, "Export 3D model", os.path.join(start, f"{self._stem()}_3d.glb"),
                                                    ";;".join(opts), "glTF binary model (*.glb)")
            if path:
                src = opts[flt]
                ext = os.path.splitext(src)[1]
                if not path.lower().endswith(ext):
                    path += ext
                shutil.copyfile(src, path)
                if ext == ".obj":  # OBJ needs its material file next to it
                    mtl = os.path.splitext(src)[0] + ".mtl"
                    if os.path.exists(mtl):
                        shutil.copyfile(mtl, os.path.join(os.path.dirname(path), os.path.basename(mtl)))
        else:
            src = self._primary_file(key)
            if not src:
                return
            ext = os.path.splitext(src)[1]
            path, _ = QFileDialog.getSaveFileName(self, "Save as", os.path.join(start, os.path.basename(src)), f"*{ext}")
            if path:
                shutil.copyfile(src, path)
        if path:
            self.settings.setValue("export_dir", os.path.dirname(path))
            self.status.setText(f"Saved {path}")

    def _stem(self):
        return os.path.splitext(os.path.basename(self.image_path or "plan"))[0]

    def _export_csv(self, path):
        t = self.table
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow([t.horizontalHeaderItem(c).text() for c in range(t.columnCount())])
            for r in range(t.rowCount()):
                w.writerow([t.item(r, c).text() if t.item(r, c) else "" for c in range(t.columnCount())])

    def closeEvent(self, e):
        if self.busy:
            ans = QMessageBox.question(self, "Analysis running", "An analysis is still running. Quit anyway?")
            if ans != QMessageBox.Yes:
                e.ignore(); return
            self._save_settings()
            os._exit(0)  # the pipeline can't be interrupted mid-run
        self._save_settings()
        e.accept()
