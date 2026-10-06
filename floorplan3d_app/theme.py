"""Colours, icons and the Qt stylesheet. One dark 'drafting table' theme; the accent
blue and the red / green match the colours used in the detection image."""
from PySide6.QtCore import QByteArray, QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

C = dict(
    bg="#12161b", panel="#181e25", card="#1f2730", card_hi="#26303b", border="#2c3743",
    text="#e7ebef", muted="#8f9cab", faint="#5d6a78",
    accent="#3d8bfd", accent_hi="#5a9dff", accent_lo="#1f4f96",
    ok="#3fb27f", warn="#e8a33d", warn_bg="#3a2c17", danger="#e5534b",
)

# 24x24 line icons (stroke = currentColor, replaced at render time)
_ICONS = {
    "upload": '<path d="M12 16V4m0 0l-5 5m5-5l5 5"/><path d="M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3"/>',
    "play": '<path d="M7 4.5v15l12-7.5z"/>',
    "folder": '<path d="M3 6.5A1.5 1.5 0 014.5 5H9l2 2.5h8.5A1.5 1.5 0 0121 9v9.5a1.5 1.5 0 01-1.5 1.5h-15A1.5 1.5 0 013 18.5z"/>',
    "export": '<path d="M12 4v11m0 0l-4.5-4.5M12 15l4.5-4.5"/><path d="M5 19h14"/>',
    "external": '<path d="M14 4h6v6m0-6l-9 9"/><path d="M18 14v4.5a1.5 1.5 0 01-1.5 1.5h-11A1.5 1.5 0 014 18.5v-11A1.5 1.5 0 015.5 6H10"/>',
    "fit": '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
    "image": '<rect x="3.5" y="4.5" width="17" height="15" rx="1.5"/><path d="M3.5 16l5-5 4 4 3-3 5 5"/><circle cx="15.5" cy="9" r="1.5"/>',
    "plan": '<path d="M3.5 3.5h17v17h-17z"/><path d="M3.5 11h7v9.5M10.5 3.5V8M14 11h6.5M14 11v4"/>',
    "warn": '<path d="M12 3.5l9.5 16.5h-19z"/><path d="M12 10v4.5M12 17.2v.3"/>',
}


def icon(name, color=None, size=20):
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%s" '
           'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">%s</svg>') % (color or C["text"], _ICONS[name])
    if name == "play":
        svg = svg.replace('fill="none"', 'fill="%s"' % (color or C["text"]))
    r = QSvgRenderer(QByteArray(svg.encode()))
    ic = QIcon()
    for s in (size, size * 2):
        pm = QPixmap(QSize(s, s)); pm.fill(Qt.transparent)
        p = QPainter(pm); r.render(p); p.end()
        ic.addPixmap(pm)
    return ic


def app_icon():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
           '<rect width="64" height="64" rx="14" fill="%s"/>'
           '<path d="M14 40l18-10 18 10-18 10z" fill="%s" opacity=".55"/>'
           '<path d="M14 24l18-10 18 10-18 10z" fill="none" stroke="#fff" stroke-width="3" stroke-linejoin="round"/>'
           '<path d="M14 24v16M50 24v16M32 34v16" stroke="#fff" stroke-width="3" stroke-linecap="round"/></svg>') % (C["accent_lo"], C["accent_hi"])
    r = QSvgRenderer(QByteArray(svg.encode()))
    ic = QIcon()
    for s in (16, 32, 64, 128, 256):
        pm = QPixmap(QSize(s, s)); pm.fill(Qt.transparent)
        p = QPainter(pm); r.render(p); p.end()
        ic.addPixmap(pm)
    return ic


STYLE = """
* {{ outline: none; }}
QMainWindow, QWidget#root {{ background: {bg}; }}
QWidget {{ color: {text}; font-size: 13px; }}
QToolTip {{ background: {card_hi}; color: {text}; border: 1px solid {border}; padding: 6px 8px; border-radius: 6px; }}

QFrame#sidebar {{ background: {panel}; border-right: 1px solid {border}; }}
QLabel#brand {{ font-size: 18px; font-weight: 700; letter-spacing: 0.3px; }}
QLabel#brandSub {{ color: {muted}; font-size: 12px; }}
QLabel#section {{ color: {muted}; font-size: 11px; font-weight: 600; letter-spacing: 1.2px; padding-top: 6px; }}
QLabel#muted {{ color: {muted}; }}
QLabel#faint {{ color: {faint}; font-size: 11px; }}
QLabel#title {{ font-size: 17px; font-weight: 600; }}
QLabel#h1 {{ font-size: 22px; font-weight: 700; }}

QFrame#dropZone {{ background: {card}; border: 1.5px dashed {border}; border-radius: 12px; }}
QFrame#dropZone[hover="true"] {{ border-color: {accent}; background: {card_hi}; }}
QFrame#dropZone[loaded="true"] {{ border-style: solid; }}

QFrame#card {{ background: {card}; border: 1px solid {border}; border-radius: 10px; }}
QFrame#stat {{ background: {card}; border: 1px solid {border}; border-radius: 10px; }}
QLabel#statValue {{ font-size: 19px; font-weight: 700; }}
QLabel#statLabel {{ color: {muted}; font-size: 11px; letter-spacing: 0.6px; }}
QFrame#banner {{ background: {warn_bg}; border: 1px solid {warn}; border-radius: 8px; }}
QLabel#bannerText {{ color: #f3c98a; }}

QPushButton {{ background: {card}; border: 1px solid {border}; border-radius: 8px; padding: 7px 14px; }}
QPushButton:hover {{ background: {card_hi}; border-color: #3a4757; }}
QPushButton:pressed {{ background: {border}; }}
QPushButton:disabled {{ color: {faint}; background: {panel}; border-color: {card}; }}
QPushButton#primary {{ background: {accent}; border: none; color: white; font-weight: 600; font-size: 14px; padding: 11px 16px; border-radius: 9px; }}
QPushButton#primary:hover {{ background: {accent_hi}; }}
QPushButton#primary:disabled {{ background: {accent_lo}; color: #9fb7da; }}
QPushButton#link {{ background: transparent; border: none; color: {accent_hi}; padding: 2px 4px; }}
QPushButton#link:hover {{ color: white; }}
QPushButton#iconBtn {{ padding: 7px; }}

QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {{
    background: {card}; border: 1px solid {border}; border-radius: 7px; padding: 6px 9px; min-height: 18px;
    selection-background-color: {accent};
}}
QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover, QLineEdit:hover {{ border-color: #3a4757; }}
QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus {{ border-color: {accent}; }}
QComboBox:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled {{ color: {faint}; background: {panel}; }}
QComboBox::drop-down {{ border: none; width: 26px; }}
QComboBox::down-arrow {{ image: url(__ARROW__); width: 12px; height: 12px; margin-right: 10px; }}
QComboBox QAbstractItemView {{ background: {card}; border: 1px solid {border}; border-radius: 8px; padding: 4px; selection-background-color: {accent_lo}; outline: none; }}
QComboBox QAbstractItemView::item {{ min-height: 30px; padding: 0 8px; border-radius: 5px; }}
QComboBox#viewSelect {{ font-weight: 600; min-width: 230px; padding: 7px 12px; }}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button, QSpinBox::up-button, QSpinBox::down-button {{ width: 0; border: none; }}

QCheckBox {{ spacing: 9px; }}
QCheckBox::indicator {{ width: 17px; height: 17px; border-radius: 5px; border: 1px solid {border}; background: {card}; }}
QCheckBox::indicator:hover {{ border-color: {accent}; }}
QCheckBox::indicator:checked {{ background: {accent}; border-color: {accent};
    image: url(__CHECK__); }}

QProgressBar {{ background: {card}; border: none; border-radius: 3px; max-height: 6px; min-height: 6px; }}
QProgressBar::chunk {{ background: {accent}; border-radius: 3px; }}

QTableWidget {{ background: {card}; alternate-background-color: #222b35; border: 1px solid {border}; border-radius: 10px;
    gridline-color: transparent; selection-background-color: {accent_lo}; }}
QTableWidget::item {{ padding: 4px 10px; border-bottom: 1px solid {border}; }}
QHeaderView::section {{ background: {panel}; color: {muted}; border: none; border-bottom: 1px solid {border};
    padding: 8px 10px; font-size: 11px; font-weight: 600; letter-spacing: 0.5px; }}
QTableCornerButton::section {{ background: {panel}; border: none; }}

QPlainTextEdit, QTextBrowser {{ background: {card}; border: 1px solid {border}; border-radius: 10px; padding: 10px;
    selection-background-color: {accent_lo}; }}
QGraphicsView {{ background: #0d1014; border: 1px solid {border}; border-radius: 10px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #34404d; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #45525f; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #34404d; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

QStatusBar {{ background: {panel}; color: {muted}; border-top: 1px solid {border}; }}
QStatusBar QLabel {{ color: {muted}; padding: 0 8px; }}
QMessageBox {{ background: {panel}; }}
""".format(**C)


def stylesheet(asset_dir):
    """Write the two tiny SVGs Qt stylesheets need as files and return the final QSS."""
    import os
    os.makedirs(asset_dir, exist_ok=True)
    files = {
        "check.svg": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M3.5 8.5l3 3 6-7" fill="none" '
                     'stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
        "arrow.svg": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 12 12"><path d="M2.5 4.5l3.5 3.5 3.5-3.5" fill="none" '
                     'stroke="%s" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>' % C["muted"],
    }
    for name, svg in files.items():
        with open(os.path.join(asset_dir, name), "w", encoding="utf-8") as f:
            f.write(svg)
    url = lambda n: os.path.join(asset_dir, n).replace("\\", "/")
    return STYLE.replace("__CHECK__", url("check.svg")).replace("__ARROW__", url("arrow.svg"))
