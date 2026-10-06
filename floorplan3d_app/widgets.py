"""Reusable widgets: image drop zone, zoom/pan image viewer, stat card, warning banner."""
import os

from PySide6.QtCore import QEvent, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QImageReader, QPainter, QPixmap
from PySide6.QtWidgets import (QFrame, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QGridLayout,
                               QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget)

from .theme import C, icon

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def is_image(path):
    return bool(path) and path.lower().endswith(IMAGE_EXTS) and os.path.isfile(path)


def _restyle(w):
    w.style().unpolish(w); w.style().polish(w)


class ElidedLabel(QLabel):
    """One-line label that shortens long text with '…' to fit its width instead of
    forcing the layout wider (a plain QLabel never shrinks below its text width).
    When the text is shortened, the full text is shown as a tooltip."""

    def __init__(self, text="", mode=Qt.ElideMiddle):
        super().__init__()
        self._full, self._mode = "", mode
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setText(text)

    def setText(self, text):
        self._full = text or ""
        self._apply()

    def text(self):
        return self._full

    def _apply(self):
        shown = self.fontMetrics().elidedText(self._full, self._mode, max(0, self.width()))
        super().setText(shown)
        self.setToolTip(self._full if shown != self._full else "")

    def minimumSizeHint(self):
        return QSize(16, super().minimumSizeHint().height())

    def sizeHint(self):
        return QSize(self.fontMetrics().horizontalAdvance(self._full) + 4, super().sizeHint().height())

    def resizeEvent(self, e):
        super().resizeEvent(e); self._apply()

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() in (QEvent.FontChange, QEvent.StyleChange):
            self._apply()


class DropZone(QFrame):
    """Click or drop an image here. Shows a thumbnail once loaded."""
    fileSelected = Signal(str)
    clicked = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(190)
        lay = QVBoxLayout(self); lay.setContentsMargins(14, 14, 14, 14); lay.setSpacing(6)
        self.thumb = QLabel(); self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.title = ElidedLabel("Drop a floor plan here"); self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-weight:600; font-size:14px;")
        self.sub = QLabel("or click to browse · PNG, JPG, BMP, TIFF, WebP"); self.sub.setObjectName("muted")
        self.sub.setAlignment(Qt.AlignCenter); self.sub.setWordWrap(True)
        lay.addWidget(self.thumb, 1); lay.addWidget(self.title); lay.addWidget(self.sub)
        self._pix = None
        self._show_placeholder()

    def _show_placeholder(self):
        self.thumb.setPixmap(icon("upload", C["accent_hi"], 40).pixmap(40, 40))

    def set_image(self, path):
        pm = QPixmap(path)
        if pm.isNull():
            return False
        self._pix = pm
        self._fit_thumb()
        size = QImageReader(path).size()
        self.title.setText(os.path.basename(path))
        self.sub.setText(f"{size.width()} × {size.height()} px · click to change")
        self.setProperty("loaded", True); _restyle(self)
        return True

    def _fit_thumb(self):
        if self._pix is not None:
            w, h = max(40, self.thumb.width() - 4), max(40, self.thumb.height() - 4)
            self.thumb.setPixmap(self._pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e); self._fit_thumb()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit()

    def dragEnterEvent(self, e):
        urls = e.mimeData().urls()
        if urls and is_image(urls[0].toLocalFile()):
            e.acceptProposedAction(); self.setProperty("hover", True); _restyle(self)

    def dragLeaveEvent(self, e):
        self.setProperty("hover", False); _restyle(self)

    def dropEvent(self, e):
        self.setProperty("hover", False); _restyle(self)
        self.fileSelected.emit(e.mimeData().urls()[0].toLocalFile())


class ImageView(QGraphicsView):
    """Image viewer: wheel to zoom around the cursor, drag to pan, double-click to fit."""

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self._item = QGraphicsPixmapItem(); self._item.setTransformationMode(Qt.SmoothTransformation)
        self.scene().addItem(self._item)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self._fitted = True

    def set_image(self, path):
        pm = QPixmap(path)
        self._item.setPixmap(pm)
        self.scene().setSceneRect(QRectF(pm.rect()))
        self.fit()
        return not pm.isNull()

    def fit(self):
        if not self._item.pixmap().isNull():
            self.fitInView(self._item, Qt.KeepAspectRatio)
            self._fitted = True

    def wheelEvent(self, e):
        f = 1.2 if e.angleDelta().y() > 0 else 1 / 1.2
        cur = self.transform().m11()
        if 0.02 < cur * f < 20:
            self.scale(f, f); self._fitted = False

    def mouseDoubleClickEvent(self, e):
        self.fit()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._fitted:
            self.fit()


class StatCard(QFrame):
    def __init__(self, label):
        super().__init__()
        self.setObjectName("stat")
        lay = QVBoxLayout(self); lay.setContentsMargins(14, 10, 14, 10); lay.setSpacing(1)
        # elided labels let the seven cards share a narrow window instead of overflowing it
        self.value = ElidedLabel("–", Qt.ElideRight); self.value.setObjectName("statValue")
        self.label = ElidedLabel(label.upper(), Qt.ElideRight); self.label.setObjectName("statLabel")
        self.sub = ElidedLabel("", Qt.ElideRight); self.sub.setObjectName("faint")
        lay.addWidget(self.value); lay.addWidget(self.label); lay.addWidget(self.sub)
        self.sub.hide()

    def set(self, value, sub=None, color=None):
        self.value.setText(str(value))
        self.value.setStyleSheet(f"color:{color};" if color else "")
        self.sub.setText(sub or ""); self.sub.setVisible(bool(sub))
        self.updateGeometry()

    def sizeHint(self):
        m = self.layout().contentsMargins()
        w = max(l.sizeHint().width() for l in (self.value, self.label, self.sub))
        return QSize(w + m.left() + m.right(), super().sizeHint().height())


class StatsRow(QWidget):
    """Lays out stat cards in as many equal columns as fit at their natural width,
    wrapping to further rows on narrow windows so no value has to be cut off."""

    def __init__(self, cards, gap=10):
        super().__init__()
        self.cards, self.gap = list(cards), gap
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0); self._grid.setSpacing(gap)
        self._cols = 0
        self.relayout()

    def _needed(self):
        return max(c.sizeHint().width() for c in self.cards)

    def relayout(self):
        n = len(self.cards)
        hints = [c.sizeHint().width() for c in self.cards]
        if self.width() <= 0 or sum(hints) + self.gap * (n - 1) <= self.width():
            cols, stretch = n, hints    # one row; wider content gets a wider card
        else:
            cols = max(1, min(n, (self.width() + self.gap) // (max(hints) + self.gap)))
            cols = -(-n // -(-n // cols))   # balance the rows: 7 cards -> 4 + 3, not 6 + 1
            stretch = [1] * cols
        key = (cols, tuple(stretch))
        if key == self._cols:
            return
        self._cols = key
        for c in self.cards:
            self._grid.removeWidget(c)
        for i in range(self._grid.columnCount()):
            self._grid.setColumnStretch(i, 0)
        for i, c in enumerate(self.cards):
            self._grid.addWidget(c, i // cols, i % cols)
        for i in range(cols):
            self._grid.setColumnStretch(i, stretch[i])

    def minimumSizeHint(self):
        return QSize(self._needed(), super().minimumSizeHint().height())

    def resizeEvent(self, e):
        super().resizeEvent(e); self.relayout()


class Banner(QFrame):
    """Amber warning strip shown above the results (e.g. estimated scale)."""

    def __init__(self):
        super().__init__()
        self.setObjectName("banner")
        lay = QHBoxLayout(self); lay.setContentsMargins(12, 9, 12, 9); lay.setSpacing(10)
        ic = QLabel(); ic.setPixmap(icon("warn", C["warn"], 18).pixmap(18, 18))
        self.text = QLabel(); self.text.setObjectName("bannerText"); self.text.setWordWrap(True)
        lay.addWidget(ic, 0, Qt.AlignTop); lay.addWidget(self.text, 1)
        self.hide()

    def show_text(self, text):
        self.text.setText(text); self.setVisible(bool(text))
