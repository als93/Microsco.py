#!/usr/bin/env python3
"""Add a beginner-friendly, grid-based Stitch tab to Microsco.py.

Usage: python3 add_stitching.py Microsco.py
Creates a .bak backup and refuses to patch unless every anchor is unique.
Stitching is optional: the existing viewer remains unchanged until this script is run,
and the resulting Stitch tab is independent of capture/stacking workflows.
"""
import shutil
import sys
from pathlib import Path

FUNCTIONS = r'''# ======================================================================
# Stitching (grid mode): join focus-stacked microscope tiles
# ======================================================================
MAX_STITCH_TILES = 16
MAX_STITCH_PIXELS = 40_000_000
MIN_STITCH_SCORE = 0.30


def _stitch_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return gray - cv2.GaussianBlur(gray, (0, 0), 12)


def _stitch_pair(a, b, side, overlap):
    """Estimate b's offset from a using normalized correlation in their expected overlap."""
    ga, gb = _stitch_gray(a), _stitch_gray(b)
    if ga.shape != gb.shape:
        raise ValueError("All tiles must have the same image dimensions")
    if side == "below":
        ga, gb = np.ascontiguousarray(ga.T), np.ascontiguousarray(gb.T)
    h, w = ga.shape
    scale = min(1.0, 640.0 / max(h, w))
    if scale < 1:
        ga = cv2.resize(ga, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gb = cv2.resize(gb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw = ga.shape
    tw = max(8, int(sw * overlap * 0.8))
    y0, y1 = int(sh * .2), int(sh * .8)
    templ = gb[y0:y1, :tw]
    x0, x1 = int(sw * .1), min(sw, int(sw * (1 - overlap)) + tw)
    area = ga[:, x0:x1]
    if area.shape[0] < templ.shape[0] or area.shape[1] < templ.shape[1]:
        raise ValueError("Tiles are too small for the selected overlap")
    result = cv2.matchTemplate(area, templ, cv2.TM_CCOEFF_NORMED)
    result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)
    _, score, _, loc = cv2.minMaxLoc(result)
    x = loc[0] + x0
    y = loc[1] - y0
    if scale < 1:
        x, y = round(x / scale), round(y / scale)
    if side == "below":
        x, y = y, x
    return (float(x), float(y)), float(score)


def stitch_grid(tiles, rows, cols, overlap, cancelled=None, progress=None):
    """Stitch tiles in reading order. Returns (mosaic, warnings)."""
    if not tiles or len(tiles) != rows * cols:
        raise ValueError(f"This {rows} x {cols} grid needs {rows * cols} tiles")
    if len(tiles) > MAX_STITCH_TILES:
        raise ValueError(f"Please use {MAX_STITCH_TILES} tiles or fewer")
    if any(t.shape != tiles[0].shape for t in tiles):
        raise ValueError("All tiles must have the same image dimensions")
    h, w = tiles[0].shape[:2]
    # Conservative upper bound before allocating the mosaic and float blend buffers.
    estimated = int(w * cols * h * rows)
    if estimated > MAX_STITCH_PIXELS:
        raise ValueError("This grid is too large for safe stitching. Use fewer or smaller tiles.")
    positions = {(0, 0): (0.0, 0.0)}
    warnings, done = [], 0
    total = rows * (cols - 1) + (rows - 1) * cols
    for r in range(rows):
        for c in range(cols):
            if (r, c) == (0, 0):
                continue
            if cancelled and cancelled():
                raise InterruptedError("stitch cancelled")
            idx = r * cols + c
            guesses = []
            for side, pr, pc in (("right", r, c - 1), ("below", r - 1, c)):
                if pr < 0 or pc < 0:
                    continue
                (dx, dy), score = _stitch_pair(
                    tiles[pr * cols + pc], tiles[idx], side, overlap)
                px, py = positions[(pr, pc)]
                guesses.append((px + dx, py + dy, score))
                if score < MIN_STITCH_SCORE:
                    warnings.append(f"Tile {idx + 1} has a weak neighbour match ({score:.2f})")
                done += 1
                if progress:
                    progress(done, total)
            best = max(g[2] for g in guesses)
            keep = [g for g in guesses if g[2] >= max(MIN_STITCH_SCORE, best * .5)]
            if not keep:
                raise ValueError(f"Could not confidently match tile {idx + 1}. "
                                 "Check overlap and image texture, then try again.")
            weights = [max(g[2], .01) for g in keep]
            positions[(r, c)] = (
                sum(g[0] * wt for g, wt in zip(keep, weights)) / sum(weights),
                sum(g[1] * wt for g, wt in zip(keep, weights)) / sum(weights))

    points = [positions[(r, c)] for r in range(rows) for c in range(cols)]
    x0, y0 = min(p[0] for p in points), min(p[1] for p in points)
    offsets = [(round(x - x0), round(y - y0)) for x, y in points]
    out_w = max(x for x, _ in offsets) + w
    out_h = max(y for _, y in offsets) + h
    if out_w * out_h > MAX_STITCH_PIXELS:
        raise ValueError("The matched layout is larger than the safe limit. Try fewer tiles.")
    weight_y = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32) + 1
    weight_x = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32) + 1
    weight = np.minimum.outer(weight_y, weight_x)
    numerator = np.zeros((out_h, out_w, 3), np.float32)
    denominator = np.zeros((out_h, out_w), np.float32)
    for i, (tile, (ox, oy)) in enumerate(zip(tiles, offsets)):
        if cancelled and cancelled():
            raise InterruptedError("stitch cancelled")
        numerator[oy:oy+h, ox:ox+w] += tile.astype(np.float32) * weight[..., None]
        denominator[oy:oy+h, ox:ox+w] += weight
    mosaic = np.clip(numerator / np.maximum(denominator, 1e-6)[..., None], 0, 255).astype(np.uint8)
    return mosaic, warnings


'''

THREAD = r'''class StitchThread(QThread):
    progress = Signal(int, int)
    done = Signal(object, object)
    failed = Signal(str)

    def __init__(self, paths, rows, cols, overlap):
        super().__init__()
        self.paths, self.rows, self.cols, self.overlap = paths, rows, cols, overlap

    def run(self):
        try:
            if len(self.paths) != self.rows * self.cols:
                raise ValueError(f"Grid needs {self.rows * self.cols} images")
            if len(self.paths) > MAX_STITCH_TILES:
                raise ValueError(f"Please use {MAX_STITCH_TILES} tiles or fewer")
            tiles = []
            for i, path in enumerate(self.paths):
                if self.isInterruptionRequested():
                    raise InterruptedError()
                encoded = np.fromfile(str(path), dtype=np.uint8)
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Could not read {Path(path).name}")
                tiles.append(image)
                self.progress.emit(i + 1, len(self.paths) + self.rows * (self.cols - 1) + (self.rows - 1) * self.cols)
            mosaic, warnings = stitch_grid(
                tiles, self.rows, self.cols, self.overlap,
                cancelled=self.isInterruptionRequested,
                progress=lambda i, total: self.progress.emit(len(tiles) + i, len(tiles) + total))
            self.done.emit(mosaic, warnings)
        except InterruptedError:
            return
        except Exception as exc:
            logging.exception("stitching failed")
            self.failed.emit(str(exc))


'''

METHODS = r'''    # ---------------- stitching ----------------
    def build_stitch_tab(self):
        tab = QWidget()
        v = QVBoxLayout(tab)
        hint = QLabel("Stitch focus-stacked tiles in reading order: left to right, then down. "
                      "Use a 2x2 or 3x3 grid, overlap neighbouring images by 20-40%, lock "
                      "exposure, and avoid rotating the specimen. This is a best-fit result; "
                      "check seams before relying on measurements.")
        hint.setWordWrap(True)
        v.addWidget(hint)
        self.stitch_list = QListWidget()
        self.stitch_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.stitch_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.stitch_list.setToolTip("Images are stitched in this order; drag to rearrange")
        v.addWidget(self.stitch_list, 1)
        row = QHBoxLayout()
        add = QPushButton("Add images...")
        add.clicked.connect(lambda _=False: self.add_stitch_images())
        remove = QPushButton("Remove selected")
        remove.clicked.connect(lambda _=False: [self.stitch_list.takeItem(self.stitch_list.row(i))
                                                 for i in self.stitch_list.selectedItems()])
        clear = QPushButton("Clear")
        clear.clicked.connect(lambda _=False: self.stitch_list.clear())
        for button in (add, remove, clear):
            row.addWidget(button)
        v.addLayout(row)
        form = QFormLayout()
        self.stitch_rows = QSpinBox()
        self.stitch_rows.setRange(1, 4)
        self.stitch_rows.setValue(2)
        self.stitch_cols = QSpinBox()
        self.stitch_cols.setRange(1, 4)
        self.stitch_cols.setValue(2)
        self.stitch_overlap = QSpinBox()
        self.stitch_overlap.setRange(10, 60)
        self.stitch_overlap.setValue(25)
        self.stitch_overlap.setSuffix(" %")
        form.addRow("Rows", self.stitch_rows)
        form.addRow("Columns", self.stitch_cols)
        form.addRow("Minimum overlap", self.stitch_overlap)
        v.addLayout(form)
        self.stitch_warning = QLabel("")
        self.stitch_warning.setWordWrap(True)
        v.addWidget(self.stitch_warning)
        self.stitch_progress = QProgressBar()
        self.stitch_progress.hide()
        v.addWidget(self.stitch_progress)
        self.stitch_btn = QPushButton("Stitch")
        self.stitch_btn.clicked.connect(lambda _=False: self.start_stitch())
        v.addWidget(self.stitch_btn)
        return tab

    def add_stitch_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add tile images", str(LIBRARY),
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        existing = {self.stitch_list.item(i).data(Qt.UserRole)
                    for i in range(self.stitch_list.count())}
        for path in paths:
            if path not in existing:
                item = QListWidgetItem(Path(path).name)
                item.setData(Qt.UserRole, path)
                item.setToolTip(path)
                self.stitch_list.addItem(item)
                existing.add(path)

    def start_stitch(self):
        paths = [self.stitch_list.item(i).data(Qt.UserRole)
                 for i in range(self.stitch_list.count())]
        rows, cols = self.stitch_rows.value(), self.stitch_cols.value()
        if len(paths) != rows * cols:
            self.stitch_warning.setText(f"Select exactly {rows * cols} tiles for this grid.")
            return
        if len(paths) > MAX_STITCH_TILES:
            self.stitch_warning.setText(f"Use {MAX_STITCH_TILES} tiles or fewer.")
            return
        if getattr(self, "stitch_thread", None) and self.stitch_thread.isRunning():
            return
        self.stitch_paths = paths
        self.stitch_btn.setEnabled(False)
        self.stitch_warning.clear()
        self.stitch_progress.setValue(0)
        self.stitch_progress.setMaximum(len(paths) + rows * (cols - 1) + (rows - 1) * cols)
        self.stitch_progress.show()
        self.stitch_thread = StitchThread(paths, rows, cols, self.stitch_overlap.value() / 100)
        self.stitch_thread.progress.connect(lambda i, _n: self.stitch_progress.setValue(i))
        self.stitch_thread.done.connect(self.on_stitch_done)
        self.stitch_thread.failed.connect(self.on_stitch_failed)
        self.stitch_thread.start()
        self.status.setText("Stitching tiles...")

    @Slot(object, object)
    def on_stitch_done(self, mosaic, warnings):
        self.stitch_btn.setEnabled(True)
        self.stitch_progress.hide()
        h, w = mosaic.shape[:2]
        text = f"Stitched {len(self.stitch_paths)} tiles into {w}x{h}. Check the seams."
        if warnings:
            text += " Warnings: " + "; ".join(warnings[:5])
        self.stitch_warning.setText(text)
        self.cal = None  # calibration is not inherited: tile scales may differ
        self.update_cal_label()
        self._last_stitch_result = mosaic
        self._last_stitch_text = text
        self.show_still(mosaic, text)
        try:
            folder = day_folder("stitched")
            folder.mkdir(parents=True, exist_ok=True)
            path = unique_path(folder, "stitched")
        except OSError as exc:
            self.status.setText(f"Stitched, but could not prepare save: {exc}")
            return
        worker = SaveThread(path, mosaic)
        self._track_saver(worker)
        worker.done.connect(lambda saved: self.status.setText(f"{self._last_stitch_text} Saved {saved}"))
        worker.failed.connect(lambda err: self.status.setText(f"{self._last_stitch_text} Save failed: {err}"))
        worker.start()

    @Slot(str)
    def on_stitch_failed(self, text):
        self.stitch_btn.setEnabled(True)
        self.stitch_progress.hide()
        self.stitch_warning.setText(f"Stitching failed: {text}")
        self.status.setText(f"Stitching failed: {text}")

'''

EDITS = [
    ("# ======================================================================\n# V4L2 controls: parsing, and a worker thread that runs v4l2-ctl\n",
     FUNCTIONS + "# ======================================================================\n# V4L2 controls: parsing, and a worker thread that runs v4l2-ctl\n"),
    ("class ThumbThread(QThread):\n", THREAD + "class ThumbThread(QThread):\n"),
    ('        self.tabs.addTab(self.build_stack_tab(), "Stack")\n',
     '        self.tabs.addTab(self.build_stack_tab(), "Stack")\n        self.tabs.addTab(self.build_stitch_tab(), "Stitch")\n'),
    ("    def closeEvent(self, event):\n", METHODS + "    def closeEvent(self, event):\n"),
    ("        super().closeEvent(event)\n",
     "        if getattr(self, \"stitch_thread\", None) and self.stitch_thread.isRunning():\n"
     "            self.stitch_thread.requestInterruption()\n"
     "            self.stitch_thread.wait()\n"
     "        super().closeEvent(event)\n"),
]


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python3 add_stitching.py Microsco.py")
    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")
    if "def stitch_grid(" in text:
        sys.exit("Stitching is already present. Nothing to do.")
    for anchor, _ in EDITS:
        count = text.count(anchor)
        if count != 1:
            sys.exit(f"Cannot patch safely: expected one anchor, found {count}:\\n{anchor}")
    backup = path.with_name(path.name + ".bak")
    shutil.copy2(path, backup)
    for anchor, replacement in EDITS:
        text = text.replace(anchor, replacement, 1)
    path.write_text(text, encoding="utf-8")
    print(f"Added the optional Stitch tab to {path}; backup saved as {backup}")


if __name__ == "__main__":
    main()
