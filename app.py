#!/usr/bin/env python3
"""Microscope viewer (single file).

Live preview and capture, camera controls, calibrated measurement on frozen or opened
images, and auto-captured focus stacks.

Needs (inside the container): python3, PySide6, opencv-python-headless, numpy, v4l-utils
Usage:  python3 app.py [/dev/videoN] [--size WIDTHxHEIGHT]
Keys:   C capture, F freeze/live, S start/stop auto-capture, [ ] previous/next library image
"""
import argparse
import logging
import math
import os
import queue
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QFile, QRectF, QSize, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import (QColor, QIcon, QImage, QKeySequence, QPainter, QPixmap,
                           QShortcut)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox,
                               QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QInputDialog, QLabel, QListView,
                               QListWidget, QListWidgetItem, QMainWindow, QProgressBar,
                               QPushButton, QRadioButton, QScrollArea, QSlider,
                               QSpinBox, QTabWidget, QToolButton, QVBoxLayout,
                               QWidget)

FONT = cv2.FONT_HERSHEY_SIMPLEX
MAX_STACK = 60            # frames are held in memory, about 5.5 MB each at 1600x1200
LIBRARY = Path.home() / "Pictures" / "Microscope"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
THUMB = 160               # thumbnail size (longest side) in the Library tab
TUNE_TARGET = 115         # average brightness (0-255) that auto-tune aims for
TUNE_TOLERANCE = 10       # close enough, either side of the target


# ======================================================================
# Files and folders
# ======================================================================
def day_folder(sub):
    return LIBRARY / datetime.now().strftime("%Y-%m-%d") / sub


def unique_path(folder, stem):
    stamp = datetime.now().strftime("%H%M%S-%f")
    path = folder / f"{stem}-{stamp}.png"
    n = 1
    while path.exists():
        path = folder / f"{stem}-{stamp}-{n}.png"
        n += 1
    return path


def save_png(path, frame):
    """Atomic PNG write; raises OSError on any failure."""
    try:
        ok, buf = cv2.imencode(".png", frame)
    except cv2.error as exc:
        raise OSError(f"PNG encoding failed: {exc}")
    if not ok:
        raise OSError("PNG encoding failed")
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(buf.tobytes())
    os.replace(tmp, path)


def library_days(root=LIBRARY):
    """Date folders, newest first."""
    if not root.is_dir():
        return []
    return sorted((p.name for p in root.iterdir()
                   if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name)), reverse=True)


def library_files(day_dir):
    """Images saved on one day, newest first. Raw stack frames (frame-*) are left out."""
    files = [p for p in day_dir.rglob("*")
             if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("frame-")]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def thumb_label(name):
    """'photo-145055-993858.png' -> 'photo' and the time on two lines."""
    m = re.match(r"(.+?)-(\d{2})(\d{2})(\d{2})-\d", name)
    return f"{m.group(1)}\n{m.group(2)}:{m.group(3)}:{m.group(4)}" if m else name


# ======================================================================
# Measurement helpers
# ======================================================================
def fmt_length(px, ppu, unit):
    return f"{px / ppu:.4g} {unit}" if ppu else f"{px:.1f} px"


def parse_length(text):
    """'5 mm' -> (5.0, 'mm'). None if it is not a positive number plus an optional unit."""
    m = re.fullmatch(r"\s*(\d*\.?\d+)\s*(\S*)\s*", text)
    if not m or float(m.group(1)) <= 0:
        return None
    return float(m.group(1)), m.group(2) or "mm"


def nice_length(target):
    """Largest 1/2/5 x 10^n that is <= target."""
    exp = math.floor(math.log10(target))
    for m in (5, 2, 1):
        if m * 10 ** exp <= target:
            return m * 10 ** exp
    return 10 ** exp


def compose(frame, ppu, unit, lines, scale_bar):
    """Copy of `frame` with measurement lines and an optional scale bar drawn on it."""
    img = frame.copy()
    h, w = img.shape[:2]
    t = max(1, round(w / 700))
    fs = max(0.5, w / 1400)
    unit = unit.replace("\u00b5", "u")           # Hershey fonts have no micro sign

    def label(text, org, color):
        cv2.putText(img, text, org, FONT, fs, (0, 0, 0), t + 2, cv2.LINE_AA)
        cv2.putText(img, text, org, FONT, fs, color, t, cv2.LINE_AA)

    for p1, p2 in lines:
        a = (int(round(p1[0])), int(round(p1[1])))
        b = (int(round(p2[0])), int(round(p2[1])))
        cv2.line(img, a, b, (0, 255, 255), t, cv2.LINE_AA)
        for p in (a, b):
            cv2.circle(img, p, t + 1, (0, 255, 255), -1, cv2.LINE_AA)
        text = fmt_length(math.hypot(p2[0] - p1[0], p2[1] - p1[1]), ppu, unit)
        label(text, ((a[0] + b[0]) // 2 + 8, (a[1] + b[1]) // 2 - 8), (0, 255, 255))

    if scale_bar and ppu:
        length = nice_length(w * 0.2 / ppu)
        bar = max(2, int(round(length * ppu)))
        margin = int(w * 0.03)
        y = h - margin
        cv2.rectangle(img, (margin - 2, y - 2 * t - 2), (margin + bar + 2, y + 2), (0, 0, 0), -1)
        cv2.rectangle(img, (margin, y - 2 * t), (margin + bar, y), (255, 255, 255), -1)
        label(f"{length:g} {unit}", (margin, y - 2 * t - 10), (255, 255, 255))
    return img


# ======================================================================
# Stacking (pure OpenCV)
# ======================================================================
def align_to(ref_gray, frame):
    """Warp `frame` onto the reference (shift, rotation, slight scale). Falls back to the
    unaligned frame if the optimiser cannot converge."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    s = 0.5
    a = cv2.resize(ref_gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    b = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
    try:
        _, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_AFFINE, criteria, None, 5)
    except cv2.error:
        return frame
    warp[:, 2] /= s                                 # translation back to full resolution
    h, w = frame.shape[:2]
    return cv2.warpAffine(frame, warp, (w, h),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def stack_frames(frames, align=True, mode="focus", progress=None):
    """mode 'focus': per-pixel sharpest frame (extended depth of field).
    mode 'average': mean of frames (noise reduction for a static subject)."""
    n = len(frames)
    total = n * (3 if mode == "focus" else 2)
    done = 0

    def tick():
        nonlocal done
        done += 1
        if progress:
            progress(done, total)

    ref = n // 2                                     # middle frame: smallest scale change
    ref_gray = cv2.cvtColor(frames[ref], cv2.COLOR_BGR2GRAY)
    aligned = []
    for i, f in enumerate(frames):
        aligned.append(align_to(ref_gray, f) if align and i != ref else f)
        tick()

    if mode == "average":
        acc = np.zeros(aligned[0].shape, np.float32)
        for f in aligned:
            acc += f
            tick()
        return np.clip(acc / n, 0, 255).astype(np.uint8)

    best = None
    index = np.zeros(aligned[0].shape[:2], np.int32)
    for i, f in enumerate(aligned):
        g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        sharp = cv2.GaussianBlur(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)), (0, 0), 3)
        if best is None:
            best = sharp
        else:
            better = sharp > best
            best[better] = sharp[better]
            index[better] = i
        tick()

    h, w = index.shape                               # soft-blend the choice to hide seams
    num = np.zeros((h, w, 3), np.float32)
    den = np.zeros((h, w), np.float32)
    for i, f in enumerate(aligned):
        weight = cv2.GaussianBlur((index == i).astype(np.float32), (0, 0), 4)
        num += f.astype(np.float32) * weight[..., None]
        den += weight
        tick()
    return np.clip(num / np.maximum(den, 1e-6)[..., None], 0, 255).astype(np.uint8)


# ======================================================================
# V4L2 controls: parsing, and a worker thread that runs v4l2-ctl
# ======================================================================
CTRL_RE = re.compile(r"^\s*(\w+)\s+0x[0-9a-fA-F]+\s+\((\w+)\)\s*:\s*(.*)$")
MENU_RE = re.compile(r"^\s+(\d+):\s*(.+?)\s*$")
PARAM_RE = re.compile(r"\b(min|max|step|default|value)=(-?\d+)")
AUTO_EXPOSURE_NAMES = ("auto_exposure", "exposure_auto")   # name differs by kernel
MANUAL_EXPOSURE = 1                                         # fallback menu value
EXPOSURE_TIME_NAMES = ("exposure_time_absolute", "exposure_absolute")


def is_auto_control(c):
    """Exposure mode or auto white balance: the controls that make other controls inactive."""
    return c.name in AUTO_EXPOSURE_NAMES or (
        c.kind == "bool" and "white_balance" in c.name and "auto" in c.name)


@dataclass
class Control:
    name: str
    kind: str                  # int, bool or menu
    section: str
    min: int = 0
    max: int = 1
    step: int = 1
    default: int = 0
    value: int = 0
    inactive: bool = False
    menu: dict = field(default_factory=dict)


def parse_controls(text):
    """Parse the output of `v4l2-ctl --list-ctrls-menus`."""
    controls, section, current = [], "Controls", None
    for line in text.splitlines():
        m = CTRL_RE.match(line)
        if m:
            name, kind, rest = m.groups()
            c = Control(name, kind, section)
            for key, val in PARAM_RE.findall(rest):
                setattr(c, key, int(val))
            c.inactive = "flags=" in rest and "inactive" in rest.split("flags=", 1)[1]
            if c.kind == "int" and c.min == 0 and c.max == 1:
                c.kind = "bool"                      # a 0..1 slider is really a checkbox
            controls.append(c)
            current = c
        elif line.strip() and not line[0].isspace():
            section, current = line.strip(), None    # a section header
        else:
            mm = MENU_RE.match(line)
            if mm and current is not None and current.kind == "menu":
                current.menu[int(mm.group(1))] = mm.group(2)
            elif line.strip():
                current = None                       # don't leak menu items to a stale control
    return [c for c in controls if c.kind in ("int", "bool", "menu")]


def list_controls(device):
    out = subprocess.run(["v4l2-ctl", "-d", device, "--list-ctrls-menus"],
                         capture_output=True, text=True, timeout=5, check=True).stdout
    return parse_controls(out)


def set_control(device, name, value):
    """Returns (ok, error_text)."""
    try:
        r = subprocess.run(["v4l2-ctl", "-d", device, f"--set-ctrl={name}={value}"],
                           capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    return r.returncode == 0, r.stderr.strip()


class V4L2Worker(QThread):
    """Runs v4l2-ctl jobs one at a time, so the GUI never waits on a subprocess."""
    listed = Signal(object)      # list[Control]
    message = Signal(str)
    failed = Signal(str)

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.jobs = queue.Queue()

    def post(self, *job):
        self.jobs.put(job)

    def stop(self):
        self.post("quit")
        self.wait(3000)

    def run(self):
        while True:
            job = self.jobs.get()
            kind = job[0]
            if kind == "quit":
                return
            try:
                if kind == "list":
                    try:
                        self.listed.emit(list_controls(self.device))
                    except (OSError, subprocess.SubprocessError) as exc:
                        self.failed.emit(str(exc))
                elif kind == "set":
                    _, name, value, refresh = job
                    ok, err = set_control(self.device, name, value)
                    self.message.emit(f"{name} = {value}" if ok else f"Could not set {name}: {err}")
                    if refresh:
                        self.listed.emit(list_controls(self.device))
                elif kind == "lock":
                    self.lock_exposure()
                elif kind == "reset":
                    self.reset_defaults()
            except (OSError, subprocess.SubprocessError) as exc:
                self.message.emit(f"v4l2-ctl error: {exc}")

    def lock_exposure(self):
        controls = list_controls(self.device)
        errors = []
        exp = next((c for c in controls if c.name in AUTO_EXPOSURE_NAMES), None)
        if exp is None:
            errors.append("no auto-exposure control")
        else:
            value = 0 if exp.kind == "bool" else next(
                (k for k, lbl in exp.menu.items() if "manual" in lbl.lower()), MANUAL_EXPOSURE)
            ok, err = set_control(self.device, exp.name, value)
            if not ok:
                errors.append(f"{exp.name}: {err}")
        wb = next((c for c in controls if c.kind == "bool"
                   and "white_balance" in c.name and "auto" in c.name), None)
        if wb is None:
            errors.append("no auto-white-balance control")
        else:
            ok, err = set_control(self.device, wb.name, 0)
            if not ok:
                errors.append(f"{wb.name}: {err}")
        self.message.emit("Exposure and white balance set to manual" if not errors
                          else "Partially applied: " + "; ".join(errors))
        self.listed.emit(list_controls(self.device))

    def reset_defaults(self):
        errors = []
        for _ in range(2):          # twice: some controls only unlock after others change
            errors = []
            controls = list_controls(self.device)
            # manual values first (while they are still active), then the automatic modes
            controls.sort(key=is_auto_control)
            for c in controls:
                if not c.inactive:
                    ok, err = set_control(self.device, c.name, c.default)
                    if not ok:
                        errors.append(f"{c.name}: {err}")
        self.message.emit("All controls reset to defaults" if not errors
                          else "Reset with errors: " + "; ".join(errors[:3]))
        self.listed.emit(list_controls(self.device))


class ControlsPanel(QWidget):
    """Widgets generated from the camera's own control list."""
    message = Signal(str)
    tune_requested = Signal()

    def __init__(self, device):
        super().__init__()
        self.entries = {}          # name -> dict(ctrl, input, label, container, timer, reset, row)
        self.setMinimumWidth(330)

        outer = QVBoxLayout(self)
        row = QHBoxLayout()
        tune_btn = QPushButton("Auto-tune exposure")
        tune_btn.setToolTip("Finds an exposure that gives a well-lit image, then locks it")
        lock_btn = QPushButton("Lock exposure")
        lock_btn.setToolTip("Manual exposure and manual white balance, so stack frames match")
        row.addWidget(tune_btn)
        row.addWidget(lock_btn)
        outer.addLayout(row)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(QLabel("Loading controls..."))
        outer.addWidget(self.scroll, 1)
        reset_btn = QPushButton("Reset all to defaults")
        reset_btn.setToolTip("Restores every setting, including automatic exposure and white balance")
        outer.addWidget(reset_btn)

        self.worker = V4L2Worker(device)
        self.worker.listed.connect(self.on_listed)
        self.worker.failed.connect(self.on_failed)
        self.worker.message.connect(self.message)
        tune_btn.clicked.connect(lambda _=False: self.tune_requested.emit())
        lock_btn.clicked.connect(lambda _=False: self.lock_exposure())
        reset_btn.clicked.connect(lambda _=False: self.worker.post("reset"))
        self.worker.start()
        self.worker.post("list")

    def shutdown(self):
        self.worker.stop()

    # ---- used by the main window (auto-tune) ----
    def lock_exposure(self):
        self.worker.post("lock")

    def set_control(self, name, value):
        self.worker.post("set", name, value, True)

    def exposure_control(self):
        for name in EXPOSURE_TIME_NAMES:
            if name in self.entries:
                return self.entries[name]["ctrl"]
        return None

    # ---- building and updating ----
    @Slot(str)
    def on_failed(self, text):
        self.scroll.setWidget(QLabel(
            f"Could not read controls:\n{text}\n\nIs v4l-utils installed in this container?"))

    @Slot(object)
    def on_listed(self, controls):
        if not controls:
            self.scroll.setWidget(QLabel("No adjustable controls found."))
            return
        if not self.entries:
            self.build(controls)
        self.update_values(controls)

    def build(self, controls):
        body = QWidget()
        vbox = QVBoxLayout(body)
        forms = {}
        for c in controls:
            if c.section not in forms:
                box = QGroupBox(c.section)
                forms[c.section] = QFormLayout(box)
                vbox.addWidget(box)
            entry = self.make_entry(c)
            forms[c.section].addRow(c.name.replace("_", " ").title(), entry["row"])
            self.entries[c.name] = entry
        vbox.addStretch(1)
        self.scroll.setWidget(body)

    @staticmethod
    def default_text(c):
        if c.kind == "menu":
            return c.menu.get(c.default, str(c.default))
        if c.kind == "bool":
            return "on" if c.default else "off"
        return str(c.default)

    def make_entry(self, c):
        """The input widget plus a small reset button that is only live when the value differs."""
        entry = self.make_input(c)
        reset = QToolButton()
        reset.setText("\u21ba")
        reset.setToolTip(f"Reset to default ({self.default_text(c)})")
        reset.clicked.connect(lambda _=False, n=c.name: self.reset_one(n))
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(entry["container"], 1)
        h.addWidget(reset)
        entry["reset"], entry["row"] = reset, row
        return entry

    def make_input(self, c):
        name = c.name
        if c.kind == "bool":
            box = QCheckBox()
            box.toggled.connect(lambda on, n=name: self.apply(n, int(on)))
            return {"ctrl": c, "input": box, "label": None, "container": box, "timer": None}

        if c.kind == "menu":
            combo = QComboBox()
            for key in sorted(c.menu):
                combo.addItem(c.menu[key], key)
            combo.currentIndexChanged.connect(
                lambda i, n=name, cb=combo: self.apply(n, cb.itemData(i)))
            return {"ctrl": c, "input": combo, "label": None, "container": combo, "timer": None}

        slider = QSlider(Qt.Horizontal)
        slider.setRange(c.min, c.max)
        slider.setSingleStep(max(c.step, 1))
        label = QLabel(str(c.value))
        label.setMinimumWidth(42)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(150)                    # debounce: apply 150 ms after the last change
        timer.timeout.connect(lambda n=name, s=slider: self.apply(n, s.value()))

        def on_value(v, lb=label, t=timer):
            lb.setText(str(v))
            t.start()
        slider.valueChanged.connect(on_value)

        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(slider, 1)
        h.addWidget(label)
        return {"ctrl": c, "input": slider, "label": label, "container": container, "timer": timer}

    def apply(self, name, value):
        c = self.entries[name]["ctrl"]
        # bool/menu changes can grey out or unlock other controls, so re-read afterwards
        self.worker.post("set", name, value, c.kind != "int")

    def reset_one(self, name):
        self.worker.post("set", name, self.entries[name]["ctrl"].default, True)

    def update_values(self, controls):
        fresh = {c.name: c for c in controls}
        for name, entry in self.entries.items():
            c = fresh.get(name)
            if c is None:
                continue
            entry["ctrl"] = c
            inp, timer = entry["input"], entry["timer"]
            busy = timer is not None and (timer.isActive() or inp.isSliderDown())
            if not busy:                          # never fight the user's own drag
                inp.blockSignals(True)
                if c.kind == "bool":
                    inp.setChecked(bool(c.value))
                elif c.kind == "menu":
                    idx = inp.findData(c.value)
                    if idx >= 0:
                        inp.setCurrentIndex(idx)
                else:
                    inp.setValue(c.value)
                    entry["label"].setText(str(c.value))
                inp.blockSignals(False)
            entry["container"].setEnabled(not c.inactive)
            entry["reset"].setEnabled(not c.inactive and c.value != c.default)


# ======================================================================
# Background threads
# ======================================================================
class CameraThread(QThread):
    """Reads frames off the GUI thread so the window never freezes."""
    frame_ready = Signal(object)   # BGR numpy array
    mode_known = Signal(str)       # negotiated "WxH"
    error = Signal(str)

    def __init__(self, device, width, height):
        super().__init__()
        self.device, self.width, self.height = device, width, height
        self._pending = False      # True while the GUI still owes us a frame

    def frame_consumed(self):
        self._pending = False

    def run(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            self.error.emit(f"Could not open {self.device}. Is guvcview still running?")
            return
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.mode_known.emit(f"{w}x{h}" if w > 0 and h > 0 else "unknown size")

            failures = 0
            while not self.isInterruptionRequested():
                ok, frame = cap.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= 30:
                        self.error.emit(f"{self.device} stopped delivering frames")
                        break
                    self.msleep(100)
                    continue
                failures = 0
                if not self._pending:          # drop frames if the GUI is behind
                    self._pending = True
                    self.frame_ready.emit(frame)
        except Exception as exc:
            logging.exception("camera thread failed")
            self.error.emit(f"Camera error: {exc}")
        finally:
            cap.release()

    def stop(self):
        self.requestInterruption()
        if not self.wait(3000):
            # Only reached if cap.read() is blocked on a misbehaving device.
            logging.warning("camera thread did not stop in time; terminating")
            self.terminate()
            self.wait(1000)


class StackThread(QThread):
    progress = Signal(int, int)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, frames, align, mode):
        super().__init__()
        self.frames, self.align, self.mode = frames, align, mode

    def run(self):
        try:
            result = stack_frames(self.frames, self.align, self.mode,
                                  lambda i, n: self.progress.emit(i, n))
            self.done.emit(result)
        except Exception as exc:
            logging.exception("stacking failed")
            self.failed.emit(str(exc))


class ThumbThread(QThread):
    """Builds Library thumbnails one by one, so the tab fills in without freezing."""
    thumb = Signal(int, int, object)       # generation, row, small BGR array

    def __init__(self, paths, gen):
        super().__init__()
        self.paths, self.gen = paths, gen

    def run(self):
        for row, path in enumerate(self.paths):
            if self.isInterruptionRequested():
                return
            try:
                img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            except (OSError, cv2.error):
                img = None
            if img is None:
                continue
            scale = THUMB / max(img.shape[:2])
            self.thumb.emit(self.gen, row, cv2.resize(img, None, fx=scale, fy=scale,
                                                      interpolation=cv2.INTER_AREA))


# ======================================================================
# Preview widget: paints the image, reports mouse positions in image pixels
# ======================================================================
class PreviewView(QWidget):
    pressed = Signal(float, float)
    moved = Signal(float, float)
    released = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 240)
        self._frame = None         # keeps the numpy memory alive for the QImage
        self._img = None
        self._text = "Waiting for camera..."

    def set_frame(self, frame):
        h, w = frame.shape[:2]
        self._frame = frame
        self._img = QImage(frame.data, w, h, frame.strides[0], QImage.Format_BGR888)
        self.update()

    def set_text(self, text):
        self._frame = self._img = None
        self._text = text
        self.update()

    def _fit(self):
        if self._img is None:
            return None
        iw, ih = self._img.width(), self._img.height()
        s = min(self.width() / iw, self.height() / ih)
        w, h = iw * s, ih * s
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h), s

    def paintEvent(self, event):
        p = QPainter(self)
        try:
            p.fillRect(self.rect(), QColor("#111111"))
            fit = self._fit()
            if fit is None:
                p.setPen(QColor("#aaaaaa"))
                p.drawText(self.rect(), Qt.AlignCenter, self._text)
                return
            p.setRenderHint(QPainter.SmoothPixmapTransform, True)
            p.drawImage(fit[0], self._img)
        finally:
            p.end()

    def _to_image(self, pos):
        fit = self._fit()
        if fit is None:
            return None
        rect, s = fit
        x = min(max((pos.x() - rect.x()) / s, 0), self._img.width() - 1)
        y = min(max((pos.y() - rect.y()) / s, 0), self._img.height() - 1)
        return x, y

    def mousePressEvent(self, e):
        pt = self._to_image(e.position()) if e.button() == Qt.LeftButton else None
        if pt:
            self.pressed.emit(*pt)

    def mouseMoveEvent(self, e):
        pt = self._to_image(e.position())
        if pt:
            self.moved.emit(*pt)

    def mouseReleaseEvent(self, e):
        pt = self._to_image(e.position()) if e.button() == Qt.LeftButton else None
        if pt:
            self.released.emit(*pt)


# ======================================================================
# Main window
# ======================================================================
class MainWindow(QMainWindow):
    def __init__(self, device, width, height):
        super().__init__()
        self.setWindowTitle(f"Microscope - {device}")
        self.requested = f"{width}x{height}"
        self.latest = None            # newest live frame
        self.still = None             # frozen or opened image (measuring happens here)
        self.camera_failed = False
        self.cal = None               # {"ppu": px per unit, "unit": str, "width": image width}
        self.measures = []            # [((x1, y1), (x2, y2)), ...] in image pixels
        self.drag = None
        self.scale_bar_on = False
        self.stack_frames, self.stack_dir, self.stack_thread = [], None, None
        self.frame_counter = 0        # counts live frames (detects a stalled camera)
        self.auto_state = "idle"      # idle | delay | running
        self.auto_next = self.auto_start = 0.0
        self.auto_last_frame = -1
        self.thumb_thread, self.thumb_gen = None, 0
        self.tune_state = "idle"      # idle | locking | measuring
        self.tune_exp = self.tune_iter = self.tune_frames = 0
        self.tune_until = self.tune_started = 0.0

        self.view = PreviewView()
        self.view.pressed.connect(self.on_press)
        self.view.moved.connect(self.on_move)
        self.view.released.connect(self.on_release)

        self.capture_btn = QPushButton("Capture (C)")
        self.freeze_btn = QPushButton("Freeze (F)")
        self.open_btn = QPushButton("Open image...")
        for b in (self.capture_btn, self.freeze_btn, self.open_btn):
            b.setFocusPolicy(Qt.NoFocus)
        self.capture_btn.setEnabled(False)
        self.freeze_btn.setEnabled(False)
        row = QHBoxLayout()
        for b in (self.capture_btn, self.freeze_btn, self.open_btn):
            row.addWidget(b)
        self.status = QLabel("Starting...")

        layout = QVBoxLayout()
        layout.addWidget(self.view, 1)
        layout.addLayout(row)
        layout.addWidget(self.status)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.resize(1300, 780)

        self.controls = ControlsPanel(device)
        self.controls.message.connect(self.status.setText)
        self.controls.tune_requested.connect(self.start_tune)
        self.tune_timer = QTimer(self)
        self.tune_timer.setInterval(100)
        self.tune_timer.timeout.connect(self.tune_tick)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.controls, "Camera")
        self.tabs.addTab(self.build_measure_tab(), "Measure")
        self.tabs.addTab(self.build_stack_tab(), "Stack")
        self.tabs.addTab(self.build_library_tab(), "Library")
        self.tabs.currentChanged.connect(self.on_tab_changed)
        dock = QDockWidget("Tools", self)
        dock.setWidget(self.tabs)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self.capture_btn.clicked.connect(self.capture)
        self.freeze_btn.clicked.connect(self.toggle_freeze)
        self.open_btn.clicked.connect(self.open_image)
        for key, fn in (("C", self.capture), ("F", self.toggle_freeze), ("S", self.toggle_auto),
                        ("[", lambda: self.step_image(-1)), ("]", lambda: self.step_image(1))):
            sc = QShortcut(QKeySequence(key), self)
            sc.setAutoRepeat(False)
            sc.activated.connect(fn)

        self.cam = CameraThread(device, width, height)
        self.cam.frame_ready.connect(self.show_frame)
        self.cam.mode_known.connect(self.on_mode)
        self.cam.error.connect(self.on_camera_error)
        self.cam.start()

    # ---------------- tabs ----------------
    def build_measure_tab(self):
        tab = QWidget()
        v = QVBoxLayout(tab)
        hint = QLabel("Freeze the live image (F) or open an image, then drag across it to measure.")
        hint.setWordWrap(True)
        v.addWidget(hint)

        row = QHBoxLayout()
        self.measure_rb = QRadioButton("Measure")
        self.calibrate_rb = QRadioButton("Calibrate (draw along a ruler)")
        self.measure_rb.setChecked(True)
        row.addWidget(self.measure_rb)
        row.addWidget(self.calibrate_rb)
        v.addLayout(row)

        self.cal_label = QLabel()
        self.cal_label.setWordWrap(True)
        v.addWidget(self.cal_label)
        self.update_cal_label()
        bar = QCheckBox("Scale bar")
        bar.toggled.connect(self.on_scale_bar)
        v.addWidget(bar)

        self.measure_list = QListWidget()
        v.addWidget(self.measure_list, 1)
        btns = QHBoxLayout()
        undo = QPushButton("Undo")
        undo.clicked.connect(lambda _=False: self.undo_measure())
        clear = QPushButton("Clear")
        clear.clicked.connect(lambda _=False: self.clear_measures())
        btns.addWidget(undo)
        btns.addWidget(clear)
        v.addLayout(btns)
        ann = QPushButton("Save annotated image")
        ann.clicked.connect(lambda _=False: self.save_annotated())
        v.addWidget(ann)
        return tab

    def build_library_tab(self):
        tab = QWidget()
        v = QVBoxLayout(tab)
        hint = QLabel("Double-click an image to open it, then measure it. [ and ] step through "
                      "the list. Raw stack frames are hidden; they stay in their stack folder.")
        hint.setWordWrap(True)
        v.addWidget(hint)

        row = QHBoxLayout()
        row.addWidget(QLabel("Day"))
        self.day_combo = QComboBox()
        self.day_combo.currentTextChanged.connect(lambda _t: self.refresh_library())
        row.addWidget(self.day_combo, 1)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(lambda _=False: self.reload_library())
        row.addWidget(refresh)
        v.addLayout(row)

        self.library = QListWidget()
        self.library.setViewMode(QListView.IconMode)
        self.library.setResizeMode(QListView.Adjust)
        self.library.setMovement(QListView.Static)
        self.library.setIconSize(QSize(THUMB, THUMB))
        # Fixed cells. Without this, items are placed before their thumbnails exist and
        # the grid never re-flows when the icons arrive, so they overlap.
        self.library.setGridSize(QSize(THUMB + 24, THUMB + 44))
        self.library.setUniformItemSizes(True)
        self.library.setWordWrap(True)
        self.library.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.library.itemDoubleClicked.connect(lambda item: self.load_image(item.data(Qt.UserRole)))
        v.addWidget(self.library, 1)
        trash = QPushButton("Move selected to Trash")
        trash.clicked.connect(lambda _=False: self.trash_selected())
        v.addWidget(trash)
        self.library_tab = tab
        self.refresh_days()
        return tab

    def build_stack_tab(self):
        tab = QWidget()
        v = QVBoxLayout(tab)
        hint = QLabel("Lock exposure on the Camera tab first. Turn the focus wheel a little "
                      "after each beep, and keep the stand still while a shot is taken.")
        hint.setWordWrap(True)
        v.addWidget(hint)

        auto = QGroupBox("Auto-capture")
        auto_form = QFormLayout(auto)
        self.auto_settings = QWidget()
        sf = QFormLayout(self.auto_settings)
        sf.setContentsMargins(0, 0, 0, 0)
        self.auto_interval = QDoubleSpinBox()
        self.auto_interval.setRange(0.5, 30.0)
        self.auto_interval.setSingleStep(0.5)
        self.auto_interval.setValue(2.0)
        self.auto_interval.setSuffix(" s")
        self.auto_max_shots = QSpinBox()
        self.auto_max_shots.setRange(2, MAX_STACK)
        self.auto_max_shots.setValue(20)
        self.auto_delay = QSpinBox()
        self.auto_delay.setRange(0, 30)
        self.auto_delay.setValue(3)
        self.auto_delay.setSuffix(" s")
        sf.addRow("Time between shots", self.auto_interval)
        sf.addRow("Number of shots", self.auto_max_shots)
        sf.addRow("Start delay", self.auto_delay)
        self.auto_btn = QPushButton("Start auto-capture (S)")
        self.auto_btn.clicked.connect(lambda _=False: self.toggle_auto())
        self.auto_label = QLabel("Idle")
        self.auto_label.setWordWrap(True)
        auto_form.addRow(self.auto_settings)
        auto_form.addRow(self.auto_btn)
        auto_form.addRow(self.auto_label)
        v.addWidget(auto)
        self.auto_timer = QTimer(self)
        self.auto_timer.setInterval(100)
        self.auto_timer.timeout.connect(self.auto_tick)

        proc = QGroupBox("Processing (runs automatically after auto-capture)")
        pv = QVBoxLayout(proc)
        self.align_box = QCheckBox("Align frames first")
        self.align_box.setChecked(True)
        self.stack_mode = QComboBox()
        self.stack_mode.addItems(["Focus stack (more depth of field)",
                                  "Average (reduce noise, static subject)"])
        self.process_btn = QPushButton("Process last group again")
        self.process_btn.clicked.connect(lambda _=False: self.process_stack())
        self.progress = QProgressBar()
        self.progress.hide()
        for w in (self.align_box, self.stack_mode, self.process_btn, self.progress):
            pv.addWidget(w)
        v.addWidget(proc)
        v.addStretch(1)
        return tab

    # ---------------- camera events ----------------
    @Slot(str)
    def on_mode(self, size):
        note = "" if size == self.requested else f" (requested {self.requested})"
        self.status.setText(f"Camera mode: {size}{note}")

    @Slot(str)
    def on_camera_error(self, msg):
        self.camera_failed = True
        self.latest = None
        self.capture_btn.setEnabled(self.still is not None)
        if self.still is None:
            self.view.set_text(msg)
            self.freeze_btn.setEnabled(False)
        self.status.setText(msg)

    @Slot(object)
    def show_frame(self, frame):
        try:
            if self.camera_failed:
                return
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            self.latest = frame
            self.frame_counter += 1
            self.capture_btn.setEnabled(True)
            self.freeze_btn.setEnabled(True)
            if self.still is None:
                self.refresh_view()
        finally:
            self.cam.frame_consumed()      # always release the slot, even on error

    # ---------------- what is on screen ----------------
    def current_base(self):
        return self.still if self.still is not None else self.latest

    def current_cal(self, width):
        """(pixels per unit, unit) for an image `width` px wide; calibration scales with width."""
        if not self.cal:
            return None, ""
        return self.cal["ppu"] * width / self.cal["width"], self.cal["unit"]

    def refresh_view(self):
        base = self.current_base()
        if base is None:
            return
        if self.still is not None:             # overlays only on frozen/opened images
            ppu, unit = self.current_cal(base.shape[1])
            lines = self.measures + ([self.drag] if self.drag else [])
            if lines or (self.scale_bar_on and ppu):
                base = compose(base, ppu, unit, lines, self.scale_bar_on)
        self.view.set_frame(base)

    def show_still(self, img, message):
        self.still = img
        self.measures, self.drag = [], None
        self.freeze_btn.setText("Resume live (F)")
        self.freeze_btn.setEnabled(True)
        self.capture_btn.setEnabled(True)
        self.view.setCursor(Qt.CrossCursor)
        self.update_measure_list()
        self.refresh_view()
        self.status.setText(message)

    def toggle_freeze(self):
        if self.still is not None:
            self.still, self.measures, self.drag = None, [], None
            self.freeze_btn.setText("Freeze (F)")
            self.view.setCursor(Qt.ArrowCursor)
            self.update_measure_list()
            self.refresh_view()
            self.status.setText("Live")
        elif self.latest is not None:
            self.show_still(self.latest.copy(), "Frozen. Drag on the image to measure.")

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", str(LIBRARY), "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        if path:
            self.load_image(path)

    def load_image(self, path):
        try:
            img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        except (OSError, cv2.error) as exc:
            self.status.setText(f"Could not open image: {exc}")
            return
        if img is None:
            self.status.setText("Could not read that image")
            return
        self.show_still(img, f"Opened {Path(path).name}")

    # ---------------- library ----------------
    def on_tab_changed(self, index):
        if self.tabs.widget(index) is self.library_tab:
            self.reload_library()          # pick up anything captured since last time

    def reload_library(self):
        self.refresh_days()
        self.refresh_library()

    def refresh_days(self):
        current = self.day_combo.currentText()
        days = library_days()
        self.day_combo.blockSignals(True)
        self.day_combo.clear()
        self.day_combo.addItems(days)
        if current in days:
            self.day_combo.setCurrentText(current)
        self.day_combo.blockSignals(False)

    def stop_thumbs(self):
        if self.thumb_thread is not None and self.thumb_thread.isRunning():
            self.thumb_thread.requestInterruption()
            self.thumb_thread.wait()       # at most one image decode

    def refresh_library(self):
        self.stop_thumbs()
        self.library.clear()
        day = self.day_combo.currentText()
        paths = library_files(LIBRARY / day) if day else []
        for p in paths:
            item = QListWidgetItem(thumb_label(p.name))
            item.setData(Qt.UserRole, str(p))
            item.setToolTip(str(p))
            self.library.addItem(item)
        self.thumb_gen += 1                # stale thumbnails from an older list are ignored
        self.thumb_thread = ThumbThread(paths, self.thumb_gen)
        self.thumb_thread.thumb.connect(self.on_thumb)
        self.thumb_thread.start()

    @Slot(int, int, object)
    def on_thumb(self, gen, row, arr):
        item = self.library.item(row)
        if gen != self.thumb_gen or item is None:
            return
        h, w = arr.shape[:2]
        img = QImage(arr.data, w, h, arr.strides[0], QImage.Format_BGR888)
        item.setIcon(QIcon(QPixmap.fromImage(img)))

    def step_image(self, delta):
        count = self.library.count()
        if count == 0:
            self.status.setText("Library is empty. Open the Library tab and pick a day.")
            return
        row = min(max(self.library.currentRow() + delta, 0), count - 1)
        self.library.setCurrentRow(row)
        self.load_image(self.library.item(row).data(Qt.UserRole))

    def trash_selected(self):
        """Moves files to the desktop Trash. Never deletes permanently."""
        items = self.library.selectedItems()
        if not items:
            return
        moved = 0
        for item in items:
            res = QFile.moveToTrash(item.data(Qt.UserRole))
            moved += bool(res[0] if isinstance(res, tuple) else res)
        self.status.setText(f"Moved {moved} of {len(items)} file(s) to Trash"
                            + ("" if moved else ". Trash is not available here."))
        self.refresh_library()

    def capture(self):
        base = self.current_base()
        if base is None:
            self.status.setText("Nothing to capture yet")
            return
        folder = day_folder("photos")
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path = unique_path(folder, "photo")
            save_png(path, base)               # raw image, without overlays
        except OSError as exc:
            self.status.setText(f"Save failed: {exc}")
            return
        self.status.setText(f"Saved {path}")

    # ---------------- measuring ----------------
    def on_press(self, x, y):
        if self.still is not None:
            self.drag = ((x, y), (x, y))
            self.refresh_view()

    def on_move(self, x, y):
        if self.drag:
            self.drag = (self.drag[0], (x, y))
            self.refresh_view()

    def on_release(self, x, y):
        if not self.drag:
            return
        p1, p2 = self.drag[0], (x, y)
        self.drag = None
        px = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if px >= 3:                            # ignore accidental clicks
            if self.calibrate_rb.isChecked():
                self.finish_calibration(px)
            else:
                self.measures.append((p1, p2))
                self.update_measure_list()
        self.refresh_view()

    def finish_calibration(self, px):
        text, ok = QInputDialog.getText(
            self, "Calibrate",
            f"The line is {px:.1f} px long.\nIts real length (for example 5 mm):", text="1 mm")
        if not ok:
            return
        parsed = parse_length(text)
        if parsed is None:
            self.status.setText("Could not read that length. Type a number and a unit, like 5 mm")
            return
        length, unit = parsed
        self.cal = {"ppu": px / length, "unit": unit, "width": self.still.shape[1]}
        self.measure_rb.setChecked(True)
        self.update_cal_label()
        self.update_measure_list()
        self.status.setText(f"Calibrated: {px / length:.2f} px per {unit}")

    def update_cal_label(self):
        self.cal_label.setText(
            f"{self.cal['ppu']:.2f} px per {self.cal['unit']}. Magnification changes with the "
            "focus wheel, so recalibrate after refocusing." if self.cal
            else "Not calibrated: lengths are shown in pixels.")

    def on_scale_bar(self, on):
        self.scale_bar_on = on
        self.refresh_view()

    def update_measure_list(self):
        base = self.current_base()
        ppu, unit = self.current_cal(base.shape[1]) if base is not None else (None, "")
        self.measure_list.clear()
        for i, (p1, p2) in enumerate(self.measures, 1):
            px = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            self.measure_list.addItem(f"{i}: {fmt_length(px, ppu, unit)}")

    def undo_measure(self):
        if self.measures:
            self.measures.pop()
            self.update_measure_list()
            self.refresh_view()

    def clear_measures(self):
        self.measures = []
        self.update_measure_list()
        self.refresh_view()

    def save_annotated(self):
        if self.still is None:
            self.status.setText("Freeze (F) or open an image first")
            return
        ppu, unit = self.current_cal(self.still.shape[1])
        folder = day_folder("annotated")
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path = unique_path(folder, "annotated")
            save_png(path, compose(self.still, ppu, unit, self.measures, self.scale_bar_on))
        except OSError as exc:
            self.status.setText(f"Save failed: {exc}")
            return
        self.status.setText(f"Saved {path}")

    # ---------------- exposure auto-tune ----------------
    def start_tune(self):
        """Switch to manual exposure, then nudge the exposure time until the live image
        has a sensible average brightness. The result stays locked, ready for a stack."""
        if self.tune_state != "idle":
            return
        if self.latest is None or self.camera_failed:
            self.status.setText("No live camera image to tune against")
            return
        if self.auto_state != "idle":
            self.status.setText("Wait for auto-capture to finish first")
            return
        if self.controls.exposure_control() is None:
            self.status.setText("This camera has no exposure-time control")
            return
        if self.still is not None:
            self.toggle_freeze()                   # tune against the live image
        self.controls.lock_exposure()
        now = time.monotonic()
        self.tune_state = "locking"
        self.tune_started, self.tune_until = now, now + 0.5
        self.tune_iter, self.tune_frames = 0, self.frame_counter
        self.tune_timer.start()
        self.status.setText("Tuning exposure...")

    def tune_tick(self):
        if self.tune_state == "idle":
            return
        exp = self.controls.exposure_control()
        if self.camera_failed or self.latest is None or exp is None:
            self.finish_tune("Tuning stopped: camera not available")
            return
        now = time.monotonic()
        if now < self.tune_until or self.frame_counter < self.tune_frames:
            return                                 # let the last change reach the sensor
        if self.tune_state == "locking":
            if exp.inactive:                       # manual mode not active yet
                if now - self.tune_started > 4:
                    self.finish_tune("Could not switch to manual exposure")
                return
            self.tune_state, self.tune_exp = "measuring", exp.value
        small = np.ascontiguousarray(self.latest[::4, ::4])
        mean = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())
        ratio = min(max(TUNE_TARGET / max(mean, 1.0), 0.4), 2.5)
        new = int(min(max(round(self.tune_exp * ratio), exp.min), exp.max))
        done = f"Exposure locked at {self.tune_exp} (brightness {mean:.0f} of 255)"
        if abs(mean - TUNE_TARGET) <= TUNE_TOLERANCE:
            self.finish_tune(done)
        elif new == self.tune_exp:
            self.finish_tune(done + ". Exposure limit reached: add light or adjust Gamma.")
        elif self.tune_iter >= 10:
            self.finish_tune(done + ". Stopped after 10 tries.")
        else:
            self.tune_exp, self.tune_iter = new, self.tune_iter + 1
            self.controls.set_control(exp.name, new)
            self.tune_until = now + 0.6
            self.tune_frames = self.frame_counter + 3

    def finish_tune(self, message):
        self.tune_timer.stop()
        self.tune_state = "idle"
        self.status.setText(message)

    # ---------------- auto-capture ----------------
    def toggle_auto(self):
        if self.auto_state == "idle":
            self.start_auto()
        else:
            self.finish_auto("stopped")

    def start_auto(self):
        if self.latest is None or self.camera_failed:
            self.status.setText("No live camera image to capture from")
            return
        if self.stack_thread is not None and self.stack_thread.isRunning():
            self.status.setText("Wait for the current stack to finish processing")
            return
        if self.tune_state != "idle":
            self.status.setText("Wait for exposure tuning to finish first")
            return
        if self.still is not None:
            self.toggle_freeze()                   # back to the live view
        self.stack_frames, self.stack_dir = [], None   # every run is a fresh group/folder
        self.auto_settings.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.auto_btn.setText("Stop (S)")
        self.auto_last_frame = -1
        self.auto_state = "delay"
        self.auto_next = time.monotonic() + self.auto_delay.value()
        self.auto_timer.start()
        self.auto_tick()

    def take_shot(self):
        """Save the newest live frame into this run's folder. Returns True on success."""
        frame = self.latest.copy()
        try:
            if self.stack_dir is None:
                self.stack_dir = day_folder("stacks") / f"stack-{datetime.now():%H%M%S}"
                self.stack_dir.mkdir(parents=True, exist_ok=True)
            save_png(self.stack_dir / f"frame-{len(self.stack_frames) + 1:03d}.png", frame)
        except OSError as exc:
            self.status.setText(f"Save failed: {exc}")
            return False
        self.stack_frames.append(frame)
        return True

    def auto_tick(self):
        if self.auto_state == "idle":
            return
        if self.camera_failed:
            self.finish_auto("camera stopped")
            return
        now = time.monotonic()
        if self.auto_state == "delay":
            remaining = self.auto_next - now
            if remaining > 0:
                self.auto_label.setText(f"Starting in {math.ceil(remaining)} s. Let the stand settle.")
                return
            self.auto_state = "running"
            self.auto_next = now
        # only shoot when a fresh camera frame has arrived since the last shot
        if now >= self.auto_next and self.frame_counter != self.auto_last_frame:
            if not self.take_shot():
                self.finish_auto("save failed")
                return
            self.auto_last_frame = self.frame_counter
            self.auto_next = now + self.auto_interval.value()
            QApplication.beep()
            self.auto_label.setStyleSheet("background: #2e7d32; color: white; padding: 4px;")
            QTimer.singleShot(250, lambda: self.auto_label.setStyleSheet(""))
            if len(self.stack_frames) >= self.auto_max_shots.value():
                self.finish_auto("done")
                return
        self.auto_label.setText(
            f"Shot {len(self.stack_frames)} of {self.auto_max_shots.value()}. "
            f"Next in {max(0.0, self.auto_next - now):.1f} s")

    def finish_auto(self, reason):
        self.auto_timer.stop()
        self.auto_state = "idle"
        self.auto_settings.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.auto_btn.setText("Start auto-capture (S)")
        n = len(self.stack_frames)
        self.auto_label.setText(f"Finished: {n} frames ({reason})")
        if n < 2:
            self.status.setText(f"Only {n} frame(s) captured ({reason}); nothing to process")
        else:
            self.process_stack()

    # ---------------- stacking ----------------
    def process_stack(self):
        if self.stack_thread is not None and self.stack_thread.isRunning():
            return
        if len(self.stack_frames) < 2:
            self.status.setText("Nothing to process yet. Run auto-capture first.")
            return
        mode = "focus" if self.stack_mode.currentIndex() == 0 else "average"
        self.process_btn.setEnabled(False)
        self.progress.setValue(0)
        self.progress.show()
        self.stack_mode_used = mode
        self.stack_thread = StackThread(list(self.stack_frames), self.align_box.isChecked(), mode)
        self.stack_thread.progress.connect(self.on_stack_progress)
        self.stack_thread.done.connect(self.on_stack_done)
        self.stack_thread.failed.connect(self.on_stack_failed)
        self.stack_thread.start()
        self.status.setText("Processing stack...")

    @Slot(int, int)
    def on_stack_progress(self, i, total):
        self.progress.setMaximum(total)
        self.progress.setValue(i)

    @Slot(object)
    def on_stack_done(self, result):
        self.process_btn.setEnabled(True)
        self.progress.hide()
        try:
            path = unique_path(self.stack_dir, f"result-{self.stack_mode_used}")
            save_png(path, result)
            message = f"Stack saved to {path}"
        except OSError as exc:
            message = f"Stack done but save failed: {exc}"
        self.show_still(result, message)

    @Slot(str)
    def on_stack_failed(self, text):
        self.process_btn.setEnabled(True)
        self.progress.hide()
        self.status.setText(f"Stacking failed: {text}")

    def closeEvent(self, event):
        self.auto_timer.stop()
        self.tune_timer.stop()
        self.stop_thumbs()
        self.cam.stop()
        self.controls.shutdown()
        if self.stack_thread is not None and self.stack_thread.isRunning():
            self.stack_thread.wait(10000)
        super().closeEvent(event)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Microscope viewer")
    parser.add_argument("device", nargs="?", default="/dev/video5")
    parser.add_argument("--size", default="1600x1200", help="WIDTHxHEIGHT")
    args = parser.parse_args()
    try:
        width, height = (int(v) for v in args.size.lower().split("x"))
    except ValueError:
        parser.error("--size must look like 1600x1200")

    signal.signal(signal.SIGINT, signal.SIG_DFL)   # Ctrl+C exits at once, skipping cleanup
    app = QApplication([sys.argv[0]])
    app.setDesktopFileName("microscope-viewer")    # matches microscope-viewer.desktop
    win = MainWindow(args.device, width, height)
    win.show()
    sys.exit(app.exec())
