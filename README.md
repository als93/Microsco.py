# 🔬 Microsco.py

Single-file Python desktop app for a USB V4L2 microscope camera — live
preview, calibrated measurement, and focus stacking. Runs in a container
or natively on Linux.

![license: GPLv3](https://img.shields.io/badge/license-GPLv3-A42FDB.svg) ![python: 3.x](https://img.shields.io/badge/python-3.x-3776ab.svg) ![gui: PySide6](https://img.shields.io/badge/GUI-PySide6-17bebb.svg)

## Features

- **Live preview** at your camera’s resolution (`PreviewView`, click-drag coordinates in image pixels)
- **Capture** (`C`) — atomic PNG (`Pictures/Microscope/YYYY-MM-DD/photos/`)
- **Freeze / Live** (`F`) — measure on a frozen or opened image; overlays never pollute saved photos
- **Camera controls** dock — sliders/checkboxes/menus parsed live from `v4l2-ctl --list-ctrls-menus`; `Lock exposure` (manual exposure + white balance), per-control reset, and **Auto-tune exposure** (nudges `exposure_time_absolute` until average brightness ≈ 115)
- **Calibrated measurement** — draw a line over a ruler to calibrate (`µm/mm` etc.), then drag to measure; distances shown in list; `Scale bar` toggle; save an annotated PNG with lines + bar
- **Auto-capture focus stacks** (`S`) — timed shots into `YYYY-MM-DD/stacks/stack-HHMMSS/`, optional ECC alignment, then **Focus stack** (sharpest-pixel extended depth-of-field with seam soft-blend) or **Average** (noise reduction)
- **Library tab** — date picker, thumbnail grid (`ThumbThread` background decode), double-click to open, `[`/`]` to step, multi-select → safe Trash (`QFile.moveToTrash`)

## Requirements

| Tool | Why |
|------|-----|
| `python3` | runtime |
| `PySide6` | GUI |
| `opencv-python-headless`, `numpy` | image / stacking |
| `v4l-utils` (`v4l2-ctl`) | Camera tab |
| V4L2 USB camera (e.g. `/dev/video5`) | live feed |

Install (Fedora-container-style):

```bash
pip install PySide6 opencv-python-headless numpy
# v4l-utils comes from the OS (dnf/apt/yum), not pip
```

If `v4l-utils` is missing you’ll see *“Could not read controls … Is
v4l-utils installed?”* in the Camera tab, but viewing/measuring still works.

## Usage

```bash
python3 Microsco.py [/dev/videoN] [--size WIDTHxHEIGHT]
```

Examples:

```bash
python3 Microsco.py # defaults: /dev/video5 1600x1200
python3 Microsco.py /dev/video0 --size 1280x720
```

| Key | Action |
|-----|--------|
| `C` | Capture photo |
| `F` | Freeze live / resume |
| `S` | Start/stop auto-capture |
| `[` / `]` | Previous / next Library image |
| `Ctrl+C` | Quit (no cleanup) |

**Photos land in:**

```
~/Pictures/Microscope/YYYY-MM-DD/
  photos/   photo-HHMMSS-ffffff.png
  annotated/annotated-HHMMSS-ffffff.png   (Save annotated image)
  stacks/stack-HHMMSS/frame-001.png … result-focus-…png
```

### Typical workflow

1. Lock exposure (Camera tab) → **Auto-tune exposure** once if the image is dark
2. Press `C` for a test photo
3. `F` to freeze → Measure tab → *Calibrate* (drag over a ruler, type `5 mm`) → drag to measure; enable *Scale bar* before saving annotated
4. Stack tab: set interval/shots, `S`, turn the focus wheel after each beep (don’t touch the stand), wait for processing, then save via Capture
5. Library tab: pick a day, double-click to re-open, `[`/`]` to flip through

## Project layout

```
Microsco.py     — single file: GUI + camera + V4L2 + measure + stacking + library
README.md       — this file
GITHUB-GUIDE.md — how to publish this folder as a new GitHub repo (name: Microsco.py)
.gitignore      — __pycache__, .part files, etc.
LICENSE         — MIT
```

## Screenshot

```text
┌─────────────────────────────┬────────────┐
│ Preview (1600×1200)         │ Camera     │
│ drag to measure (when frozen)│ Measure   │
│                             │ Stack      │
│                             │ Library    │
├─────────────────────────────┴────────────┤
│ [Capture (C)] [Freeze (F)] [Open image…] │
└──────────────────────────────────────────┘
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| “Could not open /dev/videoN” | Check `ls /dev/video*`; close `guvcview` (exclusive lock); add user to `video` group |
| “Requested 1600×1200” note | camera can’t do that mode — try `v4l2-ctl --list-formats-ext` |
| Controls tab says “v4l-utils not installed” | Install `v4l-utils` in container |
| Measure looks wrong after zoom change | Recalibrate — `ppu` rescales by image width, not magnification |

## License

MIT — see `LICENSE`.

## Credits

Single-file implementation; V4L2/ OpenCV / PySide6. If you fork, keep the atomic-save + library + stacking notes in the commit history.
