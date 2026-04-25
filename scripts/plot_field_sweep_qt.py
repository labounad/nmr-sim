"""Interactive overlay plot for a field-sweep spectrum set, rendered via
pyqtgraph for smooth 60+ fps scrubbing across 101+ curves and full zoom
controls.

Same data layout and CLI as `plot_field_sweep.py`:

    <dir>/
        manifest.csv               # index, spectrometer_mhz, b0_tesla, filename
        000_XXXX.XXMHz.csv         # one absorption-mode spectrum per field
        001_XXXX.XXMHz.csv
        …
        100_XXXX.XXMHz.csv

# Performance

The big trick used here for smoothness: the 101 background spectra are
stitched into a *single* `PlotDataItem` with NaN gaps between segments
(`connect='finite'`). Qt's painter has per-item overhead; one item with
13M points draws ~10× faster than 101 items with 131k points each. The
selected (highlighted) spectrum is a separate item so we can `setData`
on it cheaply when the slider moves, without rebuilding the giant
background array.

Neither curve enables `setDownsampling` or `setClipToView`: the bg curve's
stitched X array is non-monotonic and contains NaN gaps, and the selected
curve's ppm array is monotonically decreasing (NMR convention). pyqtgraph's
clip-to-view binary search assumes increasing X, so on the selected curve
it works at full zoom but collapses to an empty interval the moment you
zoom in, making the curve vanish. See the comment in `__init__`.

For hardware acceleration on the bg curve, pass `--opengl` to route the
painter through OpenGL (requires `pip install PyOpenGL`). That's how the
13M-point bg curve stays smooth without downsampling.

# Zoom modes

The toolbar at the top has four mouse-drag modes plus a reset button:

    Pan    (P)    — left-drag pans the view (pyqtgraph default)
    Box    (B)    — left-drag draws a rubber-band rect; release zooms to it
    H zoom (H)    — left-drag rubber-band; release zooms only X (Y unchanged)
    V zoom (V)    — left-drag rubber-band; release zooms only Y (X unchanged)
    Reset  (R)    — restore the initial xlim and auto-fit Y

Mouse wheel zooms about the cursor in any mode (built-in pyqtgraph
behavior). Right-clicking the plot still gives pyqtgraph's context menu
with extra options (manual range entry, export, etc.).

# Install

    pip install pyqtgraph PyQt6                 # most common
    pip install pyqtgraph PySide6               # LGPL-friendlier
    pip install pyqtgraph PyQt5                 # legacy, ≥ 5.15.x
    pip install PyOpenGL                        # optional, for --opengl

# Usage

    python scripts/plot_field_sweep_qt.py examples/outputs/five_me_cyclohexenone_field_sweep/
    python scripts/plot_field_sweep_qt.py <dir>/ --xlim 0 8
    python scripts/plot_field_sweep_qt.py <dir>/ --absolute   # don't max-normalize
    python scripts/plot_field_sweep_qt.py <dir>/ --opengl     # if PyOpenGL is installed

# Other interactive controls

    • Drag the slider, click anywhere along it, or scroll the wheel while
      hovering it (QSlider handles wheel natively).
    • Left / Right arrow keys step through fields anywhere in the window.

Styling matches plot_field_sweep.py / plot_spectrum.py — Helvetica Neue
Light title + Bold tick labels, grey despined axes, light-grey y=0
baseline, faint grey background spectra, bold black selected spectrum.
"""
import argparse
import csv
import os
import sys

import numpy as np

# pyqtgraph re-exports whichever Qt binding the user has installed
# (PyQt6 / PySide6 / PyQt5). Importing through `pyqtgraph.Qt` makes the
# script binding-agnostic.
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


# ---------- CLI ----------

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument('directory',
                    help='directory containing manifest.csv and per-field spectrum CSVs')
parser.add_argument('--xlim', nargs=2, type=float, metavar=('LOW', 'HIGH'),
                    default=[-1.0, 15.0],
                    help='ppm range to display (default: -1 15, typical 1H window)')
parser.add_argument('--full', action='store_true',
                    help='show the full spectrum without clamping to --xlim')
parser.add_argument('--absolute', action='store_true',
                    help='plot raw intensities without per-spectrum max normalization')
parser.add_argument('--opengl', action='store_true',
                    help='enable OpenGL rendering (requires `pip install PyOpenGL`); '
                         'unnecessary for this dataset but available as an escalation')
args = parser.parse_args()


# ---------- pyqtgraph configuration ----------
#
# These flags must be set BEFORE any widget is constructed.
# White background + black foreground match the matplotlib version and
# standard scientific-paper aesthetic. antialias is on because at 0.3-pt
# linewidth the difference between aliased and antialiased rendering is
# the difference between "spectrum" and "barcode".

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOption('antialias', True)
if args.opengl:
    pg.setConfigOption('useOpenGL', True)
    pg.setConfigOption('enableExperimental', True)


# ---------- Load the sweep ----------
#
# Identical to plot_field_sweep.py — read the manifest, load each
# spectrum via np.loadtxt, verify all spectra share the same ppm grid,
# optionally max-normalize.

manifest_path = os.path.join(args.directory, 'manifest.csv')
if not os.path.exists(manifest_path):
    parser.error(f"no manifest.csv found in {args.directory}")

fields = []  # list of dicts: {'index', 'spectrometer_mhz', 'b0_tesla', 'filename'}
with open(manifest_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        fields.append({
            'index': int(row['index']),
            'spectrometer_mhz': float(row['spectrometer_mhz']),
            'b0_tesla': float(row['b0_tesla']),
            'filename': row['filename'],
        })

fields.sort(key=lambda r: r['index'])
n_fields = len(fields)
if n_fields == 0:
    parser.error(f"manifest.csv in {args.directory} is empty")

ppm = None
intensities = []
print(f"Loading {len(fields)} spectra from {args.directory}/ ...")
for rec in fields:
    path = os.path.join(args.directory, rec['filename'])
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    rec_ppm = data[:, 0]
    rec_int = data[:, 1]
    if ppm is None:
        ppm = rec_ppm
    else:
        if rec_ppm.shape != ppm.shape or np.max(np.abs(rec_ppm - ppm)) > 1e-6:
            raise SystemExit(
                f"ppm axis mismatch between {fields[0]['filename']} and "
                f"{rec['filename']} — did you mix sweeps with different SW_PPM?"
            )
    intensities.append(rec_int)
intensities = np.array(intensities)

if not args.absolute:
    # Max-normalize each spectrum so field-dependent Boltzmann / γ·B₀
    # intensity scaling doesn't swamp the lineshape comparison.
    peaks = np.max(np.abs(intensities), axis=1, keepdims=True)
    peaks[peaks == 0.0] = 1.0
    intensities = intensities / peaks

freqs_mhz = np.array([rec['spectrometer_mhz'] for rec in fields])


# ---------- Styling ----------

AXIS_GREY = '#808080'
BASELINE_GREY = '#D3D3D3'
# Background: rgba — alpha=64/255 ≈ 0.25, matching the matplotlib version.
BACKGROUND_RGBA = (160, 160, 160, 64)
BACKGROUND_LW = 1.5   # was 1, bumped 1.5× for visual weight
SELECTED_COLOR = '#000000'
SELECTED_LW = 3.0     # was 1.5; Lucas dialed it in to 3.0 by eye

FONT_FAMILY = 'Helvetica Neue'


# ---------- ZoomViewBox ----------
#
# pyqtgraph's stock ViewBox supports two mouse modes: PanMode (left-drag
# pans the view) and RectMode (left-drag draws a rubber-band rectangle
# and on release zooms to it). Box-zoom is exactly RectMode.
#
# For horizontal-only and vertical-only zoom we want the rubber band to
# behave the same way visually (so the user gets feedback while
# dragging) but on release apply only the X or only the Y component of
# the selected rectangle. The cleanest way is to subclass ViewBox and
# override `showAxRect`, the method ViewBox calls when the rubber band
# is released — at that point the selected rectangle is in data coords,
# so we just call `setRange` with one axis and pass through the other
# unchanged.

class ZoomViewBox(pg.ViewBox):
    """ViewBox with a switchable zoom mode. `set_zoom_mode('pan' | 'box'
    | 'horizontal' | 'vertical')` changes how a left-drag is interpreted:

        pan         — drag pans the view (PanMode, the pyqtgraph default)
        box         — drag rubber-bands a 2D rect; release zooms to it
        horizontal  — drag rubber-bands; release zooms only the X axis
        vertical    — drag rubber-bands; release zooms only the Y axis

    Wheel zoom and right-click context menu still work in every mode.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._zoom_mode = 'pan'

    def set_zoom_mode(self, mode):
        valid = ('pan', 'box', 'horizontal', 'vertical')
        if mode not in valid:
            raise ValueError(f'mode must be one of {valid}, got {mode!r}')
        self._zoom_mode = mode
        # PanMode for "pan", RectMode for any of the rubber-band zooms.
        if mode == 'pan':
            self.setMouseMode(pg.ViewBox.PanMode)
        else:
            self.setMouseMode(pg.ViewBox.RectMode)

    def showAxRect(self, ax, **kwargs):
        """Called by ViewBox.mouseDragEvent when the user releases a
        rubber-band selection in RectMode. `ax` is a QRectF in data
        (view) coords. We dispatch on the current zoom mode."""
        ax = ax.normalized()
        if self._zoom_mode == 'box':
            # Same as the stock RectMode behavior.
            self.setRange(rect=ax, padding=0)
        elif self._zoom_mode == 'horizontal':
            # Zoom X to the selected x-range; leave Y alone.
            self.setRange(xRange=(ax.left(), ax.right()), padding=0)
        elif self._zoom_mode == 'vertical':
            # Zoom Y to the selected y-range; leave X alone. In a
            # normalized QRectF, top() is the smaller y and bottom() is
            # the larger one — the y-axis "low to high" pyqtgraph wants.
            self.setRange(yRange=(ax.top(), ax.bottom()), padding=0)
        else:
            # In pan mode showAxRect should never fire (PanMode doesn't
            # rubber-band) but if it does, fall back to the default.
            self.setRange(rect=ax, padding=0)
        # Update zoom history the way stock showAxRect does, so the
        # right-click context menu's "back / forward" navigation works.
        self.axHistoryPointer += 1
        self.axHistory = self.axHistory[: self.axHistoryPointer] + [ax]


# ---------- Main window ----------

class FieldSweepWindow(QtWidgets.QMainWindow):
    def __init__(self, ppm, intensities, freqs_mhz, xlim, full):
        super().__init__()
        self.ppm = ppm
        self.intensities = intensities
        self.freqs_mhz = freqs_mhz
        self.n_fields = len(freqs_mhz)
        # Start at the highest field — most recognizable 1H spectrum,
        # natural anchor for "drag toward 40 MHz to see strong coupling".
        self.idx = self.n_fields - 1
        # Save the requested xlim for the Reset button.
        self._initial_xlim = sorted(xlim) if not full else None

        self.setWindowTitle('5-Me-cyclohexenone field sweep')
        self.resize(1200, 720)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(20, 8, 20, 16)
        layout.setSpacing(8)

        # --- Toolbar with zoom-mode buttons ---
        # Built first so the QActions exist before plot construction
        # (in case any plot setup wants to defer to mode state). The
        # toolbar is added to the QMainWindow chrome (not the central
        # layout), which is the standard place for it.
        self._build_toolbar()

        # --- Title (Helvetica Neue Light, centered, above the plot) ---
        title_font = QtGui.QFont(FONT_FAMILY, 13)
        title_font.setWeight(QtGui.QFont.Weight.Light)
        self.title_label = QtWidgets.QLabel()
        self.title_label.setFont(title_font)
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)

        # --- Plot widget with our custom ZoomViewBox ---
        self.viewbox = ZoomViewBox()
        self.plot_widget = pg.PlotWidget(viewBox=self.viewbox)
        layout.addWidget(self.plot_widget, stretch=1)

        plot_item = self.plot_widget.getPlotItem()

        # Hide top/right axes (default for PlotItem, but explicit for safety).
        plot_item.showAxis('top', False)
        plot_item.showAxis('right', False)

        # Style bottom + left axes.
        tick_font = QtGui.QFont(FONT_FAMILY, 10)
        tick_font.setBold(True)
        for axis_name in ('bottom', 'left'):
            ax = plot_item.getAxis(axis_name)
            ax.setPen(pg.mkPen(AXIS_GREY, width=1.5))
            ax.setTextPen(pg.mkPen('k'))
            ax.setStyle(tickFont=tick_font)

        # X-axis label. Use a real QFont (matching the title) instead of
        # inline HTML — Qt's rich-text engine interprets `font-weight: 300`
        # inconsistently across platforms, so HTML CSS doesn't reliably
        # produce Helvetica Neue Light. A QFont with setWeight(Weight.Light)
        # picks the correct font face directly.
        label_font = QtGui.QFont(FONT_FAMILY, 12)
        label_font.setWeight(QtGui.QFont.Weight.Light)
        plot_item.setLabel('bottom', 'Chemical Shift (ppm)', color='black')
        plot_item.getAxis('bottom').label.setFont(label_font)
        # No y-axis label (matches plot_field_sweep.py).

        # Baseline at y=0, behind everything.
        baseline = pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen(BASELINE_GREY, width=1.5),
        )
        plot_item.addItem(baseline, ignoreBounds=True)

        # --- Background spectra: ALL combined into a single curve ---
        # The big perf trick. Stitch every spectrum's (ppm, intensity)
        # pair into one long array, with NaN separating each spectrum.
        # `connect='finite'` tells the painter to break the line at NaN,
        # so visually we get 101 disjoint polylines — but Qt only sees
        # one graphics item and one paint call. On this dataset that
        # buys a 5-10× speedup vs the per-curve approach.
        big_x = np.concatenate(
            [np.append(self.ppm, np.nan)] * self.n_fields
        )[:-1]  # drop the trailing NaN that the last spectrum left over
        big_y = np.concatenate([
            np.append(self.intensities[k], np.nan)
            for k in range(self.n_fields)
        ])[:-1]
        bg_pen = pg.mkPen(color=BACKGROUND_RGBA, width=BACKGROUND_LW)
        # PlotDataItem (not PlotCurveItem) — the higher-level wrapper is
        # what exposes setDownsampling / setClipToView / setSkipFiniteCheck.
        # It accepts the same `pen=` and `connect=` kwargs and forwards them
        # to an internal PlotCurveItem.
        self.bg_curve = pg.PlotDataItem(
            big_x, big_y, pen=bg_pen, connect='finite',
        )
        plot_item.addItem(self.bg_curve)

        # --- Selected (highlighted) curve, on top ---
        # Separate item so we can call `setData` on it cheaply when the
        # slider moves, without rebuilding the giant background array.
        self.selected_curve = pg.PlotDataItem(
            self.ppm, self.intensities[self.idx],
            pen=pg.mkPen(SELECTED_COLOR, width=SELECTED_LW),
        )
        self.selected_curve.setZValue(10)  # above bg in any insertion order
        plot_item.addItem(self.selected_curve)

        # Performance flags. We deliberately do NOT enable
        # setDownsampling / setClipToView on EITHER curve.
        #
        #   - bg_curve violates them because the stitched array is
        #     non-monotonic (X repeats 101 times) and contains NaN gaps,
        #     so peak-mode downsampling smears across spectrum boundaries
        #     and clipToView's binary search produces nonsense.
        #
        #   - selected_curve violates clipToView because the NMR ppm
        #     array is monotonically *decreasing* (downfield-first), then
        #     visually flipped with invertX(True). pyqtgraph's clipToView
        #     binary search assumes monotonically increasing X — it works
        #     at full zoom (everything's in range) but as soon as you
        #     zoom in, the bounds collapse to an empty interval and the
        #     whole curve disappears. The selected curve is only ~131k
        #     points (one spectrum) so there's no real perf pressure to
        #     keep these flags on anyway.
        #
        # setSkipFiniteCheck is still safe and helpful on the selected
        # curve: it promises the data has no NaN/Inf (true for one
        # spectrum) so pyqtgraph can skip a per-point isfinite() check.
        try:
            self.selected_curve.setSkipFiniteCheck(True)
        except AttributeError:
            pass  # pyqtgraph < 0.13

        # X-axis inversion (NMR convention: downfield → left).
        plot_item.invertX(True)
        if not full:
            low, high = sorted(xlim)
            plot_item.setXRange(low, high, padding=0)

        # --- Slider ---
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.n_fields - 1)
        self.slider.setValue(self.idx)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setTracking(True)
        self.slider.valueChanged.connect(self.on_slider_change)
        layout.addWidget(self.slider)

        # Initial title text.
        self._update_title()

        # Slider gets focus so arrow keys feel natural even before any click.
        self.slider.setFocus()

    # ----- Toolbar / zoom-mode plumbing -----

    def _build_toolbar(self):
        toolbar = self.addToolBar('Zoom')
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)

        # ActionGroup makes the four mode actions mutually exclusive:
        # only one can be checked at a time, and clicking a different
        # one auto-unchecks the previous.
        self.mode_group = QtGui.QActionGroup(self)
        self.mode_group.setExclusive(True)

        modes = [
            ('Pan',    'pan',        'P'),
            ('Box',    'box',        'B'),
            ('H zoom', 'horizontal', 'H'),
            ('V zoom', 'vertical',   'V'),
        ]
        for label, mode_key, shortcut in modes:
            action = QtGui.QAction(label, self)
            action.setCheckable(True)
            action.setShortcut(QtGui.QKeySequence(shortcut))
            # Default mode is pan, so its action starts checked.
            if mode_key == 'pan':
                action.setChecked(True)
            # `checked` arg from QAction.triggered is unused; we always
            # forward to set_zoom_mode with the captured key.
            action.triggered.connect(
                lambda checked=False, m=mode_key: self._set_zoom_mode(m)
            )
            self.mode_group.addAction(action)
            toolbar.addAction(action)

        toolbar.addSeparator()

        reset_action = QtGui.QAction('Reset', self)
        reset_action.setShortcut(QtGui.QKeySequence('R'))
        reset_action.triggered.connect(self._reset_view)
        toolbar.addAction(reset_action)

    def _set_zoom_mode(self, mode):
        self.viewbox.set_zoom_mode(mode)

    def _reset_view(self):
        """Restore the initial xlim (or auto-fit if `--full` was passed)
        and let pyqtgraph auto-range Y to the visible data."""
        if self._initial_xlim is not None:
            low, high = self._initial_xlim
            self.viewbox.setXRange(low, high, padding=0)
        else:
            self.viewbox.enableAutoRange(axis='x', enable=True)
        self.viewbox.enableAutoRange(axis='y', enable=True)

    # ----- Slider / data plumbing -----

    def on_slider_change(self, val):
        """Slider moved (drag, click, arrow on slider, or wheel-on-slider).
        Update only the highlighted curve — the (combined) background
        curve isn't redrawn, so this is a couple-of-microseconds setData."""
        self.idx = int(val)
        self.selected_curve.setData(self.ppm, self.intensities[self.idx])
        self._update_title()

    def keyPressEvent(self, event):
        """Window-level arrow-key handling: step the slider regardless of
        which child currently has focus. The toolbar's QAction shortcuts
        (P/B/H/V/R) work via Qt's shortcut system independently of this."""
        key = event.key()
        Key = QtCore.Qt.Key
        if key in (Key.Key_Right, Key.Key_Up):
            self.slider.setValue(min(self.idx + 1, self.n_fields - 1))
        elif key in (Key.Key_Left, Key.Key_Down):
            self.slider.setValue(max(self.idx - 1, 0))
        else:
            super().keyPressEvent(event)

    def _update_title(self):
        f = self.freqs_mhz[self.idx]
        self.title_label.setText(f'5-Me-cyclohexenone 1H at {f:.2f} MHz')


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = FieldSweepWindow(ppm, intensities, freqs_mhz, args.xlim, args.full)
    win.show()
    # `app.exec()` is the Qt6/PySide6 form; PyQt5 ≥ 5.15 supports it too.
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
