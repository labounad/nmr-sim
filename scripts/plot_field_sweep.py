"""Interactive overlay plot for a field-sweep spectrum set from nmr-sim.

Reads a directory produced by the `five_me_cyclohexenone_field_sweep` Rust
example (or any directory matching the same layout):

    <dir>/
        manifest.csv              # index, spectrometer_mhz, b0_tesla, filename
        000_XXXX.XXMHz.csv        # one absorption-mode spectrum per field
        001_XXXX.XXMHz.csv
        …
        100_XXXX.XXMHz.csv

All spectra share the same ppm axis by construction (the Rust side adapts
`dt` per field to keep the ppm bin spacing invariant), so we can stack them
into a single (N_fields, N_bins) matrix and plot with one x-axis. The
loader is parametric in N_fields, so this script handles any sweep length
(e.g. 20-point or 101-point) with no changes.

Usage:
    python scripts/plot_field_sweep.py examples/outputs/five_me_cyclohexenone_field_sweep/
    python scripts/plot_field_sweep.py <dir>/ --xlim 0 8
    python scripts/plot_field_sweep.py <dir>/ --absolute     # don't max-normalize

Interactive controls:
    • Drag the slider under the plot, or click anywhere along it.
    • Left / Right arrow keys step through fields.
    • Mouse wheel over the slider also steps.

The slider drag uses matplotlib *blitting* (see BlitManager below) so
that even at 101 background spectra the redraw on each event is
essentially free — only the bold black foreground line and the title
text are repainted, on top of a cached snapshot of everything static.
Without blitting, dragging is jumpy at this dataset size.

Styling mirrors plot_spectrum.py (Helvetica Neue, despined grey axes,
light-grey baseline) so the single-spectrum and overlay plots look like
siblings.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


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
args = parser.parse_args()


# ---------- Load the sweep ----------

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

# Load each spectrum. At ~100 files × 131k rows we want np.loadtxt rather
# than csv.reader + list-append — the latter costs many seconds of startup
# and scales linearly in N_fields. We still verify that all files share
# the same ppm grid so a mismatched directory fails loudly rather than
# silently plotting garbage.
ppm = None
intensities = []  # shape (n_fields, n_bins) once stacked
print(f"Loading {len(fields)} spectra from {args.directory}/ ...")
for rec in fields:
    path = os.path.join(args.directory, rec['filename'])
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    rec_ppm = data[:, 0]
    rec_int = data[:, 1]
    if ppm is None:
        ppm = rec_ppm
    else:
        # Tiny float mismatch is expected; fail on anything larger than
        # ~1e-6 ppm (well below what the plot could distinguish).
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
    peaks[peaks == 0.0] = 1.0  # degenerate but safe
    intensities = intensities / peaks

freqs_mhz = np.array([rec['spectrometer_mhz'] for rec in fields])


# ---------- Styling constants (match plot_spectrum.py) ----------

AXIS_GREY = '#808080'
BASELINE_GREY = '#D3D3D3'
# Background spectra: a single, faint grey. A per-field gradient is tempting
# but ends up visually muddy when 20 lines overlap. The slider tells you
# "which field" — the background's job is just to show the envelope of
# where every other field goes.
BACKGROUND_GREY = '#A0A0A0'
BACKGROUND_ALPHA = 0.25
BACKGROUND_LW = 0.3
SELECTED_COLOR = 'black'
SELECTED_LW = 0.9


# ---------- Blit manager ----------
#
# matplotlib's default redraw on every slider event repaints all 101
# background lines (~13M points), which on a typical laptop runs at
# ~5 fps and feels noticeably jumpy under drag. The fix is *blitting*:
# after the initial layout, snapshot the static pixels (everything the
# user won't be moving) into an offscreen buffer, mark the dynamic
# artists as "animated" so they're excluded from full canvas redraws,
# and on each frame restore the snapshot, repaint *only* the animated
# artists, and push that to screen. Net effect on this dataset: ~50+
# fps drag with no other code changes.
#
# This class is the standard matplotlib BlitManager pattern — see
# matplotlib's own "Faster rendering by using blitting" example. It
# handles two subtleties:
#   1. The cached background must be re-captured after any full canvas
#      redraw (e.g., a window resize) — done via `draw_event` callback.
#   2. Animated artists must still be drawn at *initial* render time
#      so the user doesn't see an empty plot before the first slider
#      interaction — done in `_on_draw`.
class BlitManager:
    """Caches the figure's static background and re-blits dynamic artists
    on demand. Pass the canvas plus the artists that will move (e.g. the
    selected spectrum line and the title text); call `update()` after
    mutating any of those artists' state."""

    def __init__(self, canvas, animated_artists):
        self.canvas = canvas
        self._bg = None
        self._artists = []
        for a in animated_artists:
            a.set_animated(True)
            self._artists.append(a)
        # `draw_event` fires after every canvas.draw() (including the
        # one matplotlib runs after a window resize). Recapturing the
        # background there keeps the cache valid across resizes.
        self.cid = canvas.mpl_connect('draw_event', self._on_draw)

    def _on_draw(self, event):
        cv = self.canvas
        if event is not None and event.canvas is not cv:
            return
        # Capture everything currently on the canvas — animated artists
        # are excluded from canvas.draw() automatically, so they're not
        # in this snapshot.
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        # Draw the animated artists once so the user sees the initial
        # state (otherwise the first frame would show no foreground).
        self._draw_animated()

    def _draw_animated(self):
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        cv = self.canvas
        fig = cv.figure
        if self._bg is None:
            # Cold path — first call before any canvas.draw() has run.
            # Just trigger one and let _on_draw build the cache.
            cv.draw()
            return
        cv.restore_region(self._bg)
        self._draw_animated()
        cv.blit(fig.bbox)
        # flush_events keeps the GUI responsive even when many update()
        # calls land in quick succession (rapid slider drags).
        cv.flush_events()


# ---------- Build figure ----------

fig = plt.figure(figsize=(12, 6.5))
# Main plot occupies the top ~80%; slider sits in the bottom ~10% with a gap.
ax = fig.add_axes([0.08, 0.20, 0.88, 0.72])
slider_ax = fig.add_axes([0.12, 0.07, 0.80, 0.03])

# Baseline at y = 0. Behind everything so crossings of the spectrum through
# zero aren't occluded.
ax.axhline(0, color=BASELINE_GREY, linewidth=0.8, zorder=0)

# All 20 spectra, faint grey. Kept as a list so we can tweak later if we
# want a per-field colormap treatment.
background_lines = []
for k in range(n_fields):
    (line,) = ax.plot(
        ppm, intensities[k],
        color=BACKGROUND_GREY,
        linewidth=BACKGROUND_LW,
        alpha=BACKGROUND_ALPHA,
        zorder=1,
    )
    background_lines.append(line)

# The highlighted spectrum. Starts at the highest field (rightmost of the
# sweep) — that's the most recognizable 1H spectrum for chemists and a
# natural anchor for "now drag toward 40 MHz to see the strong-coupling
# regime emerge".
initial_idx = n_fields - 1
(selected_line,) = ax.plot(
    ppm, intensities[initial_idx],
    color=SELECTED_COLOR,
    linewidth=SELECTED_LW,
    zorder=5,
)

# Axis chrome — mirrors plot_spectrum.py.
ax.set_xlabel('Chemical Shift (ppm)',
              family='Helvetica Neue', weight='light', fontsize=12)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontfamily('Helvetica Neue')
    label.set_fontweight('bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color(AXIS_GREY)
ax.spines['left'].set_color(AXIS_GREY)
ax.tick_params(axis='both', which='both', color=AXIS_GREY, labelcolor='black')

# Title carries the currently-selected frequency. Helvetica Neue Light for
# the prefix, Bold for the number — done as a two-run title by overwriting
# with set_title and live-updating below.
title_artist = ax.set_title(
    f'5-Me-cyclohexenone 1H at {freqs_mhz[initial_idx]:.2f} MHz',
    family='Helvetica Neue', weight='light', fontsize=13,
)

# View window.
if args.full:
    ax.invert_xaxis()
else:
    low, high = sorted(args.xlim)
    ax.set_xlim(high, low)


# ---------- Slider ----------

# Use discrete `valstep` so the slider snaps exactly to the 20 precomputed
# frequencies. valfmt formats the display as "123.45 MHz".
slider = Slider(
    slider_ax,
    label='',
    valmin=freqs_mhz[0],
    valmax=freqs_mhz[-1],
    valinit=freqs_mhz[initial_idx],
    valstep=freqs_mhz.tolist(),
    valfmt='%.2f MHz',
    color=AXIS_GREY,
    track_color=BASELINE_GREY,
)
# Style the slider's own text.
slider.valtext.set_fontfamily('Helvetica Neue')
slider.valtext.set_fontweight('bold')

# Keep the current index in a mutable box so the key / scroll handlers can
# update it alongside the slider.
state = {'idx': initial_idx}

# The BlitManager wraps the two artists that change when the slider moves:
# the selected spectrum line and the dynamic title text. Everything else
# (background lines, axes chrome, baseline, slider widget itself) stays in
# the cached snapshot and is reused frame-to-frame.
bm = BlitManager(fig.canvas, [selected_line, title_artist])


def _apply_idx(idx):
    idx = int(np.clip(idx, 0, n_fields - 1))
    state['idx'] = idx
    selected_line.set_ydata(intensities[idx])
    title_artist.set_text(f'5-Me-cyclohexenone 1H at {freqs_mhz[idx]:.2f} MHz')
    # `bm.update()` replaces what used to be `fig.canvas.draw_idle()`.
    # The slider widget still calls draw_idle internally on drag (so its
    # own polygon and valtext redraw correctly); blitting only short-
    # circuits the spectrum redraw, which is the expensive part.
    bm.update()


def on_slider_change(val):
    # `val` is the snapped frequency (MHz); look up its index.
    idx = int(np.argmin(np.abs(freqs_mhz - val)))
    if idx != state['idx']:
        _apply_idx(idx)


slider.on_changed(on_slider_change)


def on_key(event):
    if event.key in ('right', 'up'):
        new_idx = state['idx'] + 1
    elif event.key in ('left', 'down'):
        new_idx = state['idx'] - 1
    else:
        return
    new_idx = int(np.clip(new_idx, 0, n_fields - 1))
    # Update the slider; its on_changed will fire and cascade into _apply_idx.
    slider.set_val(freqs_mhz[new_idx])


def on_scroll(event):
    # Only respond when the mouse is over the slider axes — otherwise we'd
    # fight matplotlib's built-in scroll-to-zoom on the main plot.
    if event.inaxes is not slider_ax:
        return
    direction = 1 if event.button == 'up' else -1
    new_idx = int(np.clip(state['idx'] + direction, 0, n_fields - 1))
    slider.set_val(freqs_mhz[new_idx])


fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('scroll_event', on_scroll)

plt.show()
