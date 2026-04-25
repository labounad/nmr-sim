"""Render a field-sweep highlight animation as MP4 or GIF.

Loads a directory produced by `five_me_cyclohexenone_field_sweep` (or any
sweep matching the same layout) and produces a video of the "highlighted
spectrum" sliding through every field from the lowest to the highest MHz,
on top of a faint backdrop of all the other spectra. Same data path as
plot_field_sweep.py — just non-interactive, fixed view, frame-by-frame.

    <dir>/
        manifest.csv              # index, spectrometer_mhz, b0_tesla, filename
        000_XXXX.XXMHz.csv        # one absorption-mode spectrum per field
        001_XXXX.XXMHz.csv
        …

Default settings (matching the request from 2026-04-25):
    * ppm window:         1.7 - 2.9   (NMR convention, displayed high → low)
    * intensity window:   -2% - 60%   (after per-spectrum max-normalization)
    * frame rate:         20 fps      (101 frames ≈ 5 s)
    * output format:      mp4 if ffmpeg available, else gif

Usage:
    python scripts/animate_field_sweep.py examples/outputs/five_me_cyclohexenone_field_sweep/
    python scripts/animate_field_sweep.py <dir>/ --output sweep.gif
    python scripts/animate_field_sweep.py <dir>/ --fps 30 --output sweep.mp4
    python scripts/animate_field_sweep.py <dir>/ --xlim 0 8 --ylim -0.05 1.05

Requirements:
    * matplotlib, numpy (standard).
    * For .mp4 output: a working `ffmpeg` on PATH (matplotlib's FFMpegWriter).
    * For .gif output: Pillow (matplotlib's PillowWriter — already a hard
      matplotlib dep on Python 3.8+).

Why blitting is on inside FuncAnimation: only two artists change per
frame (the selected line's ydata and the title text). Letting matplotlib
re-blit just those instead of redrawing all 101 background lines per
frame keeps the encoder feed fast.
"""
import argparse
import csv
import os
import shutil
import sys

import matplotlib
matplotlib.use('Agg')  # headless rendering — no GUI window pops up
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


# ---------- CLI ----------

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument('directory',
                    help='directory containing manifest.csv and per-field spectrum CSVs')
parser.add_argument('--output', '-o', default=None,
                    help='output path (.mp4 or .gif). Default: '
                         '<directory>/field_sweep.<mp4|gif> — picks mp4 if '
                         'ffmpeg is on PATH, else gif')
parser.add_argument('--xlim', nargs=2, type=float, metavar=('LOW', 'HIGH'),
                    default=[1.7, 2.9],
                    help='ppm range to display (default: 1.7 2.9)')
parser.add_argument('--ylim', nargs=2, type=float, metavar=('LOW', 'HIGH'),
                    default=[-0.02, 0.60],
                    help='intensity range as fraction of per-spectrum max '
                         '(default: -0.02 0.60 ≡ -2%% to 60%%)')
parser.add_argument('--fps', type=int, default=20,
                    help='frames per second (default: 20)')
parser.add_argument('--absolute', action='store_true',
                    help='do NOT max-normalize each spectrum '
                         '(plot raw intensities)')
parser.add_argument('--dpi', type=int, default=150,
                    help='render DPI (default: 150)')
parser.add_argument('--figsize', nargs=2, type=float, metavar=('W', 'H'),
                    default=[10.0, 5.5],
                    help='figure size in inches (default: 10 5.5)')
args = parser.parse_args()


# ---------- Pick output format ----------

def _resolve_writer(output_path):
    """Return (writer, output_path) based on extension and available tools.

    If output_path has no extension, default to .mp4 if ffmpeg is on PATH,
    else .gif. If a .mp4 was requested but ffmpeg is missing, fail loudly
    rather than silently producing a giant gif — this is the surprising
    failure mode worth flagging.
    """
    if output_path is None:
        ext = '.mp4' if shutil.which('ffmpeg') else '.gif'
        output_path = os.path.join(args.directory, f'field_sweep{ext}')
    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.mp4':
        if not shutil.which('ffmpeg'):
            sys.exit(
                "error: --output is .mp4 but `ffmpeg` was not found on PATH. "
                "Install ffmpeg (`brew install ffmpeg` on macOS, `apt install "
                "ffmpeg` on Linux), or use --output <name>.gif instead."
            )
        # libx264 + yuv420p is the most universally playable codec/pixel
        # format combo (QuickTime, browsers, slack previews, etc.). Without
        # `pix_fmt yuv420p`, browsers and many native players show a black
        # screen. Bitrate is implicit — FFMpegWriter defaults are fine for
        # this kind of plot at 1.5k×800.
        writer = FFMpegWriter(
            fps=args.fps,
            codec='libx264',
            extra_args=['-pix_fmt', 'yuv420p', '-preset', 'medium'],
        )
        return writer, output_path
    if ext == '.gif':
        return PillowWriter(fps=args.fps), output_path
    sys.exit(f"error: unsupported output extension {ext!r}; use .mp4 or .gif")


writer, output_path = _resolve_writer(args.output)


# ---------- Load the sweep ----------

manifest_path = os.path.join(args.directory, 'manifest.csv')
if not os.path.exists(manifest_path):
    parser.error(f"no manifest.csv found in {args.directory}")

fields = []
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
print(f"Loading {n_fields} spectra from {args.directory}/ ...")
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
intensities = np.asarray(intensities)

if not args.absolute:
    # Per-spectrum max normalization (matches plot_field_sweep.py default).
    # Each row → unit max, so the requested -2%/60% ylim is a clean
    # fraction of "tallest peak in this field's spectrum".
    peaks = np.max(np.abs(intensities), axis=1, keepdims=True)
    peaks[peaks == 0.0] = 1.0
    intensities = intensities / peaks

freqs_mhz = np.array([rec['spectrometer_mhz'] for rec in fields])


# ---------- Styling (mirrors plot_field_sweep.py) ----------

AXIS_GREY = '#808080'
BASELINE_GREY = '#D3D3D3'
BACKGROUND_GREY = '#A0A0A0'
BACKGROUND_ALPHA = 0.25
BACKGROUND_LW = 0.5         # slightly thicker than the interactive plot
                            # because video compression eats sub-pixel lines
SELECTED_COLOR = 'black'
SELECTED_LW = 1.6           # thick enough to read after H.264 compression


# ---------- Build figure ----------

fig = plt.figure(figsize=args.figsize, dpi=args.dpi)
ax = fig.add_axes([0.08, 0.13, 0.88, 0.78])

# Baseline at y = 0, behind everything.
ax.axhline(0, color=BASELINE_GREY, linewidth=0.8, zorder=0)

# All N background spectra in faint grey. The animation will leave these
# untouched and only re-blit the selected line + title each frame.
for k in range(n_fields):
    ax.plot(
        ppm, intensities[k],
        color=BACKGROUND_GREY,
        linewidth=BACKGROUND_LW,
        alpha=BACKGROUND_ALPHA,
        zorder=1,
    )

# The sliding (selected) spectrum. Starts at frame 0 = lowest field.
(selected_line,) = ax.plot(
    ppm, intensities[0],
    color=SELECTED_COLOR,
    linewidth=SELECTED_LW,
    zorder=5,
)

# Axis chrome.
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

# View window — NMR convention is high ppm on the LEFT, so set xlim with
# the high value first so matplotlib auto-flips the axis.
low, high = sorted(args.xlim)
ax.set_xlim(high, low)
ax.set_ylim(args.ylim[0], args.ylim[1])

# Title — live-updates per frame.
title_artist = ax.set_title(
    f'5-Me-cyclohexenone 1H at {freqs_mhz[0]:.2f} MHz',
    family='Helvetica Neue', weight='light', fontsize=13,
)


# ---------- Animation ----------

# blit=True: matplotlib will re-render only the artists in the returned
# tuple per frame, blitting them onto the cached static background.
# That's a ~10× win for the encoder feed at 101 frames * 13M bg points.
def _update(frame_idx):
    selected_line.set_ydata(intensities[frame_idx])
    title_artist.set_text(
        f'5-Me-cyclohexenone 1H at {freqs_mhz[frame_idx]:.2f} MHz'
    )
    return selected_line, title_artist


anim = FuncAnimation(
    fig,
    _update,
    frames=n_fields,
    interval=1000.0 / args.fps,
    blit=True,
)

# Render.
print(f"Rendering {n_fields} frames at {args.fps} fps → {output_path} ...")
print(f"  ppm window: [{args.xlim[0]:.3f}, {args.xlim[1]:.3f}] (display flipped)")
print(f"  ylim:       [{args.ylim[0]:.3f}, {args.ylim[1]:.3f}]")
anim.save(output_path, writer=writer, dpi=args.dpi)
plt.close(fig)
print(f"Done: {output_path}")
