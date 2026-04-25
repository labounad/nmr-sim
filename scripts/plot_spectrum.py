"""Interactive plotter for any `chemical_shift_ppm,intensity` CSV produced by
nmr-sim (first-order `Spectrum::to_csv_ppm` or quantum `DiscreteSpectrum::to_csv_ppm`).

Usage:
    python scripts/plot_spectrum.py                           # defaults to spectrum.csv, typical 1H range
    python scripts/plot_spectrum.py zeeman_1h_spectrum.csv    # any filename
    python scripts/plot_spectrum.py spec.csv --xlim -1 15     # custom ppm window
    python scripts/plot_spectrum.py spec.csv --full           # show the whole Nyquist range

The default x-range is -1 to 15 ppm, which covers the conventional 1H
chemical-shift window. This is a view clamp only — the CSV itself always
contains every FFT bin. At low spectrometer frequencies (e.g. 60 MHz) the
Nyquist window in Hz covers more than ±150 ppm, so without clamping the
interesting region would collapse into a tiny sliver of the plot.
"""
import argparse
import csv
import matplotlib
matplotlib.use('TkAgg')  # interactive backend
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('filename', nargs='?', default='spectrum.csv',
                    help='path to the CSV (default: spectrum.csv)')
parser.add_argument('--xlim', nargs=2, type=float, metavar=('LOW', 'HIGH'),
                    default=[-1.0, 15.0],
                    help='ppm range to display (default: -1 15, typical 1H window)')
parser.add_argument('--full', action='store_true',
                    help='show the full spectrum without clamping to --xlim')
args = parser.parse_args()

ppm, intensity = [], []
with open(args.filename) as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        ppm.append(float(row[0]))
        intensity.append(float(row[1]))

fig, ax = plt.subplots(figsize=(12, 5))

# Styling constants — grouped so future tweaks don't hunt through the file.
AXIS_GREY = '#808080'      # spines and tick marks
BASELINE_GREY = '#D3D3D3'  # lighter, for the horizontal baseline

# Light-grey baseline at y = 0. zorder=0 keeps it behind the spectrum so the
# black trace draws cleanly on top where they intersect.
ax.axhline(0, color=BASELINE_GREY, linewidth=0.8, zorder=0)

# Spectrum: pure black, half the previous linewidth (0.8 -> 0.4).
ax.plot(ppm, intensity, color='black', linewidth=0.4)

# X-axis title in Helvetica Neue Light; Y-axis label dropped entirely.
ax.set_xlabel('Chemical Shift (ppm)', family='Helvetica Neue', weight='light', fontsize=12)
ax.set_title(f'NMR Spectrum: {args.filename}')

# Tick labels (numeric axis values) in Helvetica Neue Bold.
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontfamily('Helvetica Neue')
    label.set_fontweight('bold')

# Remove the top/right spines; colour the remaining bottom/left spines and
# tick marks grey. labelcolor stays default black so the Helvetica Neue Bold
# numerals read cleanly against white.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color(AXIS_GREY)
ax.spines['left'].set_color(AXIS_GREY)
ax.tick_params(axis='both', which='both', color=AXIS_GREY, labelcolor='black')

if args.full:
    # Show every bin; invert so downfield is on the left per NMR convention.
    ax.invert_xaxis()
else:
    # Clamp to the requested window; set_xlim(high, low) puts the larger ppm
    # value on the left, which is the NMR convention (downfield left).
    low, high = sorted(args.xlim)
    ax.set_xlim(high, low)

plt.tight_layout()
plt.show()
