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
ax.plot(ppm, intensity, linewidth=0.8)
ax.set_xlabel('Chemical Shift (ppm)')
ax.set_ylabel('Intensity')
ax.set_title(f'NMR Spectrum: {args.filename}')

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
