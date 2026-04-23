"""Interactive plotter for any `chemical_shift_ppm,intensity` CSV produced by
nmr-sim (first-order `Spectrum::to_csv_ppm` or quantum `DiscreteSpectrum::to_csv_ppm`).

Usage:
    python scripts/plot_spectrum.py                         # defaults to spectrum.csv
    python scripts/plot_spectrum.py zeeman_1h_spectrum.csv  # any filename
"""
import csv
import sys
import matplotlib
matplotlib.use('TkAgg')  # interactive backend
import matplotlib.pyplot as plt

filename = sys.argv[1] if len(sys.argv) > 1 else 'spectrum.csv'

ppm, intensity = [], []
with open(filename) as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        ppm.append(float(row[0]))
        intensity.append(float(row[1]))

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(ppm, intensity, linewidth=0.8)
ax.set_xlabel('Chemical Shift (ppm)')
ax.set_ylabel('Intensity')
ax.set_title(f'NMR Spectrum: {filename}  (use zoom tool to inspect multiplets)')
ax.invert_xaxis()

plt.tight_layout()
plt.show()
