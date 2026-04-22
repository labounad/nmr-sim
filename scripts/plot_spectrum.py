import csv
import matplotlib
matplotlib.use('TkAgg')  # interactive backend
import matplotlib.pyplot as plt

ppm, intensity = [], []
with open('spectrum.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        ppm.append(float(row[0]))
        intensity.append(float(row[1]))

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(ppm, intensity, linewidth=0.8)
ax.set_xlabel('Chemical Shift (ppm)')
ax.set_ylabel('Intensity')
ax.set_title('NMR Spectrum (use zoom tool to inspect multiplets)')
ax.invert_xaxis()

plt.tight_layout()
plt.show()
