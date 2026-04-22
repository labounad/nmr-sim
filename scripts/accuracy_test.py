import matplotlib.pyplot as plt
import numpy as np

x_fine = np.linspace(-10, 10, 50000)
L_true = 1.0 / (1.0 + x_fine**2)
theta_fine = np.arctan(x_fine)

point_counts = [5, 10, 20, 50, 100, 500, 1000]
max_errors = []

for n in point_counts:
    theta_samples = np.linspace(-1.5, 1.5, n)
    cos2_samples = np.cos(theta_samples)**2
    cos2_interp = np.interp(theta_fine, theta_samples, cos2_samples)
    error = np.max(np.abs(L_true - cos2_interp))
    max_errors.append(error)
    print(f"{n:>5} points: max error = {error:.2e}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: max error vs number of points
ax = axes[0]
ax.semilogy(point_counts, max_errors, 'bo-', linewidth=2)
ax.set_xlabel("Number of template points")
ax.set_ylabel("Max absolute error")
ax.set_title("Interpolation accuracy vs template size")
ax.grid(True, alpha=0.3)

# Right: overlay for a few key point counts
ax = axes[1]
ax.plot(x_fine, L_true, 'k-', linewidth=1, alpha=0.5, label="True")
for n, color in [(5, 'red'), (10, 'orange'), (20, 'green'), (100, 'blue')]:
    theta_samples = np.linspace(-1.5, 1.5, n)
    cos2_samples = np.cos(theta_samples)**2
    cos2_interp = np.interp(theta_fine, theta_samples, cos2_samples)
    ax.plot(x_fine, cos2_interp, color=color, linewidth=1.5, label=f"{n} points")

ax.set_xlim(-5, 5)
ax.set_xlabel("x")
ax.set_ylabel("L(x)")
ax.set_title("Interpolated Lorentzians at different template sizes")
ax.legend()

plt.tight_layout()
plt.savefig("accuracy_test.png", dpi=150)
print("Saved accuracy_test.png")
