import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# --- Plot 1: The standard Lorentzian in x-space ---
x = np.linspace(-15, 15, 10000)
L = 1.0 / (1.0 + x**2)

ax = axes[0, 0]
ax.plot(x, L, 'b-', linewidth=1.5)
ax.set_title("Step 1: Standard Lorentzian L(x) = 1/(1+x²)")
ax.set_xlabel("x")
ax.set_ylabel("L(x)")

# --- Plot 2: The substitution — cos²(θ) in θ-space ---
theta = np.linspace(-1.5, 1.5, 10000)
cos2 = np.cos(theta)**2

ax = axes[0, 1]
ax.plot(theta, cos2, 'r-', linewidth=1.5)
ax.set_title("Step 2: After substitution x=tan(θ), L = cos²(θ)")
ax.set_xlabel("θ")
ax.set_ylabel("cos²(θ)")

# --- Plot 3: Uniform θ samples mapped to x-space ---
num_template = 30  # small number so dots are visible
theta_samples = np.linspace(-1.5, 1.5, num_template)
x_samples = np.tan(theta_samples)
L_samples = 1.0 / (1.0 + x_samples**2)

ax = axes[1, 0]
ax.plot(x, L, 'b-', linewidth=1, alpha=0.5, label="True Lorentzian")
ax.plot(x_samples, L_samples, 'ro', markersize=5, label=f"{num_template} uniform-θ samples")
ax.set_title("Step 3: Uniform θ → non-uniform x (dense at peak)")
ax.set_xlabel("x")
ax.set_ylabel("L(x)")
ax.set_xlim(-15, 15)
ax.legend()

# --- Plot 4: Linear interpolation comparison ---
# Interpolate in θ-space (good)
cos2_samples = np.cos(theta_samples)**2
x_fine = np.linspace(-14, 14, 5000)
theta_fine = np.arctan(x_fine)
cos2_interp = np.interp(theta_fine, theta_samples, cos2_samples)

# Interpolate in x-space (bad)
L_interp_x = np.interp(x_fine, x_samples, L_samples)

ax = axes[1, 1]
ax.plot(x_fine, 1.0 / (1.0 + x_fine**2), 'b-', linewidth=1, alpha=0.4, label="True Lorentzian")
ax.plot(x_fine, cos2_interp, 'g-', linewidth=2, label="Interpolated in θ-space (good)")
ax.plot(x_fine, L_interp_x, 'r--', linewidth=2, label="Interpolated in x-space (bad)")
ax.set_title(f"Step 4: Interpolation quality ({num_template} points)")
ax.set_xlabel("x")
ax.set_ylabel("L(x)")
ax.set_xlim(-5, 5)
ax.legend()

plt.tight_layout()
plt.savefig("template_explanation.png", dpi=150)
print("Saved template_explanation.png")
