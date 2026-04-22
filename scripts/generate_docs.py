"""
Generate all figures for the Lorentzian template interpolation document.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

out = "docs/doc_figures"
os.makedirs(out, exist_ok=True)

# ──────────────────────────────────────────────
# Figure 1: The problem — standard Lorentzian in x-space
# ──────────────────────────────────────────────
x = np.linspace(-15, 15, 10000)
L = 1.0 / (1.0 + x**2)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, L, 'b-', linewidth=2)
ax.set_xlabel("x (standardized frequency)", fontsize=12)
ax.set_ylabel("L(x)", fontsize=12)
ax.set_title("Standard Lorentzian: L(x) = 1 / (1 + x²)", fontsize=14)
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.grid(True, alpha=0.2)

# Annotate the sharpness
ax.annotate("Sharp peak\n(hard to interpolate)", xy=(0, 1), xytext=(4, 0.8),
            fontsize=10, arrowprops=dict(arrowstyle="->", color="red"),
            color="red")
ax.annotate("Long flat tails\n(wasted samples)", xy=(10, 0.01), xytext=(8, 0.3),
            fontsize=10, arrowprops=dict(arrowstyle="->", color="red"),
            color="red")
plt.tight_layout()
plt.savefig(f"{out}/fig1_lorentzian.png", dpi=150)
plt.close()
print("Saved fig1")

# ──────────────────────────────────────────────
# Figure 2: The substitution — cos²(θ) in θ-space
# ──────────────────────────────────────────────
theta = np.linspace(-1.5, 1.5, 10000)
cos2 = np.cos(theta)**2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(x, L, 'b-', linewidth=2)
ax1.set_xlabel("x", fontsize=12)
ax1.set_ylabel("L(x)", fontsize=12)
ax1.set_title("Before: L(x) = 1/(1+x²)", fontsize=13)
ax1.grid(True, alpha=0.2)

ax2.plot(theta, cos2, 'r-', linewidth=2)
ax2.set_xlabel("θ", fontsize=12)
ax2.set_ylabel("cos²(θ)", fontsize=12)
ax2.set_title("After: x = tan(θ) → L = cos²(θ)", fontsize=13)
ax2.grid(True, alpha=0.2)

# Add arrow between plots
fig.text(0.5, 0.5, "x = tan(θ)", fontsize=14, ha='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='orange'))

plt.tight_layout()
plt.savefig(f"{out}/fig2_substitution.png", dpi=150)
plt.close()
print("Saved fig2")

# ──────────────────────────────────────────────
# Figure 3: Uniform θ → non-uniform x (point placement)
# ──────────────────────────────────────────────
num_pts = 30
theta_s = np.linspace(-1.5, 1.5, num_pts)
x_s = np.tan(theta_s)
L_s = 1.0 / (1.0 + x_s**2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# θ-space: uniform dots on cos²
ax1.plot(theta, cos2, 'r-', linewidth=1, alpha=0.5)
ax1.plot(theta_s, np.cos(theta_s)**2, 'ko', markersize=5)
ax1.set_xlabel("θ", fontsize=12)
ax1.set_ylabel("cos²(θ)", fontsize=12)
ax1.set_title(f"Uniform sampling in θ-space ({num_pts} points)", fontsize=13)
ax1.grid(True, alpha=0.2)

# x-space: same dots, now non-uniform
ax2.plot(x, L, 'b-', linewidth=1, alpha=0.5)
ax2.plot(x_s, L_s, 'ko', markersize=5)
ax2.set_xlabel("x", fontsize=12)
ax2.set_ylabel("L(x)", fontsize=12)
ax2.set_title("Mapped to x-space: dense at peak, sparse in tails", fontsize=13)
ax2.set_xlim(-15, 15)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(f"{out}/fig3_point_placement.png", dpi=150)
plt.close()
print("Saved fig3")

# ──────────────────────────────────────────────
# Figure 4: Interpolation quality comparison
# ──────────────────────────────────────────────
x_fine = np.linspace(-10, 10, 50000)
L_true = 1.0 / (1.0 + x_fine**2)
theta_fine = np.arctan(x_fine)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for idx, n in enumerate([10, 20, 50, 100]):
    ax = axes[idx // 2, idx % 2]
    ts = np.linspace(-1.5, 1.5, n)
    cs = np.cos(ts)**2
    xs = np.tan(ts)
    Ls = 1.0 / (1.0 + xs**2)

    # θ-space interpolation (good)
    cos2_interp = np.interp(theta_fine, ts, cs)

    # x-space interpolation (bad)
    sorted_idx = np.argsort(xs)
    L_interp_x = np.interp(x_fine, xs[sorted_idx], Ls[sorted_idx])

    ax.plot(x_fine, L_true, 'k-', linewidth=1, alpha=0.4, label="True")
    ax.plot(x_fine, cos2_interp, 'g-', linewidth=2, label="θ-space interp")
    ax.plot(x_fine, L_interp_x, 'r--', linewidth=1.5, label="x-space interp")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.05, 1.1)
    ax.set_title(f"n = {n} points", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    error_theta = np.max(np.abs(L_true - cos2_interp))
    error_x = np.max(np.abs(L_true - L_interp_x))
    ax.text(0.02, 0.02, f"θ-space max err: {error_theta:.1e}\nx-space max err: {error_x:.1e}",
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle("Interpolation quality: θ-space (green) vs x-space (red)", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{out}/fig4_interpolation_quality.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig4")

# ──────────────────────────────────────────────
# Figure 5: Convergence — error vs number of points
# ──────────────────────────────────────────────
point_counts = [5, 10, 15, 20, 30, 50, 75, 100, 200, 500, 1000]
errors_theta = []
errors_x = []

for n in point_counts:
    ts = np.linspace(-1.5, 1.5, n)
    cs = np.cos(ts)**2
    xs = np.tan(ts)
    Ls = 1.0 / (1.0 + xs**2)

    cos2_interp = np.interp(theta_fine, ts, cs)
    sorted_idx = np.argsort(xs)
    L_interp_x = np.interp(x_fine, xs[sorted_idx], Ls[sorted_idx])

    errors_theta.append(np.max(np.abs(L_true - cos2_interp)))
    errors_x.append(np.max(np.abs(L_true - L_interp_x)))

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(point_counts, errors_theta, 'go-', linewidth=2, markersize=6, label="θ-space interpolation")
ax.semilogy(point_counts, errors_x, 'rs--', linewidth=2, markersize=6, label="x-space interpolation")
ax.set_xlabel("Number of template points", fontsize=12)
ax.set_ylabel("Max absolute error", fontsize=12)
ax.set_title("Convergence: interpolation error vs template size", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axhline(y=1e-3, color='gray', linestyle=':', alpha=0.5)
ax.text(800, 1.5e-3, "0.1% error", fontsize=9, color='gray')
plt.tight_layout()
plt.savefig(f"{out}/fig5_convergence.png", dpi=150)
plt.close()
print("Saved fig5")

# ──────────────────────────────────────────────
# Figure 6: The evaluate() algorithm step by step
# ──────────────────────────────────────────────
n_demo = 20
ts_demo = np.linspace(-1.5, 1.5, n_demo)
cs_demo = np.cos(ts_demo)**2

x_query = 1.8  # example query point
theta_query = np.arctan(x_query)

# Find bracketing indices
t_idx = (theta_query + 1.5) / (2 * 1.5) * (n_demo - 1)
i_lo = int(np.floor(t_idx))
i_hi = i_lo + 1
frac = t_idx - i_lo

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: θ-space view
ax1.plot(theta, cos2, 'r-', linewidth=1, alpha=0.4)
ax1.plot(ts_demo, cs_demo, 'ko', markersize=6)
ax1.axvline(theta_query, color='blue', linestyle='--', alpha=0.7, label=f"θ = arctan({x_query}) = {theta_query:.3f}")
ax1.plot(ts_demo[i_lo], cs_demo[i_lo], 'gs', markersize=12, label=f"Left neighbor (i={i_lo})")
ax1.plot(ts_demo[i_hi], cs_demo[i_hi], 'gs', markersize=12, label=f"Right neighbor (i={i_hi})")

# Show interpolation segment
interp_val = cs_demo[i_lo] * (1 - frac) + cs_demo[i_hi] * frac
ax1.plot([ts_demo[i_lo], ts_demo[i_hi]], [cs_demo[i_lo], cs_demo[i_hi]], 'g-', linewidth=2)
ax1.plot(theta_query, interp_val, 'b*', markersize=15, label=f"Interpolated = {interp_val:.4f}")

ax1.set_xlabel("θ", fontsize=12)
ax1.set_ylabel("cos²(θ)", fontsize=12)
ax1.set_title("Step-by-step: interpolation in θ-space", fontsize=13)
ax1.legend(fontsize=9, loc='lower left')
ax1.grid(True, alpha=0.2)

# Right: what this corresponds to in x-space
true_val = 1.0 / (1.0 + x_query**2)
ax2.plot(x, L, 'b-', linewidth=1, alpha=0.4)
ax2.plot(np.tan(ts_demo), 1.0/(1.0 + np.tan(ts_demo)**2), 'ko', markersize=6)
ax2.axvline(x_query, color='blue', linestyle='--', alpha=0.7, label=f"x = {x_query}")
ax2.plot(x_query, interp_val, 'b*', markersize=15, label=f"Interpolated = {interp_val:.4f}")
ax2.plot(x_query, true_val, 'rx', markersize=12, label=f"True L({x_query}) = {true_val:.4f}")
ax2.set_xlim(-8, 8)
ax2.set_xlabel("x", fontsize=12)
ax2.set_ylabel("L(x)", fontsize=12)
ax2.set_title("Result mapped back to x-space", fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(f"{out}/fig6_algorithm.png", dpi=150)
plt.close()
print("Saved fig6")

print("\nAll figures saved to doc_figures/")
