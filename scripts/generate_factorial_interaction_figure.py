#!/usr/bin/env python3
"""
Generate publication-grade 2x2 Factorial Decomposition Figure for the manuscript.
Visualizes Table 4:
- Panel (a): Interaction Plot & Individual Seed Trajectories across Head Policy x Momentum Policy.
- Panel (b): Marginal Factorial Contrasts Forest Plot (Head Substitution, Momentum Clearing, Interaction).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Color Palette
C_BLUE = '#1F77B4'      # Continuous Momentum
C_GREEN = '#2CA02C'     # Zeroed Momentum
C_RED = '#D62728'       # Head Substitution penalty
C_PURPLE = '#7D3C98'    # Interaction
C_DARK = '#1C2833'
C_GRAY = '#7F8C8D'
C_LIGHT_GRAY = '#BDC3C7'

# Data from Table 4 & Table 13 (N = 6 paired seeds: 920001 - 920006)
f00 = np.array([55.38, 57.19, 57.98, 55.85, 56.52, 55.01])  # Retained Head, Continuous Mom
f01 = np.array([57.77, 56.95, 57.59, 55.79, 57.32, 55.86])  # Retained Head, Zeroed Mom
f10 = np.array([55.47, 54.01, 55.84, 52.73, 53.87, 53.34])  # Temp Head, Continuous Mom
f11 = np.array([56.71, 54.13, 53.67, 53.40, 55.97, 54.86])  # Temp Head, Zeroed Mom

n = len(f00)
df = n - 1
t_crit = stats.t.ppf(0.975, df)

# Means and 95% CIs
m_f00, se_f00 = np.mean(f00), stats.sem(f00)
m_f01, se_f01 = np.mean(f01), stats.sem(f01)
m_f10, se_f10 = np.mean(f10), stats.sem(f10)
m_f11, se_f11 = np.mean(f11), stats.sem(f11)

# Factorial marginal effects per seed
# Head effect: mean over momentum levels of (Temp - Retained)
diff_head = 0.5 * ((f10 - f00) + (f11 - f01))
# Momentum effect: mean over head levels of (Zeroed - Continuous)
diff_mom = 0.5 * ((f01 - f00) + (f11 - f10))
# Interaction: (F11 - F10) - (F01 - F00)
diff_int = (f11 - f10) - (f01 - f00)

effects = [
    ("Head Substitution\n" + r"($\Delta_{\mathrm{head}} = \bar{F}_{1\cdot} - \bar{F}_{0\cdot}$)", diff_head, C_RED),
    ("Momentum Reset\n" + r"($\Delta_{\mathrm{mom}} = \bar{F}_{\cdot 1} - \bar{F}_{\cdot 0}$)", diff_mom, C_GREEN),
    ("Factorial Interaction\n" + r"($\Delta_{\mathrm{interaction}}$)", diff_int, C_PURPLE)
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=300, gridspec_kw={'width_ratios': [1.15, 1.0]})

# =========================================================================
# Panel (a): 2x2 Factorial Interaction Plot
# =========================================================================
x_vals = [0.0, 1.0]
x_labels = [r"Retained Target Head ($h_A$)", r"Temporary Head ($h_{\mathrm{temp}}$)"]

# Plot individual seed trajectories
for i in range(n):
    ax1.plot(x_vals, [f00[i], f10[i]], color=C_BLUE, alpha=0.25, lw=1.1, linestyle='-')
    ax1.plot(x_vals, [f01[i], f11[i]], color=C_GREEN, alpha=0.25, lw=1.1, linestyle='--')

# Plot population means with error bars
ax1.errorbar(x_vals, [m_f00, m_f10], yerr=[se_f00 * t_crit, se_f10 * t_crit],
             fmt='o-', color=C_BLUE, lw=2.4, markersize=8, capsize=5, capthick=1.8,
             label=r"Continuous Momentum ($\mathbf{v}$ intact)", zorder=5)

ax1.errorbar(x_vals, [m_f01, m_f11], yerr=[se_f01 * t_crit, se_f11 * t_crit],
             fmt='s--', color=C_GREEN, lw=2.4, markersize=8, capsize=5, capthick=1.8,
             label=r"Momentum Reset ($\mathbf{v} \leftarrow \mathbf{0}$)", zorder=5)

# Label arm values
ax1.text(0.0 + 0.04, m_f00 - 0.50, f"F00 (C0): {m_f00:.2f}%", ha='left', fontsize=8.8, fontweight='bold', color=C_BLUE)
ax1.text(0.0 + 0.04, m_f01 + 0.35, f"F01: {m_f01:.2f}%", ha='left', fontsize=8.8, fontweight='bold', color=C_GREEN)
ax1.text(1.0 - 0.04, m_f10 - 0.50, f"F10: {m_f10:.2f}%", ha='right', fontsize=8.8, fontweight='bold', color=C_BLUE)
ax1.text(1.0 - 0.04, m_f11 + 0.35, f"F11 (C3): {m_f11:.2f}%", ha='right', fontsize=8.8, fontweight='bold', color=C_GREEN)

# Annotate parallel trajectory / additivity
ax1.annotate("Strictly Parallel Slopes\n(Zero Interaction: +0.02 pp, p = 0.967)",
             xy=(0.5, 55.5), xytext=(0.5, 53.0),
             ha='center', fontsize=8.8, fontweight='bold', color=C_DARK,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='#F8F9F9', edgecolor=C_LIGHT_GRAY, lw=1.0),
             arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.2))

ax1.set_xticks(x_vals)
ax1.set_xticklabels(x_labels, fontsize=10.0, fontweight='bold')
ax1.set_ylabel("Terminal CIFAR-100 Test Accuracy (%)", fontsize=10.5, fontweight='bold')
ax1.set_ylim(52.0, 59.0)
ax1.set_xlim(-0.25, 1.25)
ax1.grid(True, linestyle=':', alpha=0.5, axis='y')
ax1.legend(loc='upper right', framealpha=0.95, fontsize=9.2)
ax1.set_title("(a) Interaction Plot: Head Policy x Momentum Policy (N = 6 Seeds)",
              fontsize=11.0, fontweight='bold', loc='left', pad=10)

# =========================================================================
# Panel (b): Marginal Factorial Contrasts Forest Plot
# =========================================================================
y_pos = np.arange(len(effects))

# Shaded equivalence band [-0.5, +0.5]
ax2.axvspan(-0.5, 0.5, color='#E8F8F5', alpha=0.9, zorder=1, label=r"Practical Equivalence Band ($\pm 0.50$ pp)")
ax2.axvline(0, color=C_DARK, linestyle='--', lw=1.2, zorder=2)

for idx, (label, diff_arr, color) in enumerate(effects):
    mean_val = np.mean(diff_arr)
    se_val = stats.sem(diff_arr)
    ci_t = se_val * t_crit
    t_stat, p_val = stats.ttest_1samp(diff_arr, 0.0)
    
    # Bootstrap CI
    np.random.seed(42 + idx)
    boot_means = [np.mean(np.random.choice(diff_arr, size=n, replace=True)) for _ in range(10000)]
    ci_boot_low, ci_boot_high = np.percentile(boot_means, [2.5, 97.5])
    
    # Outer whisker: Parametric Student's t
    ax2.plot([mean_val - ci_t, mean_val + ci_t], [idx, idx], color=color, lw=2.0, zorder=4)
    ax2.plot([mean_val - ci_t, mean_val - ci_t], [idx - 0.08, idx + 0.08], color=color, lw=2.0, zorder=4)
    ax2.plot([mean_val + ci_t, mean_val + ci_t], [idx - 0.08, idx + 0.08], color=color, lw=2.0, zorder=4)
    
    # Inner thicker bar: Bootstrap CI
    ax2.plot([ci_boot_low, ci_boot_high], [idx, idx], color=color, lw=4.5, solid_capstyle='round', zorder=5)
    
    # Point estimate
    ax2.scatter(mean_val, idx, color=color, s=75, zorder=6, edgecolor='white', lw=1.2)
    
    # Annotation
    sig_str = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
    res_str = f"{mean_val:+.2f} pp [{mean_val-ci_t:.2f}, {mean_val+ci_t:+.2f}] | {sig_str}"
    text_x = mean_val + ci_t + 0.15 if mean_val >= 0 else mean_val - ci_t - 0.15
    ha = 'left' if mean_val >= 0 else 'right'
    ax2.text(text_x, idx, res_str, va='center', ha=ha, fontsize=8.6, fontweight='bold', color=color)

ax2.set_yticks(y_pos)
ax2.set_yticklabels([e[0] for e in effects], fontsize=9.5, fontweight='bold')
ax2.set_xlabel("Paired Difference in Terminal Accuracy (percentage points)", fontsize=10.0, fontweight='bold')
ax2.set_xlim(-4.8, 3.2)
ax2.set_ylim(-0.6, 2.6)
ax2.grid(True, linestyle=':', alpha=0.5, axis='x')
ax2.set_title("(b) Factorial Marginal Main Effects and Interaction (95% CIs)",
              fontsize=11.0, fontweight='bold', loc='left', pad=10)

# Add whisker legend
ax2.plot([], [], color=C_DARK, lw=2.0, label="95% Paired Student's t-CI")
ax2.plot([], [], color=C_DARK, lw=4.5, label="95% Percentile Bootstrap CI")
ax2.legend(loc='lower left', framealpha=0.95, fontsize=8.2)

plt.tight_layout()
pdf_path = os.path.join(FIGURES_DIR, "figure_factorial_decomposition.pdf")
png_path = os.path.join(FIGURES_DIR, "figure_factorial_decomposition.png")
plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
plt.savefig(png_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"Generated 2x2 Factorial figure successfully:\n  {pdf_path}\n  {png_path}")
