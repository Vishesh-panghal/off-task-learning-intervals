#!/usr/bin/env python3
"""
Generate publication-grade Paired Contrasts & Inferential Sensitivity Figure
for Pattern Analysis and Applications (PAA, Springer Nature).

Visualizes Table 3 and Table 4 confirmatory statistical data:
- Panel (a): Seed-Level Paired Slopes (N = 20 Seeds, C0 vs. S1)
- Panel (b): Confirmatory Forest Plot across Parametric & Bootstrap Estimators
- Panel (c): Exact 1,048,576-Point Sign-Flip Permutation Null Distribution

Features:
- Full N = 20 paired seeds from Table 3 (7 positive vs. 13 negative)
- Dual CI whiskers: Parametric Student's t (95%) and Percentile Bootstrap (95%)
- Exact combinatorial permutation histogram (2^20 = 1,048,576 outcomes, p = 0.0906)
- High-contrast PAA palette matching Figures 1 & 2
- Dual export: 300 DPI PNG + vector PDF
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Canvas styling
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../figures"))
os.makedirs(FIGURES_DIR, exist_ok=True)

# Color Palette matching Figure 1
C_BLUE = '#2980B9'      # C0 Continuous Target Baseline
C_PURPLE = '#8E44AD'    # C3 Target Reset Control
C_ORANGE = '#D35400'    # S1 Auxiliary STL-10 Interval
C_GREEN = '#27AE60'     # Positive contrast / Success
C_ROSE = '#C0392B'      # Negative contrast / Null
C_DARK = '#1C2833'
C_GRAY = '#7F8C8D'

# Seed data from Table 3 (N = 20 Paired Seeds)
seeds = [f"9200{i:02d}" for i in range(1, 21)]
c0_35k = np.array([
    55.38, 57.19, 57.98, 55.85, 56.52, 55.01,
    57.35, 56.95, 56.95, 58.06, 56.35, 56.35,
    57.05, 56.45, 57.15, 56.75, 56.60, 56.20,
    57.10, 56.65
])
s1_35k = np.array([
    55.45, 57.48, 56.51, 53.69, 57.19, 52.25,
    57.31, 56.25, 59.07, 57.26, 55.17, 55.16,
    56.85, 56.05, 57.55, 55.85, 57.20, 55.45,
    57.45, 56.05
])
diffs_primary = s1_35k - c0_35k

# Grand means
mean_c0 = np.mean(c0_35k)   # 56.69%
mean_s1 = np.mean(s1_35k)   # 56.26%
mean_diff = np.mean(diffs_primary)  # -0.4325 pp

# Table 4 Contrast Estimands (Two-line format for optimal spacing)
contrasts = [
    r"$S_1(35{\rm k}) - C_0(35{\rm k})$" + "\n(Primary)",
    r"$S_1(35{\rm k}) - C_3(35{\rm k})$" + "\n(vs. Reset)",
    r"$C_3(35{\rm k}) - C_0(35{\rm k})$" + "\n(Reset Cost)",
    r"$S_1(35{\rm k}) - C_0(30{\rm k})$" + "\n(Matched Pre-P3)"
]
mean_diffs = [-0.4325, -0.0300, -0.4025, +0.2660]
t_ci_low = [-0.9390, -0.7618, -1.1424, -0.4691]
t_ci_high = [+0.0740, +0.7018, +0.3374, +1.0011]
boot_ci_low = [-0.8880, -0.6700, -1.1200, -0.4300]
boot_ci_high = [+0.0200, +0.6700, +0.2400, +0.9200]

# Canvas Setup: 3 Panels
fig = plt.figure(figsize=(17.4, 5.8), dpi=300)
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.38, 0.95], wspace=0.42)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

# =========================================================================
# PANEL A: Seed-Level Paired Slopes (N = 20 Seeds)
# =========================================================================
x_c0 = 0.0
x_s1 = 1.0

# Plot individual seed trajectories
for i in range(len(seeds)):
    y0 = c0_35k[i]
    y1 = s1_35k[i]
    d = diffs_primary[i]
    is_pos = (d >= 0)
    col = C_GREEN if is_pos else C_ROSE
    
    # Paired slope line
    ax1.plot([x_c0, x_s1], [y0, y1], color=col, lw=1.2, alpha=0.55, zorder=3)
    ax1.scatter([x_c0, x_s1], [y0, y1], color=col, s=22, alpha=0.70, zorder=4)

# Grand Mean Trajectory (Thick diamond line)
ax1.plot([x_c0, x_s1], [mean_c0, mean_s1], color=C_DARK, lw=3.4, marker='D', markersize=8, zorder=6, label=r"Grand Mean ($\Delta = -0.43$ pp)")
ax1.text(x_c0 - 0.05, mean_c0 + 0.35, f"Mean: {mean_c0:.2f}%", ha='right', va='bottom', fontsize=8.2, fontweight='bold', color=C_DARK)
ax1.text(x_s1 + 0.05, mean_s1 - 0.35, f"Mean: {mean_s1:.2f}%\n({mean_diff:+.2f} pp)", ha='left', va='top', fontsize=8.2, fontweight='bold', color=C_DARK)

ax1.set_xlim(-0.35, 1.45)
ax1.set_ylim(51.0, 60.5)
ax1.set_xticks([x_c0, x_s1])
ax1.set_xticklabels([r"Continuous Baseline $C_0$" + "\n(35k Updates)", r"Auxiliary Schedule $S_1$" + "\n(35k Updates)"], fontsize=8.2, fontweight='bold')
ax1.set_ylabel("Terminal CIFAR-100 Test Accuracy (%)", fontsize=8.8, fontweight='bold', color=C_DARK)
ax1.set_title("(a) Seed-Level Paired Slopes (N = 20 Seeds)", fontsize=9.6, fontweight='bold', color=C_DARK, pad=10)
ax1.legend(loc='upper right', frameon=True, framealpha=0.92, fontsize=7.8, edgecolor='#BDC3C7')
ax1.grid(True, linestyle=':', alpha=0.5, color='#BDC3C7', axis='y')

# Annotations showing balanced variation
ax1.text(0.5, 51.5, "Seed Variation (N = 20):\n7 Positive Seeds (+0.07 to +2.12 pp)\n13 Negative Seeds (-0.04 to -2.76 pp)\n$t(19) = -1.79, p = 0.090$",
         ha='center', va='bottom', fontsize=7.5, color='#2C3E50',
         bbox=dict(boxstyle="round,pad=0.4", facecolor='#F8F9F9', edgecolor='#BDC3C7', lw=1.0))

# =========================================================================
# PANEL B: Confirmatory Forest Plot across Parametric & Bootstrap Estimators
# =========================================================================
y_pos = np.arange(len(contrasts))[::-1]  # Top to bottom

# Zero reference line
ax2.axvline(0.0, color=C_ROSE, linestyle='--', lw=1.3, zorder=1)
ax2.axvspan(-0.5, 0.5, facecolor='#EAFAF1', alpha=0.35, zorder=0)  # Shaded bounds

for i, y in enumerate(y_pos):
    m = mean_diffs[i]
    t_lo, t_hi = t_ci_low[i], t_ci_high[i]
    b_lo, b_hi = boot_ci_low[i], boot_ci_high[i]
    
    col = C_BLUE if i == 0 else C_GRAY
    point_col = C_DARK if i == 0 else C_GRAY
    
    # 1. Parametric Student's t 95% CI (Thinner outer bar with caps)
    ax2.plot([t_lo, t_hi], [y + 0.08, y + 0.08], color=col, lw=2.0, zorder=3)
    ax2.plot([t_lo, t_lo], [y + 0.02, y + 0.14], color=col, lw=2.0, zorder=3)
    ax2.plot([t_hi, t_hi], [y + 0.02, y + 0.14], color=col, lw=2.0, zorder=3)
    
    # 2. Bootstrap 95% CI (Thicker inner bar)
    ax2.plot([b_lo, b_hi], [y - 0.08, y - 0.08], color=C_PURPLE, lw=3.0, alpha=0.85, zorder=3)
    
    # Point estimate marker
    ax2.scatter(m, y, color=point_col, s=55, zorder=5)
    
    # Text label for point estimate & CI
    sign_m = f"+{m:.2f}" if m > 0 else f"{m:.2f}"
    ax2.text(m, y + 0.22, f"{sign_m} pp", ha='center', va='bottom', fontsize=7.6, fontweight='bold', color=C_DARK)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(contrasts, fontsize=8.0, fontweight='bold')
ax2.set_xlim(-1.6, 1.4)
ax2.set_ylim(-0.6, 3.8)
ax2.set_xlabel("Paired Difference in Terminal Test Accuracy (pp)", fontsize=8.8, fontweight='bold', color=C_DARK)
ax2.set_title("(b) Confirmatory Forest Plot & Inferential Sensitivity", fontsize=9.6, fontweight='bold', color=C_DARK, pad=10)
ax2.grid(True, linestyle=':', alpha=0.5, color='#BDC3C7')

# Custom Legend for Estimators
line_t = plt.Line2D([0], [0], color=C_BLUE, lw=2.0, label="Parametric Student's t 95% CI")
line_b = plt.Line2D([0], [0], color=C_PURPLE, lw=3.0, label="Percentile Bootstrap 95% CI (10k resamples)")
line_zero = plt.Line2D([0], [0], color=C_ROSE, linestyle='--', lw=1.3, label="Null Effect Line (Zero Difference)")
ax2.legend(handles=[line_t, line_b, line_zero], loc='lower right', frameon=True, framealpha=0.92, fontsize=7.5, edgecolor='#BDC3C7')

# =========================================================================
# PANEL C: Exact 1,048,576-Point Sign-Flip Permutation Null Distribution
# =========================================================================
n_seeds = 20
total_combinations = 1 << n_seeds
indices = np.arange(total_combinations, dtype=np.uint32)
weights = np.abs(diffs_primary)
obs_stat = np.mean(diffs_primary)

bit_sums = np.zeros(total_combinations, dtype=np.float64)
for i in range(n_seeds):
    mask = (indices >> i) & 1
    bit_sums += mask * weights[i]

perm_means = (2 * bit_sums - np.sum(weights)) / n_seeds
extreme_count = np.sum(np.abs(perm_means) >= np.abs(obs_stat) - 1e-9)
p_perm = extreme_count / total_combinations

# Histogram of permutation null distribution
n, bins, patches_hist = ax3.hist(perm_means, bins=35, edgecolor=C_DARK, linewidth=0.8, facecolor='#EBF5FB', zorder=3)

# Highlight extreme tails (|d| >= 0.4325)
for patch, bin_left, bin_right in zip(patches_hist, bins[:-1], bins[1:]):
    if bin_right <= -np.abs(obs_stat) or bin_left >= np.abs(obs_stat):
        patch.set_facecolor('#FADBD8')
        patch.set_edgecolor(C_ROSE)

# Observed statistic line
ax3.axvline(obs_stat, color=C_ROSE, linestyle='-', lw=2.0, zorder=5)
ax3.axvline(-obs_stat, color=C_ROSE, linestyle=':', lw=1.5, zorder=5)

y_max = ax3.get_ylim()[1]
ax3.annotate(f"Observed Difference\n{obs_stat:.2f} pp", xy=(obs_stat, y_max * 0.65), xytext=(-0.75, y_max * 0.82),
             arrowprops=dict(arrowstyle="->", color=C_ROSE, lw=1.3), fontsize=7.5, fontweight='bold', color=C_ROSE, ha='center',
             bbox=dict(boxstyle="round,pad=0.25", facecolor='white', edgecolor=C_ROSE, lw=1.0))

# Results badge
ax3.text(0.5, 0.94, f"Exact Permutation Test:\n1,048,576 Sign Permutations ($2^{{20}}$)\nExtreme Tails: {extreme_count:,}/1,048,576\nExact $p = {p_perm:.4f}$",
         transform=ax3.transAxes, ha='center', va='top', fontsize=7.6, fontweight='bold', color=C_DARK,
         bbox=dict(boxstyle="round,pad=0.4", facecolor='#FFFFFF', edgecolor='#1C2833', lw=1.2))

ax3.set_xlim(-1.2, 1.2)
ax3.set_xlabel("Mean Paired Difference Under Null (pp)", fontsize=8.8, fontweight='bold', color=C_DARK)
ax3.set_ylabel("Permutation Count", fontsize=8.8, fontweight='bold', color=C_DARK)
ax3.set_title("(c) Exact Sign-Flip Permutation Null", fontsize=9.6, fontweight='bold', color=C_DARK, pad=10)
ax3.grid(True, linestyle=':', alpha=0.5, color='#BDC3C7')

# Save figures
png_out = os.path.join(FIGURES_DIR, "figure_paired_contrasts_forest_plot.png")
pdf_out = os.path.join(FIGURES_DIR, "figure_paired_contrasts_forest_plot.pdf")

plt.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.14)
plt.savefig(png_out, dpi=300)
plt.savefig(pdf_out)
plt.close()

print(f"✓ Saved 20-Seed Forest Plot PNG: {png_out} ({os.path.getsize(png_out)/1024:.1f} KB)")
print(f"✓ Saved 20-Seed Forest Plot PDF: {pdf_out} ({os.path.getsize(pdf_out)/1024:.1f} KB)")
