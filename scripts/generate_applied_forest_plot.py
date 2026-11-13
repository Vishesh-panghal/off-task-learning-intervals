#!/usr/bin/env python3
"""
Generate publication-grade Applied Vision Transfer Contrasts & Representation Dynamics Figure (Figure 6)
for TMLR, Machine Learning (Springer Nature), and PAA.

Visualizes primary confirmatory statistical data on ISIC 2019 (N = 8 Seeds, 24 runs):
- Panel (a): Seed-Level Paired Slopes on ISIC 2019 (C0 vs. C3 vs. S1)
- Panel (b): Confirmatory Forest Plot across Parametric t & Bootstrap Estimators
- Panel (c): Longitudinal Representation Dynamics (Effective Rank Collapse & Recovery)

Refinements:
1. Panel (a):
   - Encased Grand Mean text labels (85.18%, 84.68%, 84.76%) in protective white rounded pill badges
     so gray seed trajectory lines and scatter markers do not collide with text numbers.
   - Expanded xlim to [-0.45, 2.45] and ylim to [82.9, 87.1] for generous margins.
2. Panel (b):
   - Added protective white background badges behind point estimate labels to stop the red dashed zero
     line from cutting through "+0.08 pp".
   - Relocated legend from lower right to upper right (loc='upper right'), completely eliminating
     whisker occlusion and preventing the red zero line from intersecting the legend box.
3. Panel (c):
   - Repositioned "Complete Recovery" annotation above the Phase 3 milestone (y=1220) pointing downward,
     preventing the arrow from slicing across the S1 orange recovery line.
   - Expanded xlim to [-0.40, 2.40] and ylim to [690, 1290] for complete breathing room.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# Canvas styling matching PAA and TMLR standards
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(BASE_DIR, "results", "applied_primary_runs.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Color Palette consistent with Regime 1
C_BLUE = '#2980B9'      # C0 Continuous Target Baseline
C_PURPLE = '#8E44AD'    # C3 Target Reset Control
C_ORANGE = '#D35400'    # S1 Auxiliary Detour
C_GREEN = '#27AE60'     # Positive contrast / Resilience
C_ROSE = '#C0392B'      # Negative contrast / Degradation
C_DARK = '#1C2833'
C_GRAY = '#7F8C8D'
C_LIGHT_GRAY = '#CBD5E1'

# 1. Load Data
df = pd.read_csv(CSV_PATH)

# Filter for the 8 complete seeds 1001-1008
seeds = [1001 + i for i in range(8)]
df_filtered = df[df['seed'].isin(seeds)].copy()

c0_runs = df_filtered[df_filtered['condition'] == 'C0'].sort_values('seed')
c3_runs = df_filtered[df_filtered['condition'] == 'C3'].sort_values('seed')
s1_runs = df_filtered[df_filtered['condition'] == 'S1'].sort_values('seed')

top1_c0 = c0_runs['theta3_top1'].values
top1_c3 = c3_runs['theta3_top1'].values
top1_s1 = s1_runs['theta3_top1'].values

# Paired differences
d_c3_c0 = top1_c3 - top1_c0
d_s1_c3 = top1_s1 - top1_c3
d_s1_c0 = top1_s1 - top1_c0

# Effective Ranks across phases
eff_p1_c0 = c0_runs['theta1_eff_rank'].values
eff_p2_c0 = c0_runs['theta2_eff_rank'].values
eff_p3_c0 = c0_runs['theta3_eff_rank'].values

eff_p1_c3 = c3_runs['theta1_eff_rank'].values
eff_p2_c3 = c3_runs['theta2_eff_rank'].values
eff_p3_c3 = c3_runs['theta3_eff_rank'].values

eff_p1_s1 = s1_runs['theta1_eff_rank'].values
eff_p2_s1 = s1_runs['theta2_eff_rank'].values
eff_p3_s1 = s1_runs['theta3_eff_rank'].values

# Setup figure canvas (3 panels)
fig = plt.figure(figsize=(18.0, 5.6), dpi=300)
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.25, 1.15], wspace=0.36)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

# =========================================================================
# PANEL A: Seed-Level Trajectories on ISIC 2019 (N = 8 Seeds)
# =========================================================================
x_c0, x_c3, x_s1 = 0.0, 1.0, 2.0
xs = [x_c0, x_c3, x_s1]

for i in range(len(seeds)):
    y_vals = [top1_c0[i], top1_c3[i], top1_s1[i]]
    ax1.plot(xs, y_vals, color='#CBD5E1', lw=1.2, alpha=0.75, zorder=2)
    ax1.scatter(xs, y_vals, color=[C_BLUE, C_PURPLE, C_ORANGE], s=35, alpha=0.85, zorder=3)

# Grand Means
mean_c0, mean_c3, mean_s1 = np.mean(top1_c0), np.mean(top1_c3), np.mean(top1_s1)
ax1.plot(xs, [mean_c0, mean_c3, mean_s1], color=C_DARK, lw=3.0, marker='D', markersize=7.5, zorder=5, label='Grand Mean')

# Badged mean labels to prevent clashing with seed points/lines
ax1.text(x_c0, mean_c0 + 0.38, f"{mean_c0:.2f}%", ha='center', va='bottom', fontsize=8.2, fontweight='bold', color=C_BLUE,
         bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor=C_BLUE, alpha=0.94, lw=0.8), zorder=6)
ax1.text(x_c3, mean_c3 - 0.42, f"{mean_c3:.2f}%", ha='center', va='top', fontsize=8.2, fontweight='bold', color=C_PURPLE,
         bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor=C_PURPLE, alpha=0.94, lw=0.8), zorder=6)
ax1.text(x_s1, mean_s1 + 0.38, f"{mean_s1:.2f}%", ha='center', va='bottom', fontsize=8.2, fontweight='bold', color=C_ORANGE,
         bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor=C_ORANGE, alpha=0.94, lw=0.8), zorder=6)

ax1.set_xlim(-0.45, 2.45)
ax1.set_ylim(82.9, 87.1)
ax1.set_xticks(xs)
ax1.set_xticklabels([
    r"$\mathbf{C_0}$ Continuous" + "\n(12k Target)",
    r"$\mathbf{C_3}$ Reset Ctrl" + "\n(Head Swap)",
    r"$\mathbf{S_1}$ Auxiliary" + "\n(2k Detour)"
], fontsize=8.5, fontweight='bold')
ax1.set_ylabel("Terminal ISIC 2019 Top-1 Accuracy (%)", fontsize=9.0, fontweight='bold', color=C_DARK)
ax1.set_title("(a) Seed-Level Paired Profiles (N = 8 Seeds)", fontsize=10.0, fontweight='bold', color=C_DARK, pad=10)
ax1.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1', axis='y')

ax1.text(1.0, 83.15, "Pairwise Equivalence:\nAll pairwise contrasts |Δ| ≤ 0.49 pp\nAll p > 0.20 (No Detour Penalty)",
         ha='center', va='bottom', fontsize=7.8, color='#2C3E50',
         bbox=dict(boxstyle="round,pad=0.35", facecolor='#F8F9F9', edgecolor=C_LIGHT_GRAY, lw=1.0), zorder=4)

# =========================================================================
# PANEL B: Confirmatory Forest Plot across Parametric & Bootstrap Estimators
# =========================================================================
contrasts = [
    r"$C_3(12\mathrm{k}) - C_0(12\mathrm{k})$" + "\n(Procedural Reset)",
    r"$S_1(12\mathrm{k}) - C_3(12\mathrm{k})$" + "\n(Detour vs. Reset)",
    r"$S_1(12\mathrm{k}) - C_0(12\mathrm{k})$" + "\n(Detour vs. Baseline)"
]
diff_arrays = [d_c3_c0, d_s1_c3, d_s1_c0]
y_pos = np.arange(len(contrasts))[::-1]

# Zero reference line and equivalence bounds
ax2.axvline(0.0, color=C_ROSE, linestyle='--', lw=1.3, zorder=1)
ax2.axvspan(-0.5, 0.5, facecolor='#EAFAF1', alpha=0.45, zorder=0)

np.random.seed(42)
for i, y in enumerate(y_pos):
    arr = diff_arrays[i]
    m = np.mean(arr)
    se = stats.sem(arr)
    t_crit = stats.t.ppf(0.975, df=len(arr)-1)
    t_lo, t_hi = m - t_crit * se, m + t_crit * se
    
    # Bootstrap 95% percentile interval (10k resamples)
    boot_means = np.random.choice(arr, size=(10000, len(arr)), replace=True).mean(axis=1)
    b_lo, b_hi = np.percentile(boot_means, [2.5, 97.5])
    p_val = stats.ttest_1samp(arr, 0.0)[1]
    
    col = C_BLUE if i == 1 else (C_PURPLE if i == 0 else C_ORANGE)
    
    # 1. Parametric Student's t 95% CI (outer whisker with caps)
    ax2.plot([t_lo, t_hi], [y + 0.08, y + 0.08], color=col, lw=2.2, zorder=3)
    ax2.plot([t_lo, t_lo], [y + 0.02, y + 0.14], color=col, lw=2.2, zorder=3)
    ax2.plot([t_hi, t_hi], [y + 0.02, y + 0.14], color=col, lw=2.2, zorder=3)
    
    # 2. Bootstrap 95% CI (inner bar)
    ax2.plot([b_lo, b_hi], [y - 0.08, y - 0.08], color=C_DARK, lw=3.2, alpha=0.85, zorder=3)
    
    # Point estimate marker
    ax2.scatter(m, y, color=col, s=65, edgecolors=C_DARK, lw=1.2, zorder=5)
    
    # Text annotation with protective white badge so zero reference line doesn't cut through
    sign_m = f"+{m:.2f}" if m > 0 else f"{m:.2f}"
    ax2.text(m, y + 0.22, f"{sign_m} pp (p = {p_val:.3f})", ha='center', va='bottom', fontsize=7.8, fontweight='bold', color=C_DARK,
             bbox=dict(boxstyle="round,pad=0.18", facecolor='white', edgecolor='none', alpha=1.0), zorder=6)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(contrasts, fontsize=8.5, fontweight='bold')
ax2.set_xlim(-1.8, 1.4)
ax2.set_ylim(-0.6, 2.9)
ax2.set_xlabel("Paired Difference in Terminal Top-1 Accuracy (pp)", fontsize=9.0, fontweight='bold', color=C_DARK)
ax2.set_title("(b) Confirmatory Contrasts & Inferential Sensitivity", fontsize=10.0, fontweight='bold', color=C_DARK, pad=10)
ax2.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')

# Custom Legend placed in UPPER RIGHT (completely clear of all whiskers and zero line)
line_t = plt.Line2D([0], [0], color=C_BLUE, lw=2.2, label="Parametric Student's t 95% CI (df=7)")
line_b = plt.Line2D([0], [0], color=C_DARK, lw=3.2, label="Percentile Bootstrap 95% CI (10k)")
line_zero = plt.Line2D([0], [0], color=C_ROSE, linestyle='--', lw=1.3, label="Null Effect Line (Zero Difference)")
ax2.legend(handles=[line_t, line_b, line_zero], loc='upper right', frameon=True, framealpha=0.94, fontsize=7.4, edgecolor=C_LIGHT_GRAY)

# =========================================================================
# PANEL C: Representation Dynamics (Transient Collapse & Complete Recovery)
# =========================================================================
step_labels = ["Phase 1 (θ₁)\n5,000 Upd", "Phase 2 (θ₂)\n7,000 Upd", "Phase 3 (θ₃)\n12,000 Upd"]

m_eff_c0 = [np.mean(eff_p1_c0), np.mean(eff_p2_c0), np.mean(eff_p3_c0)]
m_eff_c3 = [np.mean(eff_p1_c3), np.mean(eff_p2_c3), np.mean(eff_p3_c3)]
m_eff_s1 = [np.mean(eff_p1_s1), np.mean(eff_p2_s1), np.mean(eff_p3_s1)]

s_eff_c0 = [np.std(eff_p1_c0), np.std(eff_p2_c0), np.std(eff_p3_c0)]
s_eff_c3 = [np.std(eff_p1_c3), np.std(eff_p2_c3), np.std(eff_p3_c3)]
s_eff_s1 = [np.std(eff_p1_s1), np.std(eff_p2_s1), np.std(eff_p3_s1)]

ax3.errorbar([0, 1, 2], m_eff_c0, yerr=s_eff_c0, fmt='-o', color=C_BLUE, lw=2.2, capsize=4, label=r"$C_0$ Continuous Target", zorder=4)
ax3.errorbar([0, 1, 2], m_eff_c3, yerr=s_eff_c3, fmt='-s', color=C_PURPLE, lw=2.2, capsize=4, label=r"$C_3$ Target Reset Ctrl", zorder=4)
ax3.errorbar([0, 1, 2], m_eff_s1, yerr=s_eff_s1, fmt='-^', color=C_ORANGE, lw=2.5, capsize=4, label=r"$S_1$ Auxiliary Detour", zorder=5)

# Highlight collapse at Phase 2 (cleanly centered inside the V-trough, clear of all lines)
ax3.annotate(r"$\mathbf{-359.8\ dims}$" + "\nCollapse ($p < 10^{-10}$)",
             xy=(1, m_eff_s1[1]), xytext=(1.0, 940),
             arrowprops=dict(facecolor=C_ROSE, edgecolor=C_ROSE, shrinkA=8, shrinkB=12, width=1.5, headwidth=5.5),
             fontsize=7.8, fontweight='bold', color=C_ROSE, ha='center',
             bbox=dict(boxstyle='round,pad=0.28', facecolor='#FDEDEC', edgecolor=C_ROSE, lw=0.9), zorder=6)

# Highlight complete recovery at Phase 3 (placed cleanly above milestone with shrinkB to avoid marker touch)
ax3.annotate(r"$\mathbf{Complete\ Recovery}$" + "\n(Stat. Parity at 1140 dims)",
             xy=(2, m_eff_s1[2]), xytext=(1.68, 1225),
             arrowprops=dict(facecolor=C_GREEN, edgecolor=C_GREEN, shrinkA=5, shrinkB=20, width=1.5, headwidth=5.5),
             fontsize=7.8, fontweight='bold', color=C_GREEN, ha='center',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#EAFAF1', edgecolor=C_GREEN, lw=0.9), zorder=6)

ax3.set_xticks([0, 1, 2])
ax3.set_xticklabels(step_labels, fontsize=8.5, fontweight='bold')
ax3.set_xlim(-0.40, 2.40)
ax3.set_ylim(690, 1290)
ax3.set_ylabel("Probe Effective Rank (Dimensions)", fontsize=9.0, fontweight='bold', color=C_DARK)
ax3.set_title("(c) Longitudinal Representation Dimensionality", fontsize=10.0, fontweight='bold', color=C_DARK, pad=10)
ax3.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')
ax3.legend(loc='lower left', frameon=True, framealpha=0.94, fontsize=7.6, edgecolor=C_LIGHT_GRAY)

plt.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.14)

png_out = os.path.join(FIGURES_DIR, "figure_applied_contrasts_forest_plot.png")
pdf_out = os.path.join(FIGURES_DIR, "figure_applied_contrasts_forest_plot.pdf")

plt.savefig(png_out, dpi=300)
plt.savefig(pdf_out)
plt.close()

print(f"✓ Saved Applied Contrasts PNG: {png_out}")
print(f"✓ Saved Applied Contrasts PDF: {pdf_out}")
