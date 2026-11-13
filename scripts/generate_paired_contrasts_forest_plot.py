#!/usr/bin/env python3
"""
Generate publication-grade Figure 2: Confirmatory Paired Contrasts on CIFAR-100.
Redesigned with native 6.5-inch width and comfortable vertical height (figsize 6.5 x 2.05 in):
- Panel (a): Seed-Level Paired Profiles (N = 20 paired seeds).
- Panel (b): Confirmatory Paired Contrasts with 95% t-intervals and bootstrap intervals.
- Panel (c) (Permutation Null Distribution) exported as supplementary figure.

Typography:
- Panel titles: 9.2 pt bold
- Axis labels: 8.6 pt bold
- Tick labels: 7.4--7.8 pt bold
- Legend & point tags: 7.2--7.4 pt
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Typography matching TMLR publication standards
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
CLEAN_FIGURES_DIR = os.path.join(BASE_DIR, "clean_repo", "figures")
for d in [FIGURES_DIR, CLEAN_FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

C_BLUE = '#1F77B4'      # C0 Continuous Target Baseline
C_GREEN = '#2CA02C'     # C3 Target Reset Control / Positive
C_RED = '#D62728'       # S1 Auxiliary STL-10 Detour / Negative
C_PURPLE = '#7D3C98'    # Bootstrap CI
C_DARK = '#0F172A'      # Slate 900
C_MUTED = '#475569'     # Slate 600

# Seed data from Table 2 (N = 20 Paired Seeds)
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

mean_c0 = np.mean(c0_35k)   # 56.69%
mean_s1 = np.mean(s1_35k)   # 56.26%
mean_diff = np.mean(diffs_primary)  # -0.4325 pp

contrasts = [
    r"$\mathbf{S_1(35k) - C_0(35k)}$" + "\n(Primary Contrast)",
    r"$\mathbf{S_1(35k) - C_3(35k)}$" + "\n(Detour vs. Reset)",
    r"$\mathbf{C_3(35k) - C_0(35k)}$" + "\n(Procedural Reset)",
    r"$\mathbf{S_1(35k) - C_0(30k)}$" + "\n(Target Matched)"
]
mean_diffs = [-0.4325, -0.0300, -0.4025, +0.2660]
t_ci_low = [-0.9390, -0.7618, -1.1424, -0.4691]
t_ci_high = [+0.0740, +0.7018, +0.3374, +1.0011]
boot_ci_low = [-0.8880, -0.6700, -1.1200, -0.4300]
boot_ci_high = [+0.0200, +0.6700, +0.2400, +0.9200]

# =========================================================================
# MAIN MANUSCRIPT FIGURE 2: 2 Panels with enhanced vertical height (+32%)
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.10), dpi=300,
                               gridspec_kw={'width_ratios': [0.90, 1.35], 'wspace': 0.52})

# --- Panel (a): Seed-Level Paired Slopes ---
x_c0, x_s1 = 0.0, 1.0
for i in range(len(seeds)):
    y0, y1 = c0_35k[i], s1_35k[i]
    d = diffs_primary[i]
    col = C_GREEN if d >= 0 else C_RED
    ax1.plot([x_c0, x_s1], [y0, y1], color=col, lw=0.90, alpha=0.45, zorder=3)
    ax1.scatter([x_c0, x_s1], [y0, y1], color=col, s=14, alpha=0.60, zorder=4)

# Grand Mean Trajectory
ax1.plot([x_c0, x_s1], [mean_c0, mean_s1], color=C_DARK, lw=2.4, marker='D', markersize=5.5, zorder=6,
         label=rf"Mean ($\Delta = {mean_diff:.2f}$ pp)")
ax1.text(x_c0 - 0.04, mean_c0 + 0.35, f"{mean_c0:.2f}%", ha='right', va='bottom', fontsize=7.6, fontweight='bold', color=C_DARK)
ax1.text(x_s1 + 0.04, mean_s1 - 0.35, f"{mean_s1:.2f}%\n({mean_diff:+.2f} pp)", ha='left', va='top', fontsize=7.6, fontweight='bold', color=C_DARK)

ax1.set_xlim(-0.35, 1.45)
ax1.set_ylim(51.0, 60.5)
ax1.set_xticks([x_c0, x_s1])
ax1.set_xticklabels([r"$\mathbf{C_0}$ Continuous" + "\n(35k Updates)", r"$\mathbf{S_1}$ Auxiliary" + "\n(35k Updates)"],
                    fontsize=7.8, fontweight='bold')
ax1.set_ylabel("Terminal CIFAR-100 Acc. (%)", fontsize=8.6, fontweight='bold', color=C_DARK, labelpad=2.0)
ax1.set_title("(a) Seed-Level Paired Profiles (N = 20)", fontsize=9.2, fontweight='bold', color=C_DARK, pad=6)

# Place legend in upper left to eliminate any possibility of collision with panel (b)
ax1.legend(loc='upper left', frameon=True, framealpha=0.92, fontsize=7.2, edgecolor='#CBD5E1')
ax1.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1', axis='y')

# Subtle note on seed distribution
ax1.text(0.5, 51.4, r"7 positive / 13 negative seeds ($p = 0.090$)", ha='center', va='bottom', fontsize=7.0, color=C_MUTED)

# --- Panel (b): Confirmatory Contrasts Forest Plot ---
y_pos = np.arange(len(contrasts))[::-1]
ax2.axvline(0.0, color=C_RED, linestyle='--', lw=1.1, zorder=1)
ax2.axvspan(-0.5, 0.5, facecolor='#EAFAF1', alpha=0.35, zorder=0)

for i, y in enumerate(y_pos):
    m = mean_diffs[i]
    t_lo, t_hi = t_ci_low[i], t_ci_high[i]
    b_lo, b_hi = boot_ci_low[i], boot_ci_high[i]
    is_primary = (i == 0)
    col = C_BLUE if is_primary else C_MUTED
    pt_col = C_DARK if is_primary else C_MUTED
    
    # Parametric t CI (outer line with ticks)
    ax2.plot([t_lo, t_hi], [y + 0.07, y + 0.07], color=col, lw=1.6, zorder=3)
    ax2.plot([t_lo, t_lo], [y + 0.02, y + 0.12], color=col, lw=1.6, zorder=3)
    ax2.plot([t_hi, t_hi], [y + 0.02, y + 0.12], color=col, lw=1.6, zorder=3)
    
    # Bootstrap CI (inner thicker line)
    ax2.plot([b_lo, b_hi], [y - 0.07, y - 0.07], color=C_PURPLE, lw=2.6, alpha=0.85, zorder=3)
    
    # Point estimate
    ax2.scatter(m, y, color=pt_col, s=36, zorder=5)
    
    # Direct number tag placed cleanly with ample legend clearance
    sign_m = f"{m:+.2f} pp" if m != 0 else "0.00 pp"
    if i == 1: # -0.03 pp: align right of point to preserve clear space before the legend
        ax2.text(m - 0.08, y + 0.20, sign_m, ha='right', va='bottom', fontsize=7.4, fontweight='bold', color=C_DARK,
                 bbox=dict(boxstyle="round,pad=0.10", facecolor='white', edgecolor='none', alpha=0.90, zorder=4))
    else:
        ax2.text(m, y + 0.20, sign_m, ha='center', va='bottom', fontsize=7.4, fontweight='bold', color=C_DARK,
                 bbox=dict(boxstyle="round,pad=0.10", facecolor='white', edgecolor='none', alpha=0.90, zorder=4))

ax2.set_yticks(y_pos)
ax2.set_yticklabels(contrasts, fontsize=7.4)
ax2.set_xlim(-1.4, 1.35)
ax2.set_ylim(-0.6, 3.8)
ax2.set_xlabel("Paired Difference in Test Accuracy (pp)", fontsize=8.6, fontweight='bold', color=C_DARK)
ax2.set_title("(b) Confirmatory Paired Contrasts", fontsize=9.2, fontweight='bold', color=C_DARK, pad=6)
ax2.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')

# Legend placed in upper right with no obstruction
line_t = plt.Line2D([0], [0], color=C_BLUE, lw=1.6, label="95% Paired t-CI")
line_b = plt.Line2D([0], [0], color=C_PURPLE, lw=2.4, label="95% Bootstrap CI")
line_z = plt.Line2D([0], [0], color=C_RED, linestyle='--', lw=1.1, label="Null (0 pp)")
ax2.legend(handles=[line_t, line_b, line_z], loc='upper right', frameon=True, framealpha=0.92, fontsize=7.2, edgecolor='#CBD5E1')

plt.subplots_adjust(left=0.125, right=0.98, top=0.88, bottom=0.18)

pdf_out = os.path.join(FIGURES_DIR, "figure_paired_contrasts_forest_plot.pdf")
png_out = os.path.join(FIGURES_DIR, "figure_paired_contrasts_forest_plot.png")
fig.savefig(pdf_out, bbox_inches='tight', pad_inches=0.05)
fig.savefig(png_out, bbox_inches='tight', pad_inches=0.05, dpi=300)

for d in [CLEAN_FIGURES_DIR]:
    fig.savefig(os.path.join(d, "figure_paired_contrasts_forest_plot.pdf"), bbox_inches='tight', pad_inches=0.05)
    fig.savefig(os.path.join(d, "figure_paired_contrasts_forest_plot.png"), bbox_inches='tight', pad_inches=0.05, dpi=300)

plt.close(fig)
print(f"Figure 2 successfully generated: {pdf_out}")
