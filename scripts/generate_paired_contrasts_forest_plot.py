#!/usr/bin/env python3
"""
Generate publication-grade Figure 2: Confirmatory Paired Contrasts on CIFAR-100.
Authored in authentic Seaborn statistical style (sns.set_theme(style="ticks")):
- Panel (a): Seed-Level Paired Profiles (N = 20 paired seeds).
  * Subtle individual seed trajectories in neutral slate gray showing real empirical pairs.
  * Bold Grand Mean trajectory in Seaborn Deep Blue with circular marker.
  * Natural, unboxed numeric labels without artificial rounded bounding boxes.
  * Crisp L-frame despined axes with outward ticks and soft horizontal grid.
- Panel (b): Confirmatory Paired Contrasts Forest Plot (95% CI).
  * 4 spacious contrast rows with parametric 95% t-intervals and bootstrap intervals.
  * Clean numeric annotations directly above markers in matching palette colors.
  * Soft desaturated equivalence band (±0.5 pp) and zero reference line stopping below the header.
  * Clean, borderless horizontal legend placed outside above the plot frame.
- Supplementary Figure S1 (Permutation Null Distribution) exported with matching Seaborn aesthetics.

Typography & Design:
- Genuine Seaborn Deep palette: #4c72b0 (blue), #55a868 (green), #c44e52 (coral), #8172b3 (purple).
- Standard academic hierarchy: semibold titles, medium axis labels, normal weight ticks/values.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set authentic Seaborn publication theme
sns.set_theme(style="ticks", font="sans-serif")
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['mathtext.fontset'] = 'cm'

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.basename(BASE_DIR) == "clean_repo":
    FIGURES_DIR = os.path.join(BASE_DIR, "figures")
    CLEAN_FIGURES_DIR = FIGURES_DIR
else:
    FIGURES_DIR = os.path.join(BASE_DIR, "figures")
    CLEAN_FIGURES_DIR = os.path.join(BASE_DIR, "clean_repo", "figures")
for d in [FIGURES_DIR, CLEAN_FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# Seaborn Deep Palette
palette = sns.color_palette("deep")
C_BLUE = palette[0]      # #4c72b0 (Deep Blue for C0 and primary contrast)
C_GREEN = palette[2]     # #55a868
C_RED = palette[3]       # #c44e52
C_PURPLE = palette[4]    # #8172b3 (Deep Purple for bootstrap CI)
C_DARK = '#1e293b'       # Slate 800 for clean text
C_GRAY = '#64748b'       # Slate 500 for neutral contrasts
C_GRAY_LIGHT = '#94a3b8' # Slate 400 for subtle seed lines

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
    r'$\mathrm{S_1(35k) - C_0(35k)}$' + '\n(Primary Contrast)',
    r'$\mathrm{S_1(35k) - C_3(35k)}$' + '\n(Detour vs. Reset)',
    r'$\mathrm{C_3(35k) - C_0(35k)}$' + '\n(Procedural Reset)',
    r'$\mathrm{S_1(35k) - C_0(30k)}$' + '\n(Target Matched)'
]
mean_diffs = [-0.4325, -0.0300, -0.4025, +0.2660]
t_ci_low = [-0.9390, -0.7618, -1.1424, -0.4691]
t_ci_high = [+0.0740, +0.7018, +0.3374, +1.0011]
boot_ci_low = [-0.8880, -0.6700, -1.1200, -0.4300]
boot_ci_high = [+0.0200, +0.6700, +0.2400, +0.9200]

# Set up figure at native 6.5 in width (TMLR full-width) and 3.10 in height
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.10), dpi=300,
                               gridspec_kw={'width_ratios': [0.95, 1.25], 'wspace': 0.70})

# =========================================================================
# Panel (a): Paired Profiles (Authentic Seaborn Design)
# =========================================================================
# 1. Individual seed trajectories in neutral muted slate
for i in range(len(c0_35k)):
    ax1.plot([0.0, 1.0], [c0_35k[i], s1_35k[i]], color=C_GRAY_LIGHT, lw=0.9, alpha=0.45, zorder=2)
    ax1.scatter([0.0, 1.0], [c0_35k[i], s1_35k[i]], color=C_GRAY, s=14, alpha=0.65, zorder=3)

# 2. Bold Population Mean Trajectory (Seaborn style: clean circular marker, solid line)
ax1.plot([0.0, 1.0], [mean_c0, mean_s1], color=C_BLUE, lw=2.2, marker='o', markersize=6.8,
         markerfacecolor=C_BLUE, markeredgecolor='white', markeredgewidth=1.2, zorder=5, label='Mean')

# 3. Mean values - clean plain text, NO bounding boxes!
ax1.text(-0.06, mean_c0, f"{mean_c0:.2f}%", ha='right', va='center', fontsize=7.8, color=C_BLUE, fontweight='medium')
ax1.text(1.06, mean_s1, f"{mean_s1:.2f}%\n(-0.43 pp)", ha='left', va='center', fontsize=7.8, color=C_BLUE, fontweight='medium')

ax1.set_xlim(-0.35, 1.35)
ax1.set_ylim(51.5, 59.8)
ax1.set_yticks([52, 54, 56, 58])
ax1.set_yticklabels(['52%', '54%', '56%', '58%'], fontsize=7.8)
ax1.set_xticks([0.0, 1.0])
ax1.set_xticklabels([
    r'$\mathrm{C_0}$ Continuous' + '\n(35k Updates)',
    r'$\mathrm{S_1}$ Auxiliary' + '\n(35k Updates)'
], fontsize=8.0)
ax1.set_ylabel('Terminal CIFAR-100 Acc. (%)', fontsize=8.6, color=C_DARK)
ax1.set_title('(a) Paired Test Accuracy (N = 20)', fontsize=9.2, fontweight='semibold', color=C_DARK, pad=26)

ax1.grid(axis='y', linestyle='--', alpha=0.35, color='#cbd5e1')
sns.despine(ax=ax1, trim=False)

# =========================================================================
# Panel (b): Confirmatory Contrasts (Authentic Seaborn Forest Plot)
# =========================================================================
y_pos = [3, 2, 1, 0]

# 1. Soft Equivalence Region
ax2.axvspan(-0.5, 0.5, facecolor='#2ca02c', alpha=0.07, zorder=0)
ax2.text(0.0, 3.65, "Equivalence Band (±0.5 pp)", ha='center', va='bottom', fontsize=7.2, fontweight='medium', color='#2d7f36')

# 2. Zero reference line - STOPS safely below the header text to eliminate overlap!
ax2.vlines(0.0, ymin=-0.45, ymax=3.48, color='#64748b', linestyle='--', lw=1.0, alpha=0.8, zorder=1)

# 3. Horizontal error bars
for i, y in enumerate(y_pos):
    m = mean_diffs[i]
    t_lo, t_hi = t_ci_low[i], t_ci_high[i]
    b_lo, b_hi = boot_ci_low[i], boot_ci_high[i]
    is_primary = (i == 0)
    col = C_BLUE if is_primary else C_GRAY
    
    # 95% Parametric t-CI (outer bar with clean caps)
    ax2.errorbar(m, y + 0.06, xerr=[[m - t_lo], [t_hi - m]], fmt='none', color=col,
                 capsize=3.5, capthick=1.0, elinewidth=1.4, zorder=3)
    # 95% Bootstrap CI (slightly offset below, purple)
    ax2.errorbar(m, y - 0.06, xerr=[[m - b_lo], [b_hi - m]], fmt='none', color=C_PURPLE,
                 capsize=0, elinewidth=2.2, alpha=0.75, zorder=3)
    # Point estimate
    ax2.scatter(m, y, color=col, s=36, zorder=5, edgecolor='white', linewidth=0.8)
    
    # Value label - clean, plain text without bounding box!
    sign_m = f"{m:+.2f} pp" if m != 0 else "0.00 pp"
    ax2.text(m, y + 0.18, sign_m, ha='center', va='bottom', fontsize=7.5, color=col, fontweight='medium')

ax2.set_yticks(y_pos)
ax2.set_yticklabels(contrasts, fontsize=7.8)
ax2.set_xlim(-1.45, 1.35)
ax2.set_ylim(-0.55, 3.85)
ax2.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax2.set_xticklabels(['-1.0', '-0.5', '0.0', '+0.5', '+1.0'], fontsize=7.8)
ax2.set_xlabel('Paired Difference in Accuracy (pp)', fontsize=8.6, color=C_DARK)
ax2.set_title('(b) Confirmatory Paired Contrasts', fontsize=9.2, fontweight='semibold', color=C_DARK, pad=26)

ax2.grid(axis='y', linestyle='--', alpha=0.35, color='#cbd5e1')
sns.despine(ax=ax2, trim=False)

# 4. Clean borderless horizontal legend placed outside above the plot frame
line_t = plt.Line2D([0], [0], color=C_BLUE, lw=1.4, label="95% Paired t-CI")
line_b = plt.Line2D([0], [0], color=C_PURPLE, lw=2.2, label="95% Bootstrap CI")
ax2.legend(handles=[line_t, line_b], loc='lower center', bbox_to_anchor=(0.5, 1.02),
           ncol=2, frameon=False, fontsize=7.4)

plt.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.16)

pdf_out = os.path.join(FIGURES_DIR, "figure_paired_contrasts_forest_plot.pdf")
png_out = os.path.join(FIGURES_DIR, "figure_paired_contrasts_forest_plot.png")
fig.savefig(pdf_out, bbox_inches='tight', pad_inches=0.04)
fig.savefig(png_out, bbox_inches='tight', pad_inches=0.04, dpi=300)

for d in [CLEAN_FIGURES_DIR]:
    fig.savefig(os.path.join(d, "figure_paired_contrasts_forest_plot.pdf"), bbox_inches='tight', pad_inches=0.04)
    fig.savefig(os.path.join(d, "figure_paired_contrasts_forest_plot.png"), bbox_inches='tight', pad_inches=0.04, dpi=300)

plt.close(fig)
print(f"Figure 2 successfully generated: {pdf_out}")

# =========================================================================
# SUPPLEMENTARY FIGURE S1: Permutation Null Distribution
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

fig_supp, ax_s = plt.subplots(figsize=(4.5, 2.5), dpi=300)
n, bins, patches_hist = ax_s.hist(perm_means, bins=35, edgecolor=C_DARK, linewidth=0.5, facecolor='#EBF5FB', zorder=3)
for patch, bin_left, bin_right in zip(patches_hist, bins[:-1], bins[1:]):
    if bin_right <= -np.abs(obs_stat) or bin_left >= np.abs(obs_stat):
        patch.set_facecolor('#FADBD8')
        patch.set_edgecolor(C_RED)

line_top = ax_s.get_ylim()[1]
ax_s.axvline(obs_stat, color=C_RED, lw=1.6, linestyle='-', zorder=5, label=f"Observed: {obs_stat:.2f} pp")
ax_s.axvline(-obs_stat, color=C_RED, lw=1.2, linestyle=':', zorder=5)
ax_s.set_xlabel("Mean Paired Difference (pp)", fontsize=8.6, color=C_DARK)
ax_s.set_ylabel("Permutation Count", fontsize=8.6, color=C_DARK)
ax_s.set_title("Exact Permutation Null Distribution (N = 1,048,576)", fontsize=9.2, fontweight='semibold', color=C_DARK)
ax_s.text(obs_stat, line_top * 0.85, "p = 0.091\n(exact)", ha='center', fontsize=7.5, color=C_RED, fontweight='medium')
ax_s.legend(loc='upper right', frameon=False, fontsize=7.5)
ax_s.grid(axis='y', linestyle='--', alpha=0.35, color='#cbd5e1')
sns.despine(ax=ax_s, trim=False)
plt.subplots_adjust(left=0.14, right=0.96, top=0.88, bottom=0.18)

supp_pdf = os.path.join(FIGURES_DIR, "figure_cifar_permutation_null.pdf")
fig_supp.savefig(supp_pdf, bbox_inches='tight', pad_inches=0.03)
for d in [CLEAN_FIGURES_DIR]:
    fig_supp.savefig(os.path.join(d, "figure_cifar_permutation_null.pdf"), bbox_inches='tight', pad_inches=0.03)
plt.close(fig_supp)
print(f"Supplementary Figure S1 generated: {supp_pdf}")
