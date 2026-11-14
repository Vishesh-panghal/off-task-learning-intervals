#!/usr/bin/env python3
"""
Generate publication-grade Figure 2: Confirmatory Paired Contrasts on CIFAR-100.
Redesigned with native 6.5-inch width and maximum vertical clarity (figsize 6.5 x 3.45 in):
- Panel (a): Seed-Level Paired Profiles (N = 20 paired seeds).
  * High-clarity individual trajectories in neutral slate gray with collision-free beeswarm dodge.
  * Prominent Grand Mean trajectory in Primary Blue with clean annotations inside x-axis labels.
  * Zero overlap: no line intersects the summary badge; no markers hide behind the Mean diamond.
- Panel (b): Confirmatory Paired Contrasts with 95% t-intervals and bootstrap intervals.
  * Highly spacious 4-contrast forest plot with generous inter-row vertical clearance.
  * Equivalence region label positioned at bottom (-0.48), leaving upper-right legend 100% clear.
- Panel (c) (Permutation Null Distribution) exported as supplementary figure.

Typography:
- Panel titles: 9.8 pt bold
- Axis labels: 9.2 pt bold
- Tick labels: 8.0--8.4 pt bold
- Legend & point tags: 7.4--7.8 pt bold
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
if os.path.basename(BASE_DIR) == "clean_repo":
    FIGURES_DIR = os.path.join(BASE_DIR, "figures")
    CLEAN_FIGURES_DIR = FIGURES_DIR
else:
    FIGURES_DIR = os.path.join(BASE_DIR, "figures")
    CLEAN_FIGURES_DIR = os.path.join(BASE_DIR, "clean_repo", "figures")
for d in [FIGURES_DIR, CLEAN_FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

C_BLUE = '#1F77B4'      # Paper primary Blue (C0 Baseline and Primary Contrast)
C_DARK = '#0F172A'      # Slate 900
C_GRAY = '#64748B'      # Slate 500 for seed markers
C_GRAY_LINE = '#94A3B8' # Slate 400 for neutral seed trajectories
C_NULL = '#D62728'      # Null reference line
C_RED = '#E74C3C'       # Critical rejection region in Supplementary Figure S1
C_PURPLE = '#7D3C98'    # Bootstrap CI
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
# MAIN MANUSCRIPT FIGURE 2: 2 Panels with maximum vertical clarity (3.45 in)
# =========================================================================
# Collision-free 1D beeswarm algorithm
# sx: points per data unit in x (~ 85 pt)
# sy: points per data unit in y (~ 20 pt / %)
def compute_collision_free_xs(vals, center_x, mean_val=None, min_dist_pt=5.2, step_x=0.028):
    n = len(vals)
    xs = np.full(n, center_x)
    order = np.argsort(vals)
    sx = 85.0
    sy = 20.0
    
    # Pre-place the mean marker so data points do not collide with or hide behind it
    placed = []
    if mean_val is not None:
        placed.append((center_x * sx, mean_val * sy, 7.5)) # 7.5 pt radius clearance for mean diamond
        
    for idx in order:
        y_val = vals[idx]
        y_pt = y_val * sy
        best_x = center_x
        for k in range(25):
            candidates = [0.0] if k == 0 else [k * step_x, -k * step_x]
            found = False
            for cand_offset in candidates:
                cand_x = center_x + cand_offset
                cand_x_pt = cand_x * sx
                collision = False
                for px, py, req_dist in placed:
                    d = np.hypot(cand_x_pt - px, y_pt - py)
                    if d < req_dist:
                        collision = True
                        break
                if not collision:
                    best_x = cand_x
                    placed.append((cand_x_pt, y_pt, min_dist_pt))
                    found = True
                    break
            if found:
                break
        xs[idx] = best_x
    return xs

xs_c0 = compute_collision_free_xs(c0_35k, 0.0, mean_val=mean_c0, min_dist_pt=5.2, step_x=0.028)
xs_s1 = compute_collision_free_xs(s1_35k, 1.0, mean_val=mean_s1, min_dist_pt=5.2, step_x=0.028)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.45), dpi=300,
                               gridspec_kw={'width_ratios': [0.95, 1.25], 'wspace': 0.85})

# --- Panel (a): Seed-Level Paired Slopes ---
for i in range(len(seeds)):
    ax1.plot([xs_c0[i], xs_s1[i]], [c0_35k[i], s1_35k[i]], color=C_GRAY_LINE, lw=1.15, alpha=0.45, zorder=2)
    ax1.scatter(xs_c0[i], c0_35k[i], facecolor='white', edgecolor=C_GRAY, linewidth=1.1, s=18, alpha=0.95, zorder=3)
    ax1.scatter(xs_s1[i], s1_35k[i], facecolor='white', edgecolor=C_GRAY, linewidth=1.1, s=18, alpha=0.95, zorder=3)

# Bold Primary Blue Mean Trajectory (matches paper palette)
ax1.plot([0.0, 1.0], [mean_c0, mean_s1], color=C_BLUE, lw=3.6, marker='D', markersize=8.0,
         markerfacecolor=C_BLUE, markeredgecolor='white', markeredgewidth=1.8, zorder=5)

ax1.set_xlim(-0.35, 1.35)
ax1.set_ylim(51.5, 61.2)
ax1.set_yticks([52, 54, 56, 58, 60])
ax1.set_yticklabels(["52%", "54%", "56%", "58%", "60%"], fontsize=8.2, fontweight='bold')
ax1.set_xticks([0.0, 1.0])
ax1.set_xticklabels([
    r"$\mathbf{C_0}$ Continuous" + f"\n(Mean: {mean_c0:.2f}%)",
    r"$\mathbf{S_1}$ Auxiliary" + f"\n(Mean: {mean_s1:.2f}%)"
], fontsize=8.4, fontweight='bold')
ax1.set_ylabel("Terminal Test Accuracy (%)", fontsize=9.2, fontweight='bold', color=C_DARK, labelpad=4.0)
ax1.set_title(r"(a) Paired Test Accuracy Profiles ($N = 20$)", fontsize=9.8, fontweight='bold', color=C_DARK, pad=10)

ax1.grid(True, linestyle=':', alpha=0.25, color='#CBD5E1', axis='y')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Clean statistical annotation badge at top-center in empty region (Section 4.4 paired t-test)
# Placed at y = 60.95 with va='top', bottom of box sits at ~59.7% (well above highest trajectory peak at 59.07%)
ax1.text(0.5, 60.95, "13/20 seeds favor Continuous\n" + rf"Mean $\Delta = -0.43$ pp ($p = 0.090$, Paired $t$)",
         ha='center', va='top', fontsize=7.6, fontweight='bold', color=C_DARK,
         bbox=dict(boxstyle='round,pad=0.28', facecolor='white', edgecolor='#CBD5E1', lw=0.8, alpha=0.95, zorder=6))

# --- Panel (b): Confirmatory Contrasts Forest Plot ---
y_pos = np.arange(len(contrasts))[::-1]
ax2.axvline(0.0, color=C_NULL, linestyle='--', lw=1.2, zorder=1)
ax2.axvspan(-0.5, 0.5, facecolor='#EAFAF1', alpha=0.50, zorder=0)

# Placed at bottom of equivalence zone so it NEVER collides with upper-right legend!
ax2.text(0.0, -0.48, "Equivalence Region (±0.5 pp)", ha='center', va='center', fontsize=7.2, fontweight='bold', color='#27AE60')

for i, y in enumerate(y_pos):
    m = mean_diffs[i]
    t_lo, t_hi = t_ci_low[i], t_ci_high[i]
    b_lo, b_hi = boot_ci_low[i], boot_ci_high[i]
    is_primary = (i == 0)
    col = C_BLUE if is_primary else C_MUTED
    pt_col = C_BLUE if is_primary else C_MUTED
    
    # Parametric t CI (outer line with end caps)
    ax2.plot([t_lo, t_hi], [y + 0.08, y + 0.08], color=col, lw=2.0, zorder=3)
    ax2.plot([t_lo, t_lo], [y + 0.02, y + 0.14], color=col, lw=2.0, zorder=3)
    ax2.plot([t_hi, t_hi], [y + 0.02, y + 0.14], color=col, lw=2.0, zorder=3)
    
    # Bootstrap CI (inner thicker bar)
    ax2.plot([b_lo, b_hi], [y - 0.08, y - 0.08], color=C_PURPLE, lw=3.2, alpha=0.85, zorder=3)
    
    # Point estimate
    ax2.scatter(m, y, color=pt_col, s=50, zorder=5, edgecolor='white', linewidth=0.6)
    
    # Direct numeric tag placed cleanly above the point
    sign_m = f"{m:+.2f} pp" if m != 0 else "0.00 pp"
    if i == 1: # -0.03 pp: align right of point to preserve ample clear space before the legend
        ax2.text(m - 0.10, y + 0.24, sign_m, ha='right', va='bottom', fontsize=7.8, fontweight='bold', color=C_DARK,
                 bbox=dict(boxstyle="round,pad=0.15", facecolor='white', edgecolor='#E2E8F0', lw=0.8, alpha=0.95, zorder=6))
    else:
        ax2.text(m, y + 0.24, sign_m, ha='center', va='bottom', fontsize=7.8, fontweight='bold', color=C_DARK,
                 bbox=dict(boxstyle="round,pad=0.15", facecolor='white', edgecolor='#E2E8F0', lw=0.8, alpha=0.95, zorder=6))

ax2.set_yticks(y_pos)
ax2.set_yticklabels(contrasts, fontsize=8.0, fontweight='bold')
ax2.set_xlim(-1.45, 1.35)
ax2.set_ylim(-0.70, 3.95)
ax2.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax2.set_xticklabels(["-1.0", "-0.5", "0.0", "+0.5", "+1.0"], fontsize=8.2, fontweight='bold')
ax2.set_xlabel("Paired Difference in Test Accuracy (pp)", fontsize=9.2, fontweight='bold', color=C_DARK)
ax2.set_title("(b) Confirmatory Paired Contrasts", fontsize=9.8, fontweight='bold', color=C_DARK, pad=8)
ax2.grid(True, linestyle=':', alpha=0.25, color='#CBD5E1')

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Legend placed in upper right with generous clearance
line_t = plt.Line2D([0], [0], color=C_BLUE, lw=2.0, label="95% Paired t-CI")
line_b = plt.Line2D([0], [0], color=C_PURPLE, lw=3.2, label="95% Bootstrap CI")
line_z = plt.Line2D([0], [0], color=C_NULL, linestyle='--', lw=1.2, label="Null (0 pp)")
ax2.legend(handles=[line_t, line_b, line_z], loc='upper right', frameon=True, framealpha=0.95, fontsize=7.4, edgecolor='#CBD5E1')

plt.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.14)

pdf_out = os.path.join(FIGURES_DIR, "figure_paired_contrasts_forest_plot.pdf")
png_out = os.path.join(FIGURES_DIR, "figure_paired_contrasts_forest_plot.png")
fig.savefig(pdf_out, bbox_inches='tight', pad_inches=0.07)
fig.savefig(png_out, bbox_inches='tight', pad_inches=0.07, dpi=300)

for d in [CLEAN_FIGURES_DIR]:
    fig.savefig(os.path.join(d, "figure_paired_contrasts_forest_plot.pdf"), bbox_inches='tight', pad_inches=0.07)
    fig.savefig(os.path.join(d, "figure_paired_contrasts_forest_plot.png"), bbox_inches='tight', pad_inches=0.07, dpi=300)

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
n, bins, patches_hist = ax_s.hist(perm_means, bins=35, edgecolor=C_DARK, linewidth=0.6, facecolor='#EBF5FB', zorder=3)
for patch, bin_left, bin_right in zip(patches_hist, bins[:-1], bins[1:]):
    if bin_right <= -np.abs(obs_stat) or bin_left >= np.abs(obs_stat):
        patch.set_facecolor('#FADBD8')
        patch.set_edgecolor(C_RED)

line_top = ax_s.get_ylim()[1]
ax_s.axvline(obs_stat, color=C_RED, lw=1.8, linestyle='-', zorder=5, label=f"Observed: {obs_stat:.2f} pp")
ax_s.axvline(-obs_stat, color=C_RED, lw=1.4, linestyle=':', zorder=5)
ax_s.set_xlabel("Mean Paired Difference (pp)", fontsize=8.8, fontweight='bold')
ax_s.set_ylabel("Permutation Count", fontsize=8.8, fontweight='bold')
ax_s.set_title("Exact Permutation Null Distribution (N = 1,048,576)", fontsize=9.5, fontweight='bold')
ax_s.text(obs_stat, line_top * 0.85, f"p = 0.091\n(exact)", ha='center', fontsize=7.8, fontweight='bold', color=C_RED,
          bbox=dict(boxstyle="round,pad=0.2", facecolor='#FDEDEC', edgecolor=C_RED, lw=0.8))
ax_s.legend(loc='upper right', fontsize=7.5)
ax_s.grid(True, linestyle=':', alpha=0.4)
plt.subplots_adjust(left=0.14, right=0.96, top=0.88, bottom=0.18)

supp_pdf = os.path.join(FIGURES_DIR, "figure_cifar_permutation_null.pdf")
fig_supp.savefig(supp_pdf, bbox_inches='tight', pad_inches=0.02)
for d in [CLEAN_FIGURES_DIR]:
    fig_supp.savefig(os.path.join(d, "figure_cifar_permutation_null.pdf"), bbox_inches='tight', pad_inches=0.02)
plt.close(fig_supp)
print(f"Supplementary Figure S1 generated: {supp_pdf}")
