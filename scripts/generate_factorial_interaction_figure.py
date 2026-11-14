#!/usr/bin/env python3
"""
Generate publication-grade Figure 3: Factorial Decomposition of Procedural Costs.
Authored in authentic Seaborn statistical style (sns.set_theme(style="ticks")):
- Panel (a): Factorial Interaction Plot.
  * Subtle individual seed trajectories (N = 6 paired seeds) showing raw empirical data.
  * Clean population mean trajectories with 95% t-confidence intervals (Seaborn Deep palette).
  * Natural, unboxed numeric labels without artificial rounded bounding boxes.
  * Classic Seaborn legend situated in natural upper-right whitespace (zero data collisions).
- Panel (b): Marginal Contrasts Forest Plot (95% CI).
  * 3 spacious contrast rows with clean horizontal error bars and point estimates.
  * Subtle, desaturated equivalence band (±0.5 pp) and zero reference line.
  * Clean numeric annotations directly above markers in matching palette colors.
  * L-shaped despined frames with outward ticks and soft dashed horizontal grid.

Typography & Design:
- Genuine Seaborn Deep palette: #4c72b0 (blue), #55a868 (green), #c44e52 (coral), #8172b3 (purple).
- Standard academic hierarchy: semibold titles, medium axis labels, normal weight ticks/values.
"""

import os
import numpy as np
from scipy import stats
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
C_BLUE = palette[0]    # #4c72b0
C_GREEN = palette[2]   # #55a868
C_RED = palette[3]     # #c44e52
C_PURPLE = palette[4]  # #8172b3
C_DARK = '#1e293b'     # Slate 800 for clean text

# Data from Table 3 (N = 6 paired seeds)
f00 = np.array([55.38, 57.19, 57.98, 55.85, 56.52, 55.01])  # Retained Head, Continuous Mom
f01 = np.array([57.77, 56.95, 57.59, 55.79, 57.32, 55.86])  # Retained Head, Zeroed Mom
f10 = np.array([55.47, 54.01, 55.84, 52.73, 53.87, 53.34])  # Temp Head, Continuous Mom
f11 = np.array([56.71, 54.13, 53.67, 53.40, 55.97, 54.86])  # Temp Head, Zeroed Mom

n = len(f00)
df = n - 1
t_crit = stats.t.ppf(0.975, df)

m_f00, se_f00 = np.mean(f00), stats.sem(f00)
m_f01, se_f01 = np.mean(f01), stats.sem(f01)
m_f10, se_f10 = np.mean(f10), stats.sem(f10)
m_f11, se_f11 = np.mean(f11), stats.sem(f11)

# Factorial marginal effects per seed
diff_head = 0.5 * ((f10 - f00) + (f11 - f01))
diff_mom = 0.5 * ((f01 - f00) + (f11 - f10))
diff_int = (f11 - f10) - (f01 - f00)

# Set up figure at native 6.5 in width (TMLR full-width) and 3.0 in height
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0), dpi=300,
                               gridspec_kw={'width_ratios': [1.05, 1.25], 'wspace': 0.65})

# =========================================================================
# Panel (a): Factorial Interaction (Authentic Seaborn Design)
# =========================================================================
x_vals = [0, 1]

# 1. Subtle raw paired seed trajectories (real empirical data, transparent)
for i in range(n):
    ax1.plot(x_vals, [f00[i], f10[i]], color=C_BLUE, alpha=0.18, lw=0.9, zorder=2)
    ax1.plot(x_vals, [f01[i], f11[i]], color=C_GREEN, alpha=0.18, lw=0.9, linestyle='--', zorder=2)

# 2. Population Means with 95% t-CI (Seaborn pointplot style)
y_cont = [m_f00, m_f10]
y_reset = [m_f01, m_f11]
err_cont = [se_f00 * t_crit, se_f10 * t_crit]
err_reset = [se_f01 * t_crit, se_f11 * t_crit]

ax1.errorbar(x_vals, y_cont, yerr=err_cont, fmt='o-', color=C_BLUE, lw=2.0, markersize=6.5,
             capsize=3.0, capthick=1.0, elinewidth=1.2, label=r'Continuous ($\mathbf{v}$ intact)', zorder=4)
ax1.errorbar(x_vals, y_reset, yerr=err_reset, fmt='s--', color=C_GREEN, lw=2.0, markersize=6.5,
             capsize=3.0, capthick=1.0, elinewidth=1.2, label=r'Reset ($\mathbf{v} \leftarrow \mathbf{0}$)', zorder=4)

# 3. Clean numeric annotations (human academic typography, zero bounding boxes)
ax1.text(-0.06, y_cont[0], f"{y_cont[0]:.2f}%", ha='right', va='center', fontsize=7.8, color=C_BLUE)
ax1.text(-0.06, y_reset[0], f"{y_reset[0]:.2f}%", ha='right', va='center', fontsize=7.8, color=C_GREEN)
ax1.text(1.06, y_cont[1], f"{y_cont[1]:.2f}%", ha='left', va='center', fontsize=7.8, color=C_BLUE)
ax1.text(1.06, y_reset[1], f"{y_reset[1]:.2f}%", ha='left', va='center', fontsize=7.8, color=C_GREEN)

ax1.set_xticks(x_vals)
ax1.set_xticklabels([r'Retained Head ($h_A$)' + '\n(Continuous)', r'Temp Head ($h_{\mathrm{temp}}$)' + '\n(Substituted)'], fontsize=8.0)
ax1.set_ylabel('CIFAR-100 Test Accuracy (%)', fontsize=8.6, color=C_DARK)
ax1.set_title('(a) Factorial Interaction', fontsize=9.2, fontweight='semibold', color=C_DARK, pad=10)

ax1.set_ylim(52.2, 59.2)
ax1.set_yticks([53, 54, 55, 56, 57, 58])
ax1.set_yticklabels(['53%', '54%', '55%', '56%', '57%', '58%'], fontsize=7.8)
ax1.set_xlim(-0.32, 1.32)

# 4. Classic Seaborn legend: upper right inside, borderless, crisp
leg1 = ax1.legend(loc='upper right', frameon=False, fontsize=7.5, title="Momentum Buffer")
leg1.get_title().set_fontsize(7.8)
leg1.get_title().set_fontweight('semibold')

ax1.grid(axis='y', linestyle='--', alpha=0.35, color='#cbd5e1')
sns.despine(ax=ax1, trim=False)

# =========================================================================
# Panel (b): Marginal Contrasts Forest Plot (Authentic Seaborn Design)
# =========================================================================
y_pos = [2, 1, 0]
labels = [
    r'Head Substitution ($\Delta_{\mathrm{head}}$)' + '\n' + r'($F = 24.95, p = 0.004$)',
    r'Momentum Reset ($\Delta_{\mathrm{mom}}$)' + '\n' + r'($F = 1.47, p = 0.279$)',
    r'Interaction ($\Delta_{\mathrm{int}}$)' + '\n' + r'($F = 0.002, p = 0.967$)'
]
effects_vals = [np.mean(diff_head), np.mean(diff_mom), np.mean(diff_int)]
errors = [stats.sem(diff_head) * t_crit, stats.sem(diff_mom) * t_crit, stats.sem(diff_int) * t_crit]
row_colors = [C_RED, C_GREEN, C_PURPLE]

# Soft Equivalence Region
ax2.axvspan(-0.5, 0.5, facecolor='#2ca02c', alpha=0.07, zorder=0)
ax2.text(0.0, 2.50, "Equivalence Band (±0.5 pp)", color='#2d7f36', ha='center', va='bottom', fontsize=7.2, fontweight='medium')

# Zero reference line
ax2.axvline(0.0, color='#64748b', linestyle='--', lw=1.0, alpha=0.8, zorder=1)

# Horizontal error bars
for i in range(3):
    m = effects_vals[i]
    err = errors[i]
    col = row_colors[i]
    y = y_pos[i]
    
    ax2.errorbar(m, y, xerr=err, fmt='o', color=col, capsize=3.5, capthick=1.0,
                 markersize=6.5, lw=1.8, elinewidth=1.2, zorder=4)
    # Natural value label strictly above point, plain text
    ax2.text(m, y + 0.18, f"{m:+.2f} pp", ha='center', va='bottom', color=col, fontsize=7.5, fontweight='medium')

ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels, fontsize=7.8)
ax2.set_xlabel('Marginal Effect (pp, 95% CI)', fontsize=8.6, color=C_DARK)
ax2.set_title('(b) Marginal Contrasts', fontsize=9.2, fontweight='semibold', color=C_DARK, pad=10)
ax2.set_xlim(-3.4, 2.2)
ax2.set_ylim(-0.5, 2.8)
ax2.set_xticks([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
ax2.set_xticklabels(['-3.0', '-2.0', '-1.0', '0.0', '+1.0', '+2.0'], fontsize=7.8)

ax2.grid(axis='y', linestyle='--', alpha=0.35, color='#cbd5e1')
sns.despine(ax=ax2, trim=False)

plt.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.18)

pdf_out = os.path.join(FIGURES_DIR, "figure_factorial_decomposition.pdf")
png_out = os.path.join(FIGURES_DIR, "figure_factorial_decomposition.png")
fig.savefig(pdf_out, bbox_inches='tight', pad_inches=0.04)
fig.savefig(png_out, dpi=300, bbox_inches='tight', pad_inches=0.04)

for d in [CLEAN_FIGURES_DIR]:
    fig.savefig(os.path.join(d, "figure_factorial_decomposition.pdf"), bbox_inches='tight', pad_inches=0.04)
    fig.savefig(os.path.join(d, "figure_factorial_decomposition.png"), dpi=300, bbox_inches='tight', pad_inches=0.04)

plt.close(fig)
print(f"Figure 3 successfully generated: {pdf_out}")
