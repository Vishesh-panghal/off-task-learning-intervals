#!/usr/bin/env python3
"""
Generate publication-grade Figure 3: Factorial Decomposition of Procedural Costs.
Designed at native 6.5-inch width (6.5 x 3.10 in) following top-tier journal aesthetics:
- Panel (a): Factorial Interaction Plot.
  * Mean trajectories with 95% t-confidence intervals (faint background seed lines removed for clarity).
  * Zoomed y-axis (53.2--58.6%) to highlight slope parallelism.
  * Numerical annotations positioned cleanly with white protective backing.
  * Clean, bordered legend situated in clear lower-left whitespace.
  * Simple, elegant title: "(a) Factorial Interaction".
- Panel (b): Marginal Contrasts Forest Plot (95% CI).
  * 3 spacious contrast rows guided by subtle horizontal gridlines.
  * Consistent numeric annotations centered cleanly above markers.
  * Green shaded Equivalence Region (±0.5 pp) with compact header text.
  * Self-explanatory vertical dashed Null line (redundant null legend removed).

Typography:
- Panel titles: 9.8 pt bold
- Axis labels: 9.0 pt bold
- Tick labels: 8.0--8.2 pt bold
- Value tags & legend: 7.6--7.8 pt bold
"""

import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

C_BLUE = '#1F77B4'
C_GREEN = '#2CA02C'
C_RED = '#D62728'
C_PURPLE = '#7D3C98'
C_DARK = '#0F172A'
C_MUTED = '#475569'

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

effects = [
    (r"Head Substitution ($\Delta_{\mathrm{head}}$)" + "\n" + r"($F = 24.95, p = 0.004$)", diff_head, C_RED),
    (r"Momentum Reset ($\Delta_{\mathrm{mom}}$)" + "\n" + r"($F = 1.47, p = 0.279$)", diff_mom, C_GREEN),
    (r"Factorial Interaction ($\Delta_{\mathrm{int}}$)" + "\n" + r"($F = 0.002, p = 0.967$)", diff_int, C_PURPLE)
]

# Set up figure with native 6.5 in width and 3.10 in height
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.10), dpi=300,
                               gridspec_kw={'width_ratios': [1.0, 1.25], 'wspace': 0.75})

# =========================================================================
# Panel (a): Factorial Interaction
# =========================================================================
x = [0.0, 1.0]
y_cont = [m_f00, m_f10]   # [56.32, 54.21]
y_reset = [m_f01, m_f11]  # [56.88, 54.79]

err_cont = [se_f00 * t_crit, se_f10 * t_crit]
err_reset = [se_f01 * t_crit, se_f11 * t_crit]

# 1. Population mean trajectories with 95% CIs (faint background seed lines removed for maximum clarity)
ax1.errorbar(x, y_cont, yerr=err_cont, fmt='o-', color=C_BLUE, lw=3.2, markersize=8.0,
             capsize=4.5, capthick=1.5, label=r"Continuous Mom ($\mathbf{v}$ intact)", zorder=5)
ax1.errorbar(x, y_reset, yerr=err_reset, fmt='s--', color=C_GREEN, lw=3.2, markersize=8.0,
             capsize=4.5, capthick=1.5, label=r"Reset Mom ($\mathbf{v} \leftarrow \mathbf{0}$)", zorder=5)

# 2. Add numerical annotations with white bounding boxes to prevent overlap
ax1.text(-0.08, y_cont[0] - 0.12, f"{y_cont[0]:.2f}%", ha='right', va='center', fontsize=8.2, fontweight='bold', color=C_BLUE,
         bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.90, zorder=6))
ax1.text(-0.08, y_reset[0] + 0.12, f"{y_reset[0]:.2f}%", ha='right', va='center', fontsize=8.2, fontweight='bold', color=C_GREEN,
         bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.90, zorder=6))
ax1.text(1.08, y_cont[1] - 0.12, f"{y_cont[1]:.2f}%", ha='left', va='center', fontsize=8.2, fontweight='bold', color=C_BLUE,
         bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.90, zorder=6))
ax1.text(1.08, y_reset[1] + 0.12, f"{y_reset[1]:.2f}%", ha='left', va='center', fontsize=8.2, fontweight='bold', color=C_GREEN,
         bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.90, zorder=6))

# 3. Formatting
ax1.set_xticks(x)
ax1.set_xticklabels([r"Retained Head ($h_A$)" + "\n(Continuous)", r"Temp Head ($h_{\mathrm{temp}}$)" + "\n(Substituted)"], fontsize=8.2, fontweight='bold')
ax1.set_ylabel("CIFAR-100 Test Accuracy (%)", fontsize=9.0, fontweight='bold', color=C_DARK)
ax1.set_title("(a) Factorial Interaction", fontsize=9.8, fontweight='bold', color=C_DARK, pad=12)

# Zoom in to make slopes clearly readable (53.2 to 58.6)
ax1.set_ylim(53.2, 58.6)
ax1.set_yticks([54, 55, 56, 57, 58])
ax1.set_yticklabels(["54%", "55%", "56%", "57%", "58%"], fontsize=8.2, fontweight='bold')
ax1.set_xlim(-0.30, 1.30)

# Legend placed in clean lower-left whitespace
ax1.legend(loc='lower left', frameon=True, framealpha=0.95, fontsize=7.6, edgecolor='#CBD5E1')
ax1.grid(axis='y', linestyle=':', alpha=0.3, color='#CBD5E1')

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# =========================================================================
# Panel (b): Marginal Contrasts Forest Plot
# =========================================================================
y_pos = [2, 1, 0] # Top to bottom
labels = [
    r"Head Substitution ($\Delta_{\mathrm{head}}$)" + "\n" + r"($F = 24.95, p = 0.004$)",
    r"Momentum Reset ($\Delta_{\mathrm{mom}}$)" + "\n" + r"($F = 1.47, p = 0.279$)",
    r"Factorial Interaction ($\Delta_{\mathrm{int}}$)" + "\n" + r"($F = 0.002, p = 0.967$)"
]
effects_vals = [np.mean(diff_head), np.mean(diff_mom), np.mean(diff_int)]
errors = [stats.sem(diff_head) * t_crit, stats.sem(diff_mom) * t_crit, stats.sem(diff_int) * t_crit]
colors = [C_RED, C_GREEN, C_PURPLE]

# Equivalence Region
ax2.axvspan(-0.5, 0.5, facecolor='#EAFAF1', alpha=0.50, zorder=0)
ax2.text(0.0, 2.52, "Equivalence Region (±0.5 pp)", color='#27AE60', ha='center', va='bottom', fontsize=7.2, fontweight='bold')

# Null line (self-explanatory thick vertical dashed line)
ax2.axvline(0.0, color=C_DARK, linestyle='--', lw=1.2, alpha=0.7, zorder=1)

# Plot points and CIs
for i in range(3):
    m = effects_vals[i]
    err = errors[i]
    col = colors[i]
    y = y_pos[i]
    
    # 95% Confidence Interval bar with caps
    ax2.errorbar(m, y, xerr=err, fmt='o', color=col, capsize=5.0, capthick=1.5,
                 markersize=8.0, linewidth=2.4, zorder=5, markeredgecolor='white', markeredgewidth=0.8)
    
    # Clean annotations strictly above each point with white protective box
    sign_str = f"{m:+.2f} pp"
    ax2.text(m, y + 0.20, sign_str, ha='center', va='bottom', color=col, fontsize=7.8, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.90, zorder=6))

# Formatting
ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels, fontsize=8.0, fontweight='bold')
ax2.set_xlabel("Marginal Paired Effect (pp)", fontsize=9.0, fontweight='bold', color=C_DARK)
ax2.set_title("(b) Marginal Contrasts (95% CI)", fontsize=9.8, fontweight='bold', color=C_DARK, pad=12)
ax2.set_xlim(-3.6, 2.4)
ax2.set_ylim(-0.5, 2.8)
ax2.set_xticks([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
ax2.set_xticklabels(["-3.0", "-2.0", "-1.0", "0.0", "+1.0", "+2.0"], fontsize=8.2, fontweight='bold')

# Light horizontal gridlines behind the rows
ax2.grid(axis='y', linestyle=':', alpha=0.3, color='#CBD5E1')

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.16)

pdf_out = os.path.join(FIGURES_DIR, "figure_factorial_decomposition.pdf")
png_out = os.path.join(FIGURES_DIR, "figure_factorial_decomposition.png")
fig.savefig(pdf_out, bbox_inches='tight', pad_inches=0.06)
fig.savefig(png_out, bbox_inches='tight', pad_inches=0.06, dpi=300)

for d in [CLEAN_FIGURES_DIR]:
    fig.savefig(os.path.join(d, "figure_factorial_decomposition.pdf"), bbox_inches='tight', pad_inches=0.06)
    fig.savefig(os.path.join(d, "figure_factorial_decomposition.png"), bbox_inches='tight', pad_inches=0.06, dpi=300)

plt.close(fig)
print(f"Figure 3 successfully generated: {pdf_out}")
