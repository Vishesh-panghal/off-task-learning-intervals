#!/usr/bin/env python3
"""
Generate publication-grade Figure 3: Factorial Decomposition of Procedural Costs.
Designed at native 6.5-inch width:
- Panel (a): 2x2 Factorial Interaction Plot (Head Policy vs Momentum Policy).
- Panel (b): Marginal Main Effects Forest Plot (Head Substitution, Momentum Reset, Interaction).

Typography:
- Panel titles: 9.5 pt bold
- Axis labels: 8.8 pt bold
- Tick labels: 8.0 pt bold
- Value tags & legend: 7.8 pt
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 1.55), dpi=300,
                               gridspec_kw={'width_ratios': [1.05, 1.25], 'wspace': 0.32})

# =========================================================================
# Panel (a): 2x2 Factorial Interaction Plot
# =========================================================================
x_vals = [0.0, 1.0]
x_labels = [r"Retained Head ($h_A$)" + "\n(Continuous)", r"Temp Head ($h_{\mathrm{temp}}$)" + "\n(Substituted)"]

# Individual seeds (subtle)
for i in range(n):
    ax1.plot(x_vals, [f00[i], f10[i]], color=C_BLUE, alpha=0.25, lw=0.9)
    ax1.plot(x_vals, [f01[i], f11[i]], color=C_GREEN, alpha=0.25, lw=0.9, linestyle='--')

# Population means
ax1.errorbar(x_vals, [m_f00, m_f10], yerr=[se_f00 * t_crit, se_f10 * t_crit],
             fmt='o-', color=C_BLUE, lw=2.0, markersize=5.5, capsize=3.5, capthick=1.2,
             label=r"Continuous Mom ($\mathbf{v}$ intact)", zorder=5)
ax1.errorbar(x_vals, [m_f01, m_f11], yerr=[se_f01 * t_crit, se_f11 * t_crit],
             fmt='s--', color=C_GREEN, lw=2.0, markersize=5.5, capsize=3.5, capthick=1.2,
             label=r"Reset Mom ($\mathbf{v} \leftarrow \mathbf{0}$)", zorder=5)

# Value annotations
ax1.text(0.0 - 0.04, m_f00, f"{m_f00:.2f}%", ha='right', va='center', fontsize=7.5, fontweight='bold', color=C_BLUE)
ax1.text(0.0 + 0.04, m_f01 + 0.15, f"{m_f01:.2f}%", ha='left', va='bottom', fontsize=7.5, fontweight='bold', color=C_GREEN)
ax1.text(1.0 - 0.04, m_f10, f"{m_f10:.2f}%", ha='right', va='center', fontsize=7.5, fontweight='bold', color=C_BLUE)
ax1.text(1.0 + 0.04, m_f11 + 0.15, f"{m_f11:.2f}%", ha='left', va='bottom', fontsize=7.5, fontweight='bold', color=C_GREEN)

ax1.set_xlim(-0.25, 1.25)
ax1.set_ylim(52.2, 58.8)
ax1.set_xticks(x_vals)
ax1.set_xticklabels(x_labels, fontsize=8.0, fontweight='bold')
ax1.set_ylabel("CIFAR-100 Test Accuracy (%)", fontsize=8.8, fontweight='bold', color=C_DARK)
ax1.set_title("(a) Factorial Interaction Response", fontsize=9.5, fontweight='bold', color=C_DARK, pad=6)
ax1.legend(loc='lower left', frameon=True, framealpha=0.92, fontsize=7.2, edgecolor='#CBD5E1')
ax1.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')

# Note on parallel slopes
ax1.text(0.5, 52.4, r"Parallel curves: Interaction $+0.02$ pp ($p = 0.967$)", ha='center', va='bottom', fontsize=7.0, color=C_MUTED)

# =========================================================================
# Panel (b): Marginal Main Effects Forest Plot
# =========================================================================
y_pos = np.arange(len(effects))[::-1]
ax2.axvline(0.0, color=C_DARK, linestyle='--', lw=1.0, alpha=0.7, zorder=1)
ax2.axvspan(-0.5, 0.5, facecolor='#F1F5F9', alpha=0.5, zorder=0)

y_labels = []
for idx, (label, diff_arr, col) in enumerate(effects):
    y = y_pos[idx]
    y_labels.append(label)
    m = np.mean(diff_arr)
    se = stats.sem(diff_arr)
    ci_lo = m - t_crit * se
    ci_hi = m + t_crit * se
    
    # 95% Confidence Interval bar
    ax2.plot([ci_lo, ci_hi], [y, y], color=col, lw=2.0, zorder=3)
    ax2.plot([ci_lo, ci_lo], [y - 0.08, y + 0.08], color=col, lw=2.0, zorder=3)
    ax2.plot([ci_hi, ci_hi], [y - 0.08, y + 0.08], color=col, lw=2.0, zorder=3)
    
    # Point estimate marker
    ax2.scatter(m, y, color=col, s=40, zorder=5)
    
    # Tag
    sign_str = f"{m:+.2f} pp"
    ax2.text(m, y + 0.16, sign_str, ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=col,
             bbox=dict(boxstyle="round,pad=0.1", facecolor='white', edgecolor='none', alpha=0.92))

ax2.set_yticks(y_pos)
ax2.set_yticklabels(y_labels, fontsize=7.8, fontweight='bold')
ax2.set_xlim(-3.6, 2.2)
ax2.set_ylim(-0.6, 2.8)
ax2.set_xlabel("Marginal Paired Effect (pp)", fontsize=8.8, fontweight='bold', color=C_DARK)
ax2.set_title("(b) Marginal Contrasts (95% CI)", fontsize=9.5, fontweight='bold', color=C_DARK, pad=6)
ax2.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')

plt.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.18)

pdf_out = os.path.join(FIGURES_DIR, "figure_factorial_decomposition.pdf")
png_out = os.path.join(FIGURES_DIR, "figure_factorial_decomposition.png")
fig.savefig(pdf_out, bbox_inches='tight', pad_inches=0.02)
fig.savefig(png_out, bbox_inches='tight', pad_inches=0.02, dpi=300)

for d in [CLEAN_FIGURES_DIR]:
    fig.savefig(os.path.join(d, "figure_factorial_decomposition.pdf"), bbox_inches='tight', pad_inches=0.02)
    fig.savefig(os.path.join(d, "figure_factorial_decomposition.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)

plt.close(fig)
print(f"Figure 3 successfully generated: {pdf_out}")
