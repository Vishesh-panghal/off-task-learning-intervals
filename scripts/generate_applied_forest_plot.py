#!/usr/bin/env python3
"""
Generate publication-grade Figure 6: Confirmatory Paired Contrasts in Applied Clinical Transfer.
Designed at native 6.5-inch width:
- Panel (a): Seed-Level Paired Profiles on ISIC 2019 (N = 8 Seeds).
- Panel (b): Confirmatory Contrasts & Inferential Sensitivity (C3 - C0, S1 - C3, S1 - C0).
- Panel (c) (Longitudinal Representation Dimensionality) exported as supplementary figure.

Typography:
- Panel titles: 9.5 pt bold
- Axis labels: 8.8 pt bold
- Tick labels: 8.0 pt bold
- Point tags & legend: 7.5 pt
"""

import os
import numpy as np
import pandas as pd
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

CSV_PATH = os.path.join(BASE_DIR, "results", "applied_primary_runs.csv")
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(BASE_DIR, "clean_repo", "results", "applied_primary_runs.csv")

df = pd.read_csv(CSV_PATH)
seeds = [1001 + i for i in range(8)]
df_filtered = df[df['seed'].isin(seeds)].copy()

c0_runs = df_filtered[df_filtered['condition'] == 'C0'].sort_values('seed')
c3_runs = df_filtered[df_filtered['condition'] == 'C3'].sort_values('seed')
s1_runs = df_filtered[df_filtered['condition'] == 'S1'].sort_values('seed')

top1_c0 = c0_runs['theta3_top1'].values
top1_c3 = c3_runs['theta3_top1'].values
top1_s1 = s1_runs['theta3_top1'].values

eff_p1_c0 = c0_runs['theta1_eff_rank'].values
eff_p2_c0 = c0_runs['theta2_eff_rank'].values
eff_p3_c0 = c0_runs['theta3_eff_rank'].values

eff_p1_c3 = c3_runs['theta1_eff_rank'].values
eff_p2_c3 = c3_runs['theta2_eff_rank'].values
eff_p3_c3 = c3_runs['theta3_eff_rank'].values

eff_p1_s1 = s1_runs['theta1_eff_rank'].values
eff_p2_s1 = s1_runs['theta2_eff_rank'].values
eff_p3_s1 = s1_runs['theta3_eff_rank'].values

C_BLUE = '#1F77B4'
C_PURPLE = '#8E44AD'
C_ORANGE = '#D35400'
C_DARK = '#0F172A'
C_RED = '#D62728'
C_MUTED = '#475569'

# =========================================================================
# MAIN MANUSCRIPT FIGURE 6: 2 Panels (Native 6.5 x 2.45 in)
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 1.15), dpi=300,
                               gridspec_kw={'width_ratios': [0.90, 1.35], 'wspace': 0.52})

# --- Panel (a): Seed-Level Profiles ---
x_c0, x_c3, x_s1 = 0.0, 1.0, 2.0
xs = [x_c0, x_c3, x_s1]

for i in range(len(seeds)):
    y_vals = [top1_c0[i], top1_c3[i], top1_s1[i]]
    ax1.plot(xs, y_vals, color='#CBD5E1', lw=0.9, alpha=0.75, zorder=2)
    ax1.scatter(xs, y_vals, color=[C_BLUE, C_PURPLE, C_ORANGE], s=20, alpha=0.85, zorder=3)

mean_c0, mean_c3, mean_s1 = np.mean(top1_c0), np.mean(top1_c3), np.mean(top1_s1)
ax1.plot(xs, [mean_c0, mean_c3, mean_s1], color=C_DARK, lw=2.4, marker='D', markersize=5.5, zorder=5, label='Grand Mean')

ax1.text(x_c0, mean_c0 + 0.35, f"{mean_c0:.2f}%", ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=C_BLUE)
ax1.text(x_c3, mean_c3 - 0.35, f"{mean_c3:.2f}%", ha='center', va='top', fontsize=7.5, fontweight='bold', color=C_PURPLE)
ax1.text(x_s1, mean_s1 + 0.35, f"{mean_s1:.2f}%", ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=C_ORANGE)

ax1.set_xlim(-0.4, 2.4)
ax1.set_ylim(83.0, 87.0)
ax1.set_xticks(xs)
ax1.set_xticklabels([
    r"$\mathbf{C_0}$ Continuous" + "\n(12k Target)",
    r"$\mathbf{C_3}$ Reset Ctrl" + "\n(Head Swap)",
    r"$\mathbf{S_1}$ Auxiliary" + "\n(2k Detour)"
], fontsize=7.8, fontweight='bold')
ax1.set_ylabel("Terminal ISIC Acc. (%)", fontsize=8.8, fontweight='bold', color=C_DARK)
ax1.set_title("(a) Seed-Level Paired Profiles (N = 8)", fontsize=9.5, fontweight='bold', color=C_DARK, pad=6)
ax1.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1', axis='y')
ax1.text(1.0, 83.2, r"All $|d| \leq 0.49$ pp, all $p > 0.20$", ha='center', va='bottom', fontsize=7.2, color=C_MUTED)

# --- Panel (b): Confirmatory Contrasts ---
contrasts_b = [
    (r"$C_3(12{\rm k}) - C_0(12{\rm k})$" + "\n(Procedural Reset)", -0.49, 1.01, [-1.34, 0.35], [-1.22, 0.28], 0.209),
    (r"$S_1(12{\rm k}) - C_3(12{\rm k})$" + "\n(Detour vs. Reset)", +0.08, 0.73, [-0.53, 0.69], [-0.44, 0.61], 0.767),
    (r"$S_1(12{\rm k}) - C_0(12{\rm k})$" + "\n(Detour vs. Baseline)", -0.41, 1.08, [-1.31, 0.49], [-1.25, 0.40], 0.315)
]

y_pos_b = [2.0, 1.0, 0.0]
ax2.axvline(0.0, color=C_RED, linestyle='--', lw=1.1, zorder=1)
ax2.axvspan(-0.5, 0.5, facecolor='#EAFAF1', alpha=0.35, zorder=0)

y_labels_b = []
for idx, (label, m, sd, t_ci, b_ci, p_val) in enumerate(contrasts_b):
    y = y_pos_b[idx]
    y_labels_b.append(label)
    
    # Parametric t CI
    ax2.plot(t_ci, [y + 0.05, y + 0.05], color=C_BLUE, lw=1.6, zorder=3)
    ax2.plot([t_ci[0], t_ci[0]], [y + 0.01, y + 0.09], color=C_BLUE, lw=1.6, zorder=3)
    ax2.plot([t_ci[1], t_ci[1]], [y + 0.01, y + 0.09], color=C_BLUE, lw=1.6, zorder=3)
    
    # Bootstrap CI
    ax2.plot(b_ci, [y - 0.05, y - 0.05], color=C_PURPLE, lw=2.6, alpha=0.85, zorder=3)
    
    # Marker
    ax2.scatter(m, y, color=C_DARK, s=35, zorder=5)
    
    # Label
    sign_m = f"{m:+.2f} pp"
    ax2.text(m, y + 0.16, sign_m, ha='center', va='bottom', fontsize=7.4, fontweight='bold', color=C_DARK,
             bbox=dict(boxstyle="round,pad=0.1", facecolor='white', edgecolor='none', alpha=0.92))

ax2.set_yticks(y_pos_b)
ax2.set_yticklabels(y_labels_b, fontsize=7.8, fontweight='bold')
ax2.set_xlim(-1.6, 1.2)
ax2.set_ylim(-0.5, 2.7)
ax2.set_xlabel("Paired Difference in Top-1 Accuracy (pp)", fontsize=8.8, fontweight='bold', color=C_DARK)
ax2.set_title("(b) Confirmatory Paired Contrasts", fontsize=9.5, fontweight='bold', color=C_DARK, pad=6)
ax2.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')

line_t = plt.Line2D([0], [0], color=C_BLUE, lw=1.6, label="95% Paired t-CI")
line_b = plt.Line2D([0], [0], color=C_PURPLE, lw=2.6, label="95% Bootstrap CI")
line_z = plt.Line2D([0], [0], color=C_RED, linestyle='--', lw=1.1, label="Null (0 pp)")
ax2.legend(handles=[line_t, line_b, line_z], loc='upper right', frameon=True, framealpha=0.92, fontsize=7.2, edgecolor='#CBD5E1')

plt.subplots_adjust(left=0.125, right=0.98, top=0.85, bottom=0.22)

pdf_out = os.path.join(FIGURES_DIR, "figure_applied_contrasts_forest_plot.pdf")
png_out = os.path.join(FIGURES_DIR, "figure_applied_contrasts_forest_plot.png")
fig.savefig(pdf_out, bbox_inches='tight', pad_inches=0.06)
fig.savefig(png_out, bbox_inches='tight', pad_inches=0.06, dpi=300)

for d in [CLEAN_FIGURES_DIR]:
    fig.savefig(os.path.join(d, "figure_applied_contrasts_forest_plot.pdf"), bbox_inches='tight', pad_inches=0.06)
    fig.savefig(os.path.join(d, "figure_applied_contrasts_forest_plot.png"), bbox_inches='tight', pad_inches=0.06, dpi=300)

plt.close(fig)
print(f"Figure 6 successfully generated: {pdf_out}")

# =========================================================================
# SUPPLEMENTARY FIGURE S2: Longitudinal Effective Rank on ISIC 2019
# =========================================================================
fig_s, ax_s = plt.subplots(figsize=(4.5, 2.5), dpi=300)
x_p1, x_p2, x_p3 = 1.0, 2.0, 3.0
xs_eff = [x_p1, x_p2, x_p3]

m_eff_c0 = [np.mean(eff_p1_c0), np.mean(eff_p2_c0), np.mean(eff_p3_c0)]
m_eff_c3 = [np.mean(eff_p1_c3), np.mean(eff_p2_c3), np.mean(eff_p3_c3)]
m_eff_s1 = [np.mean(eff_p1_s1), np.mean(eff_p2_s1), np.mean(eff_p3_s1)]

ax_s.plot(xs_eff, m_eff_c0, color=C_BLUE, lw=1.8, marker='o', markersize=4.5, label=r"$C_0$ Continuous")
ax_s.plot(xs_eff, m_eff_c3, color=C_PURPLE, lw=1.8, linestyle='--', marker='s', markersize=4.5, label=r"$C_3$ Reset Ctrl")
ax_s.plot(xs_eff, m_eff_s1, color=C_ORANGE, lw=2.0, marker='D', markersize=4.5, label=r"$S_1$ Auxiliary Detour")

ax_s.text(x_p2, m_eff_s1[1] - 35, f"{m_eff_s1[1]:.1f}\n(-359 dims, p < 10^-9)", ha='center', va='top', fontsize=7.2, fontweight='bold', color=C_ORANGE)
ax_s.text(x_p3, m_eff_s1[2] - 35, f"{m_eff_s1[2]:.1f}\n(Recovered)", ha='center', va='top', fontsize=7.2, fontweight='bold', color=C_ORANGE)

ax_s.set_xticks(xs_eff)
ax_s.set_xticklabels([r"Phase 1 ($\boldsymbol{\theta}_1$, 5k)", r"Phase 2 ($\boldsymbol{\theta}_2$, 7k)", r"Phase 3 ($\boldsymbol{\theta}_3$, 12k)"],
                     fontsize=8.0, fontweight='bold')
ax_s.set_ylabel("Effective Rank (Dimensions)", fontsize=8.8, fontweight='bold')
ax_s.set_title("Clinical Feature Dimensionality Dynamics (ISIC 2019)", fontsize=9.5, fontweight='bold')
ax_s.set_ylim(700, 1250)
ax_s.grid(True, linestyle=':', alpha=0.4)
ax_s.legend(loc='lower left', fontsize=7.5)

plt.subplots_adjust(left=0.14, right=0.96, top=0.88, bottom=0.18)

supp_pdf = os.path.join(FIGURES_DIR, "figure_applied_effrank_longitudinal.pdf")
fig_s.savefig(supp_pdf, bbox_inches='tight', pad_inches=0.06)
for d in [CLEAN_FIGURES_DIR]:
    fig_s.savefig(os.path.join(d, "figure_applied_effrank_longitudinal.pdf"), bbox_inches='tight', pad_inches=0.06)
plt.close(fig_s)
print(f"Supplementary Figure S2 generated: {supp_pdf}")
