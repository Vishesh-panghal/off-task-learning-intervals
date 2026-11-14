#!/usr/bin/env python3
"""
Generate publication-grade Figure 6: Confirmatory Paired Contrasts in Applied Clinical Transfer.
Redesigned with authentic Seaborn statistical aesthetics:
- Panel (a): Seed-Level Paired Profiles on ISIC 2019 (N = 8 Seeds).
- Panel (b): Confirmatory Contrasts & Inferential Sensitivity (C3 - C0, S1 - C3, S1 - C0).
- Panel (c) (Longitudinal Representation Dimensionality) exported as supplementary figure.

Features:
- Seaborn 'ticks' theme with clean L-frames (sns.despine)
- Subtle neutral seed lines with bold population grand mean trajectory
- Soft equivalence band (±0.5 pp) with vertical dashed null reference line
- Borderless outside legend on top
- Clean unboxed value labels
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks", font="sans-serif")
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['mathtext.fontset'] = 'cm'

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..")) if os.path.basename(BASE_DIR) == "clean_repo" else BASE_DIR

FIGURE_DIRS = [
    os.path.join(BASE_DIR, "figures"),
    os.path.join(ROOT_DIR, "figures"),
    os.path.join(ROOT_DIR, "clean_repo", "figures")
]
FIGURE_DIRS = list(dict.fromkeys([d for d in FIGURE_DIRS if os.path.exists(os.path.dirname(d))]))
for d in FIGURE_DIRS:
    os.makedirs(d, exist_ok=True)

CSV_CANDIDATES = [
    os.path.join(BASE_DIR, "results", "applied_primary_runs.csv"),
    os.path.join(ROOT_DIR, "results", "applied_primary_runs.csv"),
    os.path.join(ROOT_DIR, "clean_repo", "results", "applied_primary_runs.csv")
]

CSV_PATH = None
for p in CSV_CANDIDATES:
    if os.path.exists(p):
        CSV_PATH = p
        break

if CSV_PATH is None:
    raise FileNotFoundError("Could not find applied_primary_runs.csv in any candidate location.")

df = pd.read_csv(CSV_PATH)
seeds = [1001 + i for i in range(8)]
df_filtered = df[df['seed'].isin(seeds)].copy()

c0_runs = df_filtered[df_filtered['condition'] == 'C0'].sort_values('seed')
c3_runs = df_filtered[df_filtered['condition'] == 'C3'].sort_values('seed')
s1_runs = df_filtered[df_filtered['condition'] == 'S1'].sort_values('seed')

top1_c0 = c0_runs['theta3_top1'].values
top1_c3 = c3_runs['theta3_top1'].values
top1_s1 = s1_runs['theta3_top1'].values

mean_c0, mean_c3, mean_s1 = np.mean(top1_c0), np.mean(top1_c3), np.mean(top1_s1)

eff_p1_c0 = c0_runs['theta1_eff_rank'].values
eff_p2_c0 = c0_runs['theta2_eff_rank'].values
eff_p3_c0 = c0_runs['theta3_eff_rank'].values

eff_p1_c3 = c3_runs['theta1_eff_rank'].values
eff_p2_c3 = c3_runs['theta2_eff_rank'].values
eff_p3_c3 = c3_runs['theta3_eff_rank'].values

eff_p1_s1 = s1_runs['theta1_eff_rank'].values
eff_p2_s1 = s1_runs['theta2_eff_rank'].values
eff_p3_s1 = s1_runs['theta3_eff_rank'].values

C_DARK = '#1e293b'
C_GRAY = '#64748b'
C_GRAY_LIGHT = '#94a3b8'
C_BLUE = '#4c72b0'       # C0 Continuous
C_PURPLE = '#8172b3'     # C3 Reset Ctrl
C_RED = '#c44e52'        # S1 Auxiliary Detour

# =========================================================================
# MAIN MANUSCRIPT FIGURE 6: 2 Panels (Native 6.5 x 2.50 in)
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.40), dpi=300,
                               gridspec_kw={'width_ratios': [0.95, 1.25], 'wspace': 0.65})

# --- Panel (a): Seed-Level Profiles (N = 8) ---
xs = [0.0, 1.0, 2.0]
for i in range(len(seeds)):
    y_vals = [top1_c0[i], top1_c3[i], top1_s1[i]]
    ax1.plot(xs, y_vals, color=C_GRAY_LIGHT, lw=0.9, alpha=0.55, zorder=2)
    ax1.scatter(xs, y_vals, color=[C_BLUE, C_PURPLE, C_RED], s=18, alpha=0.65, zorder=3)

# Bold Population Mean Trajectory
ax1.plot(xs, [mean_c0, mean_c3, mean_s1], color=C_DARK, lw=2.2, marker='D', markersize=6.0,
         markerfacecolor=C_DARK, markeredgecolor='white', markeredgewidth=1.0, zorder=5, label='Grand Mean')

# Unboxed directional percentage callouts
ax1.text(-0.08, mean_c0, f"{mean_c0:.2f}%", ha='right', va='center', fontsize=7.8, color=C_BLUE, fontweight='medium')
ax1.text(1.0, 83.82, f"{mean_c3:.2f}%", ha='center', va='top', fontsize=7.8, color=C_PURPLE, fontweight='medium')
ax1.text(2.08, mean_s1, f"{mean_s1:.2f}%", ha='left', va='center', fontsize=7.8, color=C_RED, fontweight='medium')

ax1.set_xlim(-0.45, 2.45)
ax1.set_ylim(83.0, 87.0)
ax1.set_yticks([83, 84, 85, 86, 87])
ax1.set_yticklabels(['83%', '84%', '85%', '86%', '87%'], fontsize=7.8)
ax1.set_xticks(xs)
ax1.set_xticklabels([
    r"$\mathbf{C_0}$ Continuous" + "\n(12k Target)",
    r"$\mathbf{C_3}$ Reset Ctrl" + "\n(Head Swap)",
    r"$\mathbf{S_1}$ Auxiliary" + "\n(2k Detour)"
], fontsize=7.8)
ax1.set_ylabel("Terminal ISIC Acc. (%)", fontsize=8.6, color=C_DARK)
ax1.set_title("(a) Seed-Level Paired Profiles (N = 8)", fontsize=9.2, fontweight='semibold', color=C_DARK, pad=26)
ax1.grid(axis='y', linestyle='--', alpha=0.35, color='#cbd5e1')
sns.despine(ax=ax1, trim=False)

# --- Panel (b): Confirmatory Paired Contrasts ---
contrasts_b = [
    (r"$C_3(12\mathrm{k}) - C_0(12\mathrm{k})$" + "\n(Procedural Reset)", -0.49, [-1.34, 0.35], [-1.22, 0.28]),
    (r"$S_1(12\mathrm{k}) - C_3(12\mathrm{k})$" + "\n(Detour vs. Reset)", +0.08, [-0.53, 0.69], [-0.44, 0.61]),
    (r"$S_1(12\mathrm{k}) - C_0(12\mathrm{k})$" + "\n(Detour vs. Baseline)", -0.41, [-1.31, 0.49], [-1.25, 0.40])
]

y_pos = [2.0, 1.0, 0.0]

# 1. Soft Equivalence Region
ax2.axvspan(-0.5, 0.5, facecolor='#2ca02c', alpha=0.07, zorder=0)
ax2.text(0.0, 2.65, "Equivalence Band (±0.5 pp)", ha='center', va='bottom', fontsize=7.2, fontweight='medium', color='#2d7f36')

# 2. Vertical reference line stopping safely below header text
ax2.vlines(0.0, ymin=-0.45, ymax=2.48, color='#64748b', linestyle='--', lw=1.0, alpha=0.8, zorder=1)

# 3. Horizontal error bars
for idx, (label, m, t_ci, b_ci) in enumerate(contrasts_b):
    y = y_pos[idx]
    col = C_BLUE if idx == 2 else C_GRAY
    
    # Parametric t CI (outer bar with caps)
    ax2.errorbar(m, y + 0.06, xerr=[[m - t_ci[0]], [t_ci[1] - m]], fmt='none', color=col,
                 capsize=3.5, capthick=1.0, elinewidth=1.4, zorder=3)
    # Bootstrap CI (slightly offset below, purple)
    ax2.errorbar(m, y - 0.06, xerr=[[m - b_ci[0]], [b_ci[1] - m]], fmt='none', color=C_PURPLE,
                 capsize=0, elinewidth=2.2, alpha=0.75, zorder=3)
    # Scatter point
    ax2.scatter(m, y, color=col, s=36, zorder=5, edgecolor='white', linewidth=0.8)
    
    # Value label - clean unboxed, positioned to clear zero line
    sign_m = f"{m:+.2f} pp" if m != 0 else "0.00 pp"
    if idx == 1:
        ax2.text(0.12, y + 0.18, sign_m, ha='left', va='bottom', fontsize=7.5, color=col, fontweight='medium')
    else:
        ax2.text(m, y + 0.18, sign_m, ha='center', va='bottom', fontsize=7.5, color=col, fontweight='medium')

y_labels = [c[0] for c in contrasts_b]
ax2.set_yticks(y_pos)
ax2.set_yticklabels(y_labels, fontsize=7.8)
ax2.set_xlim(-1.6, 1.2)
ax2.set_ylim(-0.55, 2.85)
ax2.set_xticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
ax2.set_xticklabels(['-1.5', '-1.0', '-0.5', '0.0', '+0.5', '+1.0'], fontsize=7.8)
ax2.set_xlabel("Paired Difference in Top-1 Accuracy (pp)", fontsize=8.6, color=C_DARK)
ax2.set_title("(b) Confirmatory Paired Contrasts", fontsize=9.2, fontweight='semibold', color=C_DARK, pad=26)
ax2.grid(axis='y', linestyle='--', alpha=0.35, color='#cbd5e1')
sns.despine(ax=ax2, trim=False)

# 4. Clean borderless legend placed outside on top
line_t = plt.Line2D([0], [0], color=C_BLUE, lw=1.4, label="95% Paired t-CI")
line_b = plt.Line2D([0], [0], color=C_PURPLE, lw=2.2, label="95% Bootstrap CI")
ax2.legend(handles=[line_t, line_b], loc='lower center', bbox_to_anchor=(0.5, 1.02),
           ncol=2, frameon=False, fontsize=7.4)

plt.subplots_adjust(left=0.12, right=0.98, top=0.82, bottom=0.18)

for d in FIGURE_DIRS:
    pdf_out = os.path.join(d, "figure_applied_contrasts_forest_plot.pdf")
    png_out = os.path.join(d, "figure_applied_contrasts_forest_plot.png")
    fig.savefig(pdf_out, bbox_inches='tight', pad_inches=0.04)
    fig.savefig(png_out, bbox_inches='tight', pad_inches=0.04, dpi=300)
    print(f"Saved: {pdf_out}")

plt.close(fig)
print("Figure 6 successfully generated.")

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
ax_s.plot(xs_eff, m_eff_s1, color=C_RED, lw=2.0, marker='D', markersize=4.5, label=r"$S_1$ Auxiliary Detour")

ax_s.text(x_p2, m_eff_s1[1] - 35, f"{m_eff_s1[1]:.1f}\n(-359 dims, p < 10^-9)", ha='center', va='top', fontsize=7.2, fontweight='medium', color=C_RED)
ax_s.text(x_p3, m_eff_s1[2] - 35, f"{m_eff_s1[2]:.1f}\n(Recovered)", ha='center', va='top', fontsize=7.2, fontweight='medium', color=C_RED)

ax_s.set_xticks(xs_eff)
ax_s.set_xticklabels([r"Phase 1 ($\boldsymbol{\theta}_1$, 5k)", r"Phase 2 ($\boldsymbol{\theta}_2$, 7k)", r"Phase 3 ($\boldsymbol{\theta}_3$, 12k)"],
                     fontsize=8.0)
ax_s.set_ylabel("Effective Rank (Dimensions)", fontsize=8.6, color=C_DARK)
ax_s.set_title("Clinical Feature Dimensionality Dynamics (ISIC 2019)", fontsize=9.2, fontweight='semibold', color=C_DARK)
ax_s.set_ylim(700, 1250)
ax_s.grid(axis='y', linestyle='--', alpha=0.35, color='#cbd5e1')
sns.despine(ax=ax_s, trim=False)
ax_s.legend(loc='lower left', frameon=False, fontsize=7.5)

plt.subplots_adjust(left=0.14, right=0.96, top=0.88, bottom=0.18)

for d in FIGURE_DIRS:
    supp_pdf = os.path.join(d, "figure_applied_effrank_longitudinal.pdf")
    fig_s.savefig(supp_pdf, bbox_inches='tight', pad_inches=0.04)

plt.close(fig_s)
print("Supplementary Figure S2 generated.")
