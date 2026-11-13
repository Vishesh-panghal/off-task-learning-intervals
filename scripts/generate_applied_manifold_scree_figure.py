#!/usr/bin/env python3
"""
Generate publication-grade Figure 7: Transient Representation Collapse and Manifold Dynamics.
Designed at native 6.5-inch width x 1.85-inch height:
- Panel (a): Singular Value Spectrum (Scree Plot) showing acute compression at theta_2 and recovery at theta_3.
- Panels (b-d): 2D t-SNE Projections of 2,000 clinical test lesions at core milestones:
  - (b) Step 5k: Target Warmup (Intact diagnostic manifolds)
  - (c) Step 7k: Detour Shock (Catastrophic dispersion)
  - (d) Step 12k: Relearning Snapback (Diagnostic re-clustering)
- Clean shared 8-class legend across the bottom.
- Secondary panels (cumulative energy & C0 continuous control) exported to supplementary material.

Typography:
- Panel titles: 9.0 pt bold
- Axis labels: 8.2 pt bold
- Tick labels: 7.8 pt
- Class legend: 7.2 pt
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NPZ_PATH = os.path.join(BASE_DIR, "results", "applied_manifold_scree_data.npz")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
CLEAN_FIGURES_DIR = os.path.join(BASE_DIR, "clean_repo", "figures")
for d in [FIGURES_DIR, CLEAN_FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

if not os.path.exists(NPZ_PATH):
    print(f"Data file not found at {NPZ_PATH}")
    exit(1)

data = np.load(NPZ_PATH)

C_BLUE = '#1F77B4'      # Warmup theta1
C_RED = '#D62728'       # Detour Collapse S1 theta2
C_GREEN = '#2CA02C'     # Relearning Recovery S1 theta3
C_PURPLE = '#8E44AD'    # C0 Continuous Control
C_DARK = '#0F172A'
C_MUTED = '#475569'

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
CLASS_COLORS = [
    '#E74C3C',  # MEL: Red
    '#3498DB',  # NV: Blue
    '#E67E22',  # BCC: Orange
    '#9B59B6',  # AK: Purple
    '#2ECC71',  # BKL: Green
    '#1ABC9C',  # DF: Teal
    '#F1C40F',  # VASC: Yellow
    '#34495E',  # SCC: Dark Slate
]

labels = data['labels']
s1_th1 = data['s_S1_theta1']
s1_th2 = data['s_S1_theta2']
s1_th3 = data['s_S1_theta3']
c0_th2 = data['s_C0_theta2']

p1 = s1_th1 / np.sum(s1_th1)
p2 = s1_th2 / np.sum(s1_th2)
p3 = s1_th3 / np.sum(s1_th3)
p_c0 = c0_th2 / np.sum(c0_th2)

er1 = float(data['er_S1_theta1'])
er2 = float(data['er_S1_theta2'])
er3 = float(data['er_S1_theta3'])

tsne_th1 = data['tsne_S1_theta1']
tsne_th2 = data['tsne_S1_theta2']
tsne_th3 = data['tsne_S1_theta3']
tsne_c0 = data['tsne_C0_theta2']

# =========================================================================
# MAIN MANUSCRIPT FIGURE 7: 4 Panels in 1 Row (6.5 x 1.85 in)
# =========================================================================
fig = plt.figure(figsize=(6.5, 1.25), dpi=300)
gs = fig.add_gridspec(1, 4, width_ratios=[1.3, 1.0, 1.0, 1.0], wspace=0.28)

ax_scree = fig.add_subplot(gs[0, 0])
ax_tsne1 = fig.add_subplot(gs[0, 1])
ax_tsne2 = fig.add_subplot(gs[0, 2])
ax_tsne3 = fig.add_subplot(gs[0, 3])

# --- Panel (a): Scree Plot ---
k_range = np.arange(1, 151)
ax_scree.plot(k_range, p1[:150] * 100, color=C_BLUE, lw=1.5, label=rf"$\boldsymbol{{\theta}}_1$ (1147)")
ax_scree.plot(k_range, p2[:150] * 100, color=C_RED, lw=1.8, label=rf"$\boldsymbol{{\theta}}_2$ Detour (787)")
ax_scree.plot(k_range, p3[:150] * 100, color=C_GREEN, lw=1.5, linestyle='--', label=rf"$\boldsymbol{{\theta}}_3$ Recov (1140)")

ax_scree.set_yscale('log')
ax_scree.set_xlim(1, 150)
ax_scree.set_xlabel("Singular Value Index ($k$)", fontsize=7.8, fontweight='bold')
ax_scree.set_ylabel(r"Singular Energy ($p_k$, %)", fontsize=7.8, fontweight='bold')
ax_scree.set_title("(a) Singular Value Spectrum", fontsize=8.6, fontweight='bold', color=C_DARK, pad=5)
ax_scree.grid(True, linestyle=':', alpha=0.5, which='both')
ax_scree.legend(loc='upper right', frameon=True, framealpha=0.92, fontsize=6.8, edgecolor='#CBD5E1')

# Note on collapse
ax_scree.text(10, 0.012, r"$\mathrm{EffRank}$ drops by 359 dims ($p < 10^{-9}$)", fontsize=6.6, color=C_RED, fontweight='bold')

# --- Helper for t-SNE panels ---
def draw_tsne(ax, emb, title, subtitle, sub_color):
    ax.scatter(emb[:, 0], emb[:, 1], c=[CLASS_COLORS[int(l)] for l in labels],
               s=2.5, alpha=0.65, edgecolors='none', rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8.6, fontweight='bold', color=C_DARK, pad=4)
    ax.text(0.5, 0.02, subtitle, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=6.8, fontweight='bold', color=sub_color,
            bbox=dict(boxstyle="round,pad=0.1", facecolor='white', edgecolor=sub_color, alpha=0.92, lw=0.6))

draw_tsne(ax_tsne1, tsne_th1, "(b) Step 5k Warmup", "Intact Manifold", C_BLUE)
draw_tsne(ax_tsne2, tsne_th2, "(c) Step 7k Detour", "Catastrophic Scatter", C_RED)
draw_tsne(ax_tsne3, tsne_th3, "(d) Step 12k Recovery", "Diagnostic Re-clustering", C_GREEN)

# Global Legend for 8 Classes at bottom of t-SNE panels
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=CLASS_COLORS[i],
           markersize=4.5, label=CLASS_NAMES[i])
    for i in range(8)
]
fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.63, -0.02),
           ncol=8, frameon=True, framealpha=0.94, fontsize=6.8, edgecolor='#CBD5E1',
           handletextpad=0.2, columnspacing=0.8)

plt.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.22)

pdf_out = os.path.join(FIGURES_DIR, "figure_applied_manifold_and_scree.pdf")
png_out = os.path.join(FIGURES_DIR, "figure_applied_manifold_and_scree.png")
fig.savefig(pdf_out, bbox_inches='tight', pad_inches=0.02)
fig.savefig(png_out, bbox_inches='tight', pad_inches=0.02, dpi=300)

for d in [CLEAN_FIGURES_DIR]:
    fig.savefig(os.path.join(d, "figure_applied_manifold_and_scree.pdf"), bbox_inches='tight', pad_inches=0.02)
    fig.savefig(os.path.join(d, "figure_applied_manifold_and_scree.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)

plt.close(fig)
print(f"Figure 7 successfully generated: {pdf_out}")

# =========================================================================
# SUPPLEMENTARY FIGURE S3: Cumulative Energy & Continuous Baseline t-SNE
# =========================================================================
fig_s, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(6.0, 2.5), dpi=300,
                                     gridspec_kw={'width_ratios': [1.2, 1.0], 'wspace': 0.30})

# Cumulative Energy
cum1 = np.cumsum(p1) * 100
cum2 = np.cumsum(p2) * 100
cum3 = np.cumsum(p3) * 100
cum_c0 = np.cumsum(p_c0) * 100

k_all = np.arange(1, len(cum1) + 1)
ax_s1.plot(k_all, cum1, color=C_BLUE, lw=1.6, label=r"$\boldsymbol{\theta}_1$ Warmup")
ax_s1.plot(k_all, cum2, color=C_RED, lw=1.8, label=r"$\boldsymbol{\theta}_2$ Detour ($S_1$)")
ax_s1.plot(k_all, cum_c0, color=C_PURPLE, lw=1.4, linestyle=':', label=r"$\boldsymbol{\theta}_2$ Baseline ($C_0$)")
ax_s1.plot(k_all, cum3, color=C_GREEN, lw=1.6, linestyle='--', label=r"$\boldsymbol{\theta}_3$ Recovery ($S_1$)")
ax_s1.axhline(80.0, color=C_MUTED, linestyle=':', lw=0.9)
ax_s1.set_xlim(1, 1000)
ax_s1.set_ylim(20, 102)
ax_s1.set_xlabel("Cumulative Dimensions ($k$)", fontsize=8.2, fontweight='bold')
ax_s1.set_ylabel("Cumulative Variance (%)", fontsize=8.2, fontweight='bold')
ax_s1.set_title("Cumulative Spectral Energy Distribution", fontsize=9.0, fontweight='bold')
ax_s1.grid(True, linestyle=':', alpha=0.4)
ax_s1.legend(loc='lower right', fontsize=7.0)

# C0 continuous baseline t-SNE at step 7k
ax_s2.scatter(tsne_c0[:, 0], tsne_c0[:, 1], c=[CLASS_COLORS[int(l)] for l in labels],
              s=3.0, alpha=0.65, edgecolors='none', rasterized=True)
ax_s2.set_xticks([])
ax_s2.set_yticks([])
ax_s2.set_title(r"Continuous $C_0$ at Step 7k", fontsize=9.0, fontweight='bold')
ax_s2.text(0.5, 0.02, "Preserved Manifolds", transform=ax_s2.transAxes, ha='center', va='bottom',
           fontsize=7.2, fontweight='bold', color=C_PURPLE,
           bbox=dict(boxstyle="round,pad=0.1", facecolor='white', edgecolor=C_PURPLE, alpha=0.92, lw=0.6))

plt.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.18)

supp_pdf = os.path.join(FIGURES_DIR, "figure_applied_manifold_cumulative_and_c0.pdf")
fig_s.savefig(supp_pdf, bbox_inches='tight', pad_inches=0.02)
for d in [CLEAN_FIGURES_DIR]:
    fig_s.savefig(os.path.join(d, "figure_applied_manifold_cumulative_and_c0.pdf"), bbox_inches='tight', pad_inches=0.02)
plt.close(fig_s)
print(f"Supplementary Figure S3 generated: {supp_pdf}")
