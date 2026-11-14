#!/usr/bin/env python3
"""
Generate publication-grade Figure 7: Transient Representation Collapse and Manifold Dynamics.
Redesigned with authentic Seaborn statistical aesthetics:
- Panel (a): Singular Value Spectrum (Scree Plot) showing acute compression at theta_2 and recovery at theta_3.
- Panels (b-d): 2D t-SNE Projections of 2,000 clinical test lesions at core milestones:
  - (b) Step 5k: Target Warmup (Intact diagnostic manifolds)
  - (c) Step 7k: Detour Shock (Catastrophic dispersion)
  - (d) Step 12k: Relearning Snapback (Diagnostic re-clustering)
- Clean shared 8-class borderless legend across the bottom.
- Secondary panels (cumulative energy & C0 continuous control) exported to supplementary material.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

NPZ_CANDIDATES = [
    os.path.join(BASE_DIR, "results", "applied_manifold_scree_data.npz"),
    os.path.join(ROOT_DIR, "results", "applied_manifold_scree_data.npz"),
    os.path.join(ROOT_DIR, "clean_repo", "results", "applied_manifold_scree_data.npz")
]

NPZ_PATH = None
for p in NPZ_CANDIDATES:
    if os.path.exists(p):
        NPZ_PATH = p
        break

if NPZ_PATH is None:
    raise FileNotFoundError("Could not find applied_manifold_scree_data.npz in any candidate location.")

data = np.load(NPZ_PATH)

C_DARK = '#1e293b'
C_MUTED = '#64748b'
C_BLUE = '#4c72b0'       # Warmup theta1
C_RED = '#c44e52'        # Detour Collapse S1 theta2
C_GREEN = '#55a868'      # Relearning Recovery S1 theta3
C_PURPLE = '#8172b3'     # C0 Continuous Control

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
CLASS_COLORS = [
    '#e74c3c',  # MEL: Red
    '#3498db',  # NV: Blue
    '#e67e22',  # BCC: Orange
    '#9b59b6',  # AK: Purple
    '#2ecc71',  # BKL: Green
    '#1abc9c',  # DF: Teal
    '#f1c40f',  # VASC: Yellow
    '#34495e',  # SCC: Dark Slate
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

tsne_th1 = data['tsne_S1_theta1']
tsne_th2 = data['tsne_S1_theta2']
tsne_th3 = data['tsne_S1_theta3']
tsne_c0 = data['tsne_C0_theta2']

# =========================================================================
# MAIN MANUSCRIPT FIGURE 7: 4 Panels in 1 Row (Spacious Native 6.5 x 2.50 in)
# =========================================================================
fig = plt.figure(figsize=(6.5, 2.50), dpi=300)
gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1.0, 1.0, 1.0], wspace=0.28)

ax_scree = fig.add_subplot(gs[0, 0])
ax_tsne1 = fig.add_subplot(gs[0, 1])
ax_tsne2 = fig.add_subplot(gs[0, 2])
ax_tsne3 = fig.add_subplot(gs[0, 3])

# --- Panel (a): Scree Plot ---
k_range = np.arange(1, 151)
ax_scree.plot(k_range, p1[:150] * 100, color=C_BLUE, lw=1.6, label=r"$\boldsymbol{\theta}_1$ Warmup (1147)")
ax_scree.plot(k_range, p2[:150] * 100, color=C_RED, lw=1.8, label=r"$\boldsymbol{\theta}_2$ Detour (787)")
ax_scree.plot(k_range, p3[:150] * 100, color=C_GREEN, lw=1.6, linestyle='--', label=r"$\boldsymbol{\theta}_3$ Recov (1140)")

ax_scree.set_yscale('log')
ax_scree.set_xlim(1, 150)
ax_scree.set_ylim(0.015, 6.0)
ax_scree.set_xticks([1, 50, 100, 150])
ax_scree.set_xticklabels(["1", "50", "100", "150"], fontsize=7.6)
ax_scree.set_yticks([0.02, 0.1, 1.0])
ax_scree.set_yticklabels(["0.02%", "0.1%", "1.0%"], fontsize=7.6)
ax_scree.set_xlabel("Singular Value Index ($k$)", fontsize=8.4, color=C_DARK)
ax_scree.set_ylabel(r"Singular Energy ($p_k$)", fontsize=8.4, color=C_DARK)
ax_scree.set_title("(a) Singular Value Spectrum\nPenultimate SVD", fontsize=8.2, fontweight='semibold', color=C_DARK, pad=6)
ax_scree.grid(axis='y', linestyle='--', alpha=0.35, color='#cbd5e1')
sns.despine(ax=ax_scree, trim=False)

# Borderless legend inside scree plot in open upper right
ax_scree.legend(loc='upper right', frameon=False, fontsize=7.2, handlelength=1.5, handletextpad=0.4)

# Clean statistical annotation inside scree plot in open lower left
ax_scree.text(0.06, 0.08, r"$\Delta\mathrm{EffRank} = -359$ dims ($p < 10^{-9}$)",
              transform=ax_scree.transAxes, fontsize=7.0, color=C_RED, fontweight='medium')

# --- t-SNE panels helper ---
def draw_tsne(ax, emb, title_line1, title_line2):
    ax.scatter(emb[:, 0], emb[:, 1], c=[CLASS_COLORS[int(l)] for l in labels],
               s=2.8, alpha=0.65, edgecolors='none', rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{title_line1}\n{title_line2}", fontsize=8.2, fontweight='semibold', color=C_DARK, pad=6)
    for spine in ax.spines.values():
        spine.set_edgecolor('#cbd5e1')
        spine.set_linewidth(0.8)

draw_tsne(ax_tsne1, tsne_th1, "(b) Step 5k Warmup", "Intact Manifold")
draw_tsne(ax_tsne2, tsne_th2, "(c) Step 7k Detour", "Transient Scatter")
draw_tsne(ax_tsne3, tsne_th3, "(d) Step 12k Recovery", "Diagnostic Re-cluster")

# Global Legend for 8 Classes at bottom of t-SNE panels
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=CLASS_COLORS[i],
           markersize=5.0, label=CLASS_NAMES[i])
    for i in range(8)
]
fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.63, 0.03),
           ncol=8, frameon=False, fontsize=7.2,
           handletextpad=0.2, columnspacing=0.8)

plt.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.16)

for d in FIGURE_DIRS:
    pdf_out = os.path.join(d, "figure_applied_manifold_and_scree.pdf")
    png_out = os.path.join(d, "figure_applied_manifold_and_scree.png")
    fig.savefig(pdf_out, bbox_inches='tight', pad_inches=0.04)
    fig.savefig(png_out, bbox_inches='tight', pad_inches=0.04, dpi=300)
    print(f"Saved: {pdf_out}")

plt.close(fig)
print("Figure 7 successfully generated.")

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
ax_s1.axhline(80.0, color=C_MUTED, linestyle='--', lw=0.9, alpha=0.7)
ax_s1.set_xlim(1, 1000)
ax_s1.set_ylim(20, 102)
ax_s1.set_xlabel("Cumulative Dimensions ($k$)", fontsize=8.4, color=C_DARK)
ax_s1.set_ylabel("Cumulative Variance (%)", fontsize=8.4, color=C_DARK)
ax_s1.set_title("Cumulative Spectral Energy Distribution", fontsize=9.2, fontweight='semibold', color=C_DARK)
ax_s1.grid(axis='y', linestyle='--', alpha=0.35, color='#cbd5e1')
sns.despine(ax=ax_s1, trim=False)
ax_s1.legend(loc='lower right', frameon=False, fontsize=7.2)

# C0 continuous baseline t-SNE at step 7k
ax_s2.scatter(tsne_c0[:, 0], tsne_c0[:, 1], c=[CLASS_COLORS[int(l)] for l in labels],
              s=3.0, alpha=0.65, edgecolors='none', rasterized=True)
ax_s2.set_xticks([])
ax_s2.set_yticks([])
ax_s2.set_title("Continuous $C_0$ at Step 7k\nPreserved Manifolds", fontsize=9.2, fontweight='semibold', color=C_DARK)
for spine in ax_s2.spines.values():
    spine.set_edgecolor('#cbd5e1')
    spine.set_linewidth(0.8)

plt.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.18)

for d in FIGURE_DIRS:
    supp_pdf = os.path.join(d, "figure_applied_manifold_cumulative_and_c0.pdf")
    fig_s.savefig(supp_pdf, bbox_inches='tight', pad_inches=0.04)

plt.close(fig_s)
print("Supplementary Figure S3 generated.")
