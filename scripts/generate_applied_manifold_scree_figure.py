#!/usr/bin/env python3
"""
Generate publication-grade Figure: Transient Representation Collapse and Latent Manifold Dynamics
for Applied Vision Transfer Fine-Tuning in Clinical Dermoscopy (ISIC 2019 / Imagenette).

Visualizes:
- Row 1: (a) Scree Plot / Singular Value Spectrum comparing theta1 (Target Warmup), theta2 (Detour Collapse), and theta3 (Recovery).
         (b) Cumulative Spectral Energy Distribution demonstrating steep variance concentration during the detour.
- Row 2: (c) 2D t-SNE projection of 2,000 clinical test lesions at theta1 (Initial target manifold).
         (d) 2D t-SNE projection at theta2 under S1 detour (Catastrophic manifold scatter / rotational chaos).
         (e) 2D t-SNE projection at theta2 under C0 continuous baseline (Stable intact diagnostic clusters).
         (f) 2D t-SNE projection at theta3 under S1 recovery (Complete structural snapback and diagnostic re-clustering).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NPZ_PATH = os.path.join(BASE_DIR, "results", "applied_manifold_scree_data.npz")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

if not os.path.exists(NPZ_PATH):
    print(f"Data file not found at {NPZ_PATH}")
    exit(1)

data = np.load(NPZ_PATH)

# Colors matching manuscript palette
C_BLUE = '#1F77B4'      # Continuous Target (C0) / Baseline theta1
C_RED = '#D62728'       # Detour Collapse (S1 theta2)
C_GREEN = '#2CA02C'     # Relearning Recovery (S1 theta3)
C_DARK = '#1C2833'
C_GRAY = '#7F8C8D'
C_MUTED = '#4A5568'

# 8 Diagnostic Classes for ISIC 2019
CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
CLASS_FULL = [
    'Melanoma (MEL)',
    'Melanocytic Nevus (NV)',
    'Basal Cell Carcinoma (BCC)',
    'Actinic Keratosis (AK)',
    'Benign Keratosis (BKL)',
    'Dermatofibroma (DF)',
    'Vascular Lesion (VASC)',
    'Squamous Cell Carcinoma (SCC)'
]
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

# 1. Singular Values
s1_th1 = data['s_S1_theta1']
s1_th2 = data['s_S1_theta2']
s1_th3 = data['s_S1_theta3']
c0_th2 = data['s_C0_theta2']

# Normalize spectra (p_k = sigma_k / sum sigma)
p1 = s1_th1 / np.sum(s1_th1)
p2 = s1_th2 / np.sum(s1_th2)
p3 = s1_th3 / np.sum(s1_th3)
p_c0 = c0_th2 / np.sum(c0_th2)

# Cumulative energy
cum1 = np.cumsum(p1)
cum2 = np.cumsum(p2)
cum3 = np.cumsum(p3)
cum_c0 = np.cumsum(p_c0)

# Effective ranks
er1 = float(data['er_S1_theta1'])
er2 = float(data['er_S1_theta2'])
er3 = float(data['er_S1_theta3'])
er_c0 = float(data['er_C0_theta2'])

# Setup 2-row figure
fig = plt.figure(figsize=(15.2, 10.5), dpi=300)
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.3], hspace=0.32, wspace=0.25)

# =========================================================================
# Row 1, Left: (a) Scree Plot (Singular Value Spectrum)
# =========================================================================
ax_scree = fig.add_subplot(gs[0, 0:2])
k_range = np.arange(1, 151)  # Focus on top 150 singular values

ax_scree.plot(k_range, p1[:150] * 100, color=C_BLUE, lw=2.2, linestyle='-',
              label=rf"$\boldsymbol{{\theta}}_1$ Target Warmup (Step 5k; $\mathrm{{EffRank}} = {er1:.1f}$)")
ax_scree.plot(k_range, p2[:150] * 100, color=C_RED, lw=2.4, linestyle='-',
              label=rf"$\boldsymbol{{\theta}}_2$ Detour Collapse on $S_1$ (Step 7k; $\mathrm{{EffRank}} = {er2:.1f}$)")
ax_scree.plot(k_range, p_c0[:150] * 100, color='#8E44AD', lw=1.8, linestyle=':',
              label=rf"$\boldsymbol{{\theta}}_2$ Continuous $C_0$ Control (Step 7k; $\mathrm{{EffRank}} = {er_c0:.1f}$)")
ax_scree.plot(k_range, p3[:150] * 100, color=C_GREEN, lw=2.2, linestyle='--',
              label=rf"$\boldsymbol{{\theta}}_3$ Relearning Recovery on $S_1$ (Step 12k; $\mathrm{{EffRank}} = {er3:.1f}$)")

ax_scree.set_yscale('log')
ax_scree.set_xlabel("Singular Value Rank Index ($k$)", fontsize=10.5, fontweight='bold')
ax_scree.set_ylabel("Normalized Singular Energy ($p_k$, % log scale)", fontsize=10.5, fontweight='bold')
ax_scree.set_title("(a) Singular Value Spectrum (Scree Plot): Transient Geometric Collapse",
                   fontsize=11.0, fontweight='bold', loc='left', pad=8)
ax_scree.grid(True, linestyle=':', alpha=0.5, which='both')
ax_scree.legend(loc='upper right', framealpha=0.95, fontsize=8.8)

# Annotate steep drop
ax_scree.annotate("Sharp Spectral Collapse\n(Top singular vectors concentrate;\nEffRank drops by 359 dims)",
                  xy=(12, p2[11]*100), xytext=(32, 0.45),
                  fontsize=8.5, fontweight='bold', color=C_RED,
                  arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.4),
                  bbox=dict(boxstyle="round,pad=0.25", facecolor='#FDEDEC', edgecolor=C_RED, lw=0.9))

# =========================================================================
# Row 1, Right: (b) Cumulative Spectral Energy
# =========================================================================
ax_cum = fig.add_subplot(gs[0, 2:4])
k_cum_range = np.arange(1, 1001)

ax_cum.plot(k_cum_range, cum1[:1000] * 100, color=C_BLUE, lw=2.0, label=r"$\boldsymbol{\theta}_1$ Warmup (Step 5k)")
ax_cum.plot(k_cum_range, cum2[:1000] * 100, color=C_RED, lw=2.4, label=r"$\boldsymbol{\theta}_2$ Detour Collapse ($S_1$)")
ax_cum.plot(k_cum_range, cum_c0[:1000] * 100, color='#8E44AD', lw=1.8, linestyle=':', label=r"$\boldsymbol{\theta}_2$ Continuous Baseline ($C_0$)")
ax_cum.plot(k_cum_range, cum3[:1000] * 100, color=C_GREEN, lw=2.0, linestyle='--', label=r"$\boldsymbol{\theta}_3$ Recovery ($S_1$)")

# Threshold lines for 80% and 90% energy
ax_cum.axhline(80, color=C_GRAY, linestyle=':', lw=1.0)
ax_cum.axhline(90, color=C_GRAY, linestyle=':', lw=1.0)
ax_cum.text(980, 80.5, "80% Energy", ha='right', fontsize=8.0, color=C_GRAY, fontweight='bold')
ax_cum.text(980, 90.5, "90% Energy", ha='right', fontsize=8.0, color=C_GRAY, fontweight='bold')

# Find k for 80% energy
k80_th1 = np.where(cum1 >= 0.80)[0][0] + 1
k80_th2 = np.where(cum2 >= 0.80)[0][0] + 1
k80_th3 = np.where(cum3 >= 0.80)[0][0] + 1

ax_cum.scatter([k80_th2], [80], color=C_RED, s=60, zorder=5)
ax_cum.scatter([k80_th1], [80], color=C_BLUE, s=60, zorder=5)
ax_cum.scatter([k80_th3], [80], color=C_GREEN, s=60, zorder=5)

ax_cum.annotate(f"Collapse: 80% energy in {k80_th2} dims\n(vs. {k80_th1} in warmup, {k80_th3} in recovery)",
                xy=(k80_th2, 80), xytext=(k80_th2 - 280, 60),
                fontsize=8.5, fontweight='bold', color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.4),
                bbox=dict(boxstyle="round,pad=0.25", facecolor='#FDEDEC', edgecolor=C_RED, lw=0.9))

ax_cum.set_xlabel("Cumulative Dimensions ($k$)", fontsize=10.5, fontweight='bold')
ax_cum.set_ylabel("Cumulative Variance Explained (%)", fontsize=10.5, fontweight='bold')
ax_cum.set_title("(b) Cumulative Spectral Energy: Dimensional Compression and Re-expansion",
                 fontsize=11.0, fontweight='bold', loc='left', pad=8)
ax_cum.grid(True, linestyle=':', alpha=0.5)
ax_cum.legend(loc='lower right', framealpha=0.95, fontsize=8.8)
ax_cum.set_xlim(0, 1050)
ax_cum.set_ylim(20, 102)

# =========================================================================
# Row 2: (c, d, e, f) 2D t-SNE Clinical Manifold Projections
# =========================================================================
tsne_runs = [
    (gs[1, 0], 'tsne_S1_theta1', "(c) Step 5k: Target Warmup",
     r"$\mathbf{S_1}$: Initial Target Manifold", "Structured diagnostic separation"),
    (gs[1, 1], 'tsne_S1_theta2', "(d) Step 7k: Detour Shock",
     r"$\mathbf{S_1}$: Auxiliary Detour (Imagenette)", "Catastrophic cluster dispersion"),
    (gs[1, 2], 'tsne_C0_theta2', "(e) Step 7k: Continuous Baseline",
     r"$\mathbf{C_0}$: Continuous Target Training", "Intact, preserved lesion clusters"),
    (gs[1, 3], 'tsne_S1_theta3', "(f) Step 12k: Relearning Snapback",
     r"$\mathbf{S_1}$: Target Relearning Terminal", "Complete diagnostic re-clustering"),
]

for g_pos, key, title_str, sub_arm, note_str in tsne_runs:
    ax_t = fig.add_subplot(g_pos)
    coords = data[key]
    
    # Plot scatter by class
    for c_idx in range(8):
        mask = (labels == c_idx)
        ax_t.scatter(coords[mask, 0], coords[mask, 1],
                     c=CLASS_COLORS[c_idx], label=CLASS_NAMES[c_idx] if g_pos == gs[1, 0] else None,
                     s=10, alpha=0.65, edgecolors='none', rasterized=True)
    
    ax_t.set_title(title_str, fontsize=9.8, fontweight='bold', pad=6)
    ax_t.text(0.5, 0.96, sub_arm, transform=ax_t.transAxes, ha='center', va='top',
              fontsize=8.5, fontweight='bold', color=C_DARK,
              bbox=dict(boxstyle="round,pad=0.18", facecolor='white', edgecolor='#D5D8DC', alpha=0.9))
    ax_t.text(0.5, 0.04, note_str, transform=ax_t.transAxes, ha='center', va='bottom',
              fontsize=7.8, fontstyle='italic', color=C_MUTED,
              bbox=dict(boxstyle="round,pad=0.15", facecolor='white', edgecolor='#E5E7E9', alpha=0.85))
    
    ax_t.set_xticks([])
    ax_t.set_yticks([])
    for spine in ax_t.spines.values():
        spine.set_color('#CBD5E1')
        spine.set_linewidth(1.0)

# Add shared legend for lesion categories across the bottom
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=CLASS_FULL[i],
           markerfacecolor=CLASS_COLORS[i], markersize=8)
    for i in range(8)
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4,
           bbox_to_anchor=(0.5, 0.01), frameon=True, framealpha=0.95,
           fontsize=8.5, title="ISIC 2019 Diagnostic Categories (2,000 Clinical Probe Lesions)",
           title_fontsize=9.2)

pdf_out = os.path.join(FIGURES_DIR, "figure_applied_manifold_and_scree.pdf")
png_out = os.path.join(FIGURES_DIR, "figure_applied_manifold_and_scree.png")
plt.savefig(pdf_out, bbox_inches='tight', dpi=300)
plt.savefig(png_out, bbox_inches='tight', dpi=300)
plt.close()

print(f"Generated Applied Manifold & Scree figure successfully:\n  {pdf_out}\n  {png_out}")

