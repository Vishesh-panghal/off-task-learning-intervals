#!/usr/bin/env python3
"""
Generate publication-grade Figure 4: Longitudinal Representation Dynamics.
Authored in authentic Seaborn statistical style (sns.set_theme(style="ticks")):
- Panel (a): Latent Dimensionality (Effective Rank across milestones).
- Panel (b): Representation Drift (Linear CKA Drift relative to theta_1).
- Panel (c): Target Decodability (Matched Frozen Linear Probe Accuracy).

Typography & Design:
- Genuine Seaborn Deep palette: #4c72b0 (C0 Blue), #55a868 (C3 Green), #c44e52 (S1 Coral).
- Standard academic hierarchy: semibold titles, medium axis labels, normal weight ticks/values.
- Soft neutral Phase 2 shading (#f1f5f9) with dashed boundary markers.
- Clean unboxed value annotations and unified borderless top legend.
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
C_BLUE = palette[0]    # #4c72b0 (C0 Continuous)
C_GREEN = palette[2]   # #55a868 (C3 Procedural Reset)
C_CORAL = palette[3]   # #c44e52 (S1 Auxiliary Detour)
C_DARK = '#1e293b'     # Slate 800 for clean text
C_GRAY = '#64748b'     # Slate 500

steps = np.array([15.0, 20.0, 35.0])

# Effective Rank (Table 4)
eff_c0 = np.array([162.7, 161.7, 160.7])
eff_c3 = np.array([162.7, 165.4, 160.8])
eff_s1 = np.array([162.7, 122.9, 152.0])

# CKA Drift from theta_1 (Table 4)
cka_c0 = np.array([0.000, 0.160, 0.320])
cka_c3 = np.array([0.000, 0.220, 0.334])
cka_s1 = np.array([0.000, 0.655, 0.461])

# Linear Probe Accuracy (Table 4)
prb_c0 = np.array([52.8, 58.0, 63.0])
prb_c3 = np.array([52.8, 60.9, 63.0])
prb_s1 = np.array([52.8, 21.9, 62.7])

# Set up figure at native 6.5 in width (TMLR full-width) and 2.15 in height
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.5, 2.15), dpi=300,
                                   gridspec_kw={'wspace': 0.38})

def setup_axis(ax, title, ylabel, ylim, yticks, yticklabels):
    # Soft neutral Phase 2 shading (subtle slate tint)
    ax.axvspan(15.0, 20.0, color='#f1f5f9', alpha=0.95, zorder=0)
    ax.axvline(15.0, color='#cbd5e1', linestyle=':', lw=0.8, zorder=1)
    ax.axvline(20.0, color='#cbd5e1', linestyle=':', lw=0.8, zorder=1)
    
    ax.set_xlim(13.2, 36.8)
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=7.6)
    ax.set_xticks([15.0, 20.0, 35.0])
    ax.set_xticklabels([r"$\boldsymbol{\theta}_1$" + "\n(15k)", r"$\boldsymbol{\theta}_2$" + "\n(20k)", r"$\boldsymbol{\theta}_3$" + "\n(35k)"],
                       fontsize=7.6)
    ax.set_title(title, fontsize=8.8, fontweight='semibold', color=C_DARK, pad=18)
    ax.set_ylabel(ylabel, fontsize=8.2, color=C_DARK)
    ax.grid(axis='y', linestyle='--', alpha=0.35, color='#cbd5e1')
    sns.despine(ax=ax, trim=False)

# --- Panel (a): Effective Rank ---
setup_axis(ax1, "(a) Latent Dimensionality", r"Effective Rank ($\mathrm{EffRank}$)", (112, 178),
           [120, 140, 160], ["120", "140", "160"])
ax1.plot(steps, eff_c0, color=C_BLUE, lw=2.0, marker='o', markersize=5.0, label=r"$C_0$ Continuous", zorder=4)
ax1.plot(steps, eff_c3, color=C_GREEN, lw=2.0, linestyle='--', marker='s', markersize=5.0, label=r"$C_3$ Reset Ctrl", zorder=4)
ax1.plot(steps, eff_s1, color=C_CORAL, lw=2.2, marker='^', markersize=5.5, label=r"$S_1$ Auxiliary", zorder=5)

ax1.text(20.0, 122.9 - 3.5, "122.9\n(-39.8)", ha='center', va='top', fontsize=7.2, color=C_CORAL, fontweight='medium')
ax1.text(35.0, 152.0 - 3.5, "152.0", ha='center', va='top', fontsize=7.2, color=C_CORAL, fontweight='medium')
ax1.text(20.0, 165.4 + 2.5, "165.4", ha='center', va='bottom', fontsize=7.2, color=C_GREEN, fontweight='medium')
ax1.text(35.0, 160.7 + 2.5, "160.7", ha='center', va='bottom', fontsize=7.2, color=C_BLUE, fontweight='medium')

# --- Panel (b): CKA Drift ---
setup_axis(ax2, "(b) Representation Drift", r"Linear CKA Drift ($D_{\mathrm{CKA}}$)", (-0.05, 0.78),
           [0.0, 0.2, 0.4, 0.6], ["0.0", "0.2", "0.4", "0.6"])
ax2.plot(steps, cka_c0, color=C_BLUE, lw=2.0, marker='o', markersize=5.0, zorder=4)
ax2.plot(steps, cka_c3, color=C_GREEN, lw=2.0, linestyle='--', marker='s', markersize=5.0, zorder=4)
ax2.plot(steps, cka_s1, color=C_CORAL, lw=2.2, marker='^', markersize=5.5, zorder=5)

ax2.text(20.0, 0.655 + 0.03, "0.655\n(Peak)", ha='center', va='bottom', fontsize=7.2, color=C_CORAL, fontweight='medium')
ax2.text(35.0, 0.461 + 0.03, "0.461", ha='center', va='bottom', fontsize=7.2, color=C_CORAL, fontweight='medium')
ax2.text(35.0, 0.320 - 0.04, "0.320", ha='center', va='top', fontsize=7.2, color=C_BLUE, fontweight='medium')

# --- Panel (c): Linear Probe Accuracy ---
setup_axis(ax3, "(c) Target Decodability", r"Frozen Probe Accuracy (%)", (12, 72),
           [20, 30, 40, 50, 60, 70], ["20%", "30%", "40%", "50%", "60%", "70%"])
ax3.plot(steps, prb_c0, color=C_BLUE, lw=2.0, marker='o', markersize=5.0, zorder=4)
ax3.plot(steps, prb_c3, color=C_GREEN, lw=2.0, linestyle='--', marker='s', markersize=5.0, zorder=4)
ax3.plot(steps, prb_s1, color=C_CORAL, lw=2.2, marker='^', markersize=5.5, zorder=5)

ax3.text(20.0, 21.9 - 3.0, "21.9%\n(-30.9 pp)", ha='center', va='top', fontsize=7.2, color=C_CORAL, fontweight='medium')
ax3.text(35.0, 63.0 + 2.5, "63.0%", ha='center', va='bottom', fontsize=7.2, color=C_BLUE, fontweight='medium')
ax3.text(35.0, 62.7 - 5.5, "62.7%", ha='center', va='top', fontsize=7.2, color=C_CORAL, fontweight='medium')

# Unified clean borderless legend placed outside at the top
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.98),
           ncol=3, frameon=False, fontsize=7.6)

plt.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.18)

pdf_out = os.path.join(FIGURES_DIR, "figure_longitudinal_representation_dynamics.pdf")
png_out = os.path.join(FIGURES_DIR, "figure_longitudinal_representation_dynamics.png")
fig.savefig(pdf_out, bbox_inches='tight', pad_inches=0.04)
fig.savefig(png_out, bbox_inches='tight', pad_inches=0.04, dpi=300)

for d in [CLEAN_FIGURES_DIR]:
    fig.savefig(os.path.join(d, "figure_longitudinal_representation_dynamics.pdf"), bbox_inches='tight', pad_inches=0.04)
    fig.savefig(os.path.join(d, "figure_longitudinal_representation_dynamics.png"), bbox_inches='tight', pad_inches=0.04, dpi=300)

plt.close(fig)
print(f"Figure 4 successfully generated: {pdf_out}")
