#!/usr/bin/env python3
"""
Generate publication-grade Figure 4: Longitudinal Representation Dynamics.
Designed at native 6.5-inch width:
- Panel (a): Latent Dimensionality (Effective Rank).
- Panel (b): Representation Drift (Linear CKA Drift relative to theta_1).
- Panel (c): Target Decodability (Matched Frozen Linear Probe Accuracy).

Typography:
- Panel titles: 9.5 pt bold
- Axis labels: 8.8 pt bold
- Tick labels: 8.0 pt bold
- Value labels & legend: 7.8 pt
"""

import os
import numpy as np
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

C_BLUE = '#1F77B4'      # C0 Continuous
C_GREEN = '#2CA02C'     # C3 Procedural Reset
C_RED = '#D62728'       # S1 Auxiliary Detour
C_DARK = '#0F172A'
C_MUTED = '#475569'

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

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.5, 1.25), dpi=300,
                                   gridspec_kw={'wspace': 0.34})

# Helper to draw Phase 2 band
def setup_axis(ax, title, ylabel, ylim):
    ax.axvspan(15.0, 20.0, color='#FADBD8', alpha=0.35, zorder=0)
    ax.set_xlim(13.5, 36.5)
    ax.set_ylim(ylim)
    ax.set_xticks([15.0, 20.0, 35.0])
    ax.set_xticklabels([r"$\boldsymbol{\theta}_1$" + "\n(15k)", r"$\boldsymbol{\theta}_2$" + "\n(20k)", r"$\boldsymbol{\theta}_3$" + "\n(35k)"],
                       fontsize=7.8, fontweight='bold')
    ax.set_title(title, fontsize=9.2, fontweight='bold', color=C_DARK, pad=6)
    ax.set_ylabel(ylabel, fontsize=8.5, fontweight='bold', color=C_DARK)
    ax.grid(True, linestyle=':', alpha=0.4, color='#CBD5E1')

# --- Panel (a): Effective Rank ---
setup_axis(ax1, "(a) Effective Dimensionality", r"Effective Rank ($\mathrm{EffRank}$)", (115, 175))
ax1.plot(steps, eff_c0, color=C_BLUE, lw=1.8, marker='o', markersize=4.5, label=r"$C_0$ Continuous", zorder=4)
ax1.plot(steps, eff_c3, color=C_GREEN, lw=1.8, linestyle='--', marker='s', markersize=4.5, label=r"$C_3$ Reset Ctrl", zorder=4)
ax1.plot(steps, eff_s1, color=C_RED, lw=2.0, marker='D', markersize=4.5, label=r"$S_1$ Auxiliary", zorder=5)

ax1.text(20.0, 122.9 - 5.5, "122.9\n(-39.8)", ha='center', va='top', fontsize=7.2, fontweight='bold', color=C_RED)
ax1.text(35.0, 152.0 - 5.5, "152.0", ha='center', va='top', fontsize=7.2, fontweight='bold', color=C_RED)
ax1.text(20.0, 168.0, "165.4", ha='center', va='bottom', fontsize=7.2, fontweight='bold', color=C_GREEN)

# --- Panel (b): CKA Drift ---
setup_axis(ax2, "(b) Representation Drift", r"Linear CKA Drift ($D_{\mathrm{CKA}}$)", (-0.05, 0.78))
ax2.plot(steps, cka_c0, color=C_BLUE, lw=1.8, marker='o', markersize=4.5, zorder=4)
ax2.plot(steps, cka_c3, color=C_GREEN, lw=1.8, linestyle='--', marker='s', markersize=4.5, zorder=4)
ax2.plot(steps, cka_s1, color=C_RED, lw=2.0, marker='D', markersize=4.5, zorder=5)

ax2.text(20.0, 0.655 + 0.04, "0.655\n(Peak)", ha='center', va='bottom', fontsize=7.2, fontweight='bold', color=C_RED)
ax2.text(35.0, 0.461 + 0.04, "0.461", ha='center', va='bottom', fontsize=7.2, fontweight='bold', color=C_RED)
ax2.text(35.0, 0.320 - 0.05, "0.320", ha='center', va='top', fontsize=7.2, fontweight='bold', color=C_BLUE)

# --- Panel (c): Linear Probe Accuracy ---
setup_axis(ax3, "(c) Target Decodability", r"Frozen Probe Accuracy (%)", (15, 70))
ax3.plot(steps, prb_c0, color=C_BLUE, lw=1.8, marker='o', markersize=4.5, zorder=4)
ax3.plot(steps, prb_c3, color=C_GREEN, lw=1.8, linestyle='--', marker='s', markersize=4.5, zorder=4)
ax3.plot(steps, prb_s1, color=C_RED, lw=2.0, marker='D', markersize=4.5, zorder=5)

ax3.text(20.0, 21.9 - 4.5, "21.9%\n(-30.9 pp)", ha='center', va='top', fontsize=7.2, fontweight='bold', color=C_RED)
ax3.text(35.0, 62.7 - 4.5, "62.7%", ha='center', va='top', fontsize=7.2, fontweight='bold', color=C_RED)
ax3.text(35.0, 63.0 + 3.0, "63.0%", ha='center', va='bottom', fontsize=7.2, fontweight='bold', color=C_BLUE)

# Unified Horizontal Legend across the top
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06),
           ncol=3, frameon=True, framealpha=0.95, fontsize=7.8, edgecolor='#CBD5E1')

plt.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.22)

pdf_out = os.path.join(FIGURES_DIR, "figure_longitudinal_representation_dynamics.pdf")
png_out = os.path.join(FIGURES_DIR, "figure_longitudinal_representation_dynamics.png")
fig.savefig(pdf_out, bbox_inches='tight', pad_inches=0.02)
fig.savefig(png_out, bbox_inches='tight', pad_inches=0.02, dpi=300)

for d in [CLEAN_FIGURES_DIR]:
    fig.savefig(os.path.join(d, "figure_longitudinal_representation_dynamics.pdf"), bbox_inches='tight', pad_inches=0.02)
    fig.savefig(os.path.join(d, "figure_longitudinal_representation_dynamics.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)

plt.close(fig)
print(f"Figure 4 successfully generated: {pdf_out}")
