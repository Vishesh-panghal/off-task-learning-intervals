#!/usr/bin/env python3
"""
Generate publication-grade Longitudinal Representation Dynamics Figure
for Pattern Analysis and Applications (PAA, Springer Nature).

Visualizes Table 5 representation trajectory data across:
- Milestone 1: Step 15,000 (Shared Checkpoint theta_1)
- Milestone 2: Step 20,000 (Post-Interruption End of Phase 2)
- Milestone 3: Step 35,000 (Terminal Target Relearning End of Phase 3)

Layout:
- Panel (a): Target Decodability (Matched Linear Probe Accuracy, %)
- Panel (b): Geometric Dimensionality (Effective Rank via SVD, EffRank)

Design features:
- High-contrast PAA palette: C0 (Blue), C3 (Purple), S1 (Orange)
- Synchronized x-axis [13.8 to 36.5] for Panels (a) and (b)
- Shaded Phase 2 (Rose) and Phase 3 (Green) background bands with pill badges
- Clean non-overlapping orthogonal callouts with zero curve intersections
- Unified horizontal legend centered below panels (a) and (b)
- Dual export: 300 DPI PNG + vector PDF
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Canvas styling
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../figures"))
os.makedirs(FIGURES_DIR, exist_ok=True)

# Color Palette matching Figure 1 and PAA specs
C_BLUE = '#2980B9'      # C0 Continuous Target Baseline
C_PURPLE = '#8E44AD'    # C3 Target Reset Control
C_ORANGE = '#D35400'    # S1 Auxiliary STL-10 Interval
C_GREEN = '#27AE60'     # Positive contrast / Success / Recovery
C_ROSE = '#C0392B'      # Negative contrast / Reset / Collapse
C_DARK = '#1C2833'

# Milestone data (Updates in thousands: 15k, 20k, 35k) from Table 5
steps = np.array([15.0, 20.0, 35.0])

# 1. Matched Linear Probe Accuracy (%)
probe_c0 = np.array([60.07, 61.85, 63.01])
probe_c0_sd = np.array([0.81, 0.65, 0.58])

probe_c3 = np.array([60.07, 56.12, 62.14])
probe_c3_sd = np.array([0.81, 1.14, 0.69])

probe_s1 = np.array([60.07, 20.65, 61.96])
probe_s1_sd = np.array([0.81, 2.87, 0.74])

# 2. Effective Rank (EffRank)
eff_c0 = np.array([178.04, 171.20, 162.45])
eff_c0_sd = np.array([2.15, 2.45, 2.85])

eff_c3 = np.array([178.04, 148.31, 157.80])
eff_c3_sd = np.array([2.15, 3.22, 2.90])

eff_s1 = np.array([178.04, 124.50, 153.69])
eff_s1_sd = np.array([2.15, 4.82, 3.10])

# Canvas Setup: 2 Panels (a & b)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.5), dpi=300)

# Helper: Phase shading synchronized across [13.8, 36.5]
def add_phase_shading(ax, y_min, y_max):
    # Phase 2: 15k to 20k (Interruption)
    ax.axvspan(15.0, 20.0, facecolor='#FDEDEC', alpha=0.45, zorder=0)
    # Phase 3: 20k to 35k (Relearning)
    ax.axvspan(20.0, 35.0, facecolor='#EAFAF1', alpha=0.35, zorder=0)
    
    # Milestone Boundary dashed lines
    ax.axvline(15.0, color='#BDC3C7', linestyle='--', lw=1.2, zorder=1)
    ax.axvline(20.0, color='#BDC3C7', linestyle='--', lw=1.2, zorder=1)
    ax.axvline(35.0, color='#BDC3C7', linestyle='--', lw=1.2, zorder=1)
    
    # Phase Header Badges positioned at top
    y_badge = y_max - (y_max - y_min) * 0.04
    ax.text(17.5, y_badge, "Phase 2: Interruption (15k–20k)", ha='center', va='top', fontsize=7.6, fontweight='bold', color='#922B21',
            bbox=dict(boxstyle="round,pad=0.25", facecolor='#FADBD8', edgecolor='#E6B0AA', lw=0.8, alpha=0.9))
    ax.text(27.5, y_badge, "Phase 3: Target Relearning (20k–35k)", ha='center', va='top', fontsize=7.6, fontweight='bold', color='#1E8449',
            bbox=dict(boxstyle="round,pad=0.25", facecolor='#D4EFDF', edgecolor='#A9DFBF', lw=0.8, alpha=0.9))

# =========================================================================
# PANEL A: Target Decodability (Matched Linear Probe Accuracy)
# =========================================================================
add_phase_shading(ax1, 10.0, 76.0)

line_c0, = ax1.plot(steps, probe_c0, color=C_BLUE, lw=2.4, marker='o', markersize=6.5, label=r"Continuous Target Baseline ($C_0$)", zorder=5)
ax1.fill_between(steps, probe_c0 - probe_c0_sd, probe_c0 + probe_c0_sd, color=C_BLUE, alpha=0.15, zorder=3)

line_c3, = ax1.plot(steps, probe_c3, color=C_PURPLE, lw=2.2, marker='s', markersize=6.0, label=r"On-Target Reset Control ($C_3$)", zorder=5)
ax1.fill_between(steps, probe_c3 - probe_c3_sd, probe_c3 + probe_c3_sd, color=C_PURPLE, alpha=0.15, zorder=3)

line_s1, = ax1.plot(steps, probe_s1, color=C_ORANGE, lw=2.6, marker='^', markersize=7.0, label=r"Auxiliary STL-10 Interval ($S_1$)", zorder=6)
ax1.fill_between(steps, probe_s1 - probe_s1_sd, probe_s1 + probe_s1_sd, color=C_ORANGE, alpha=0.20, zorder=4)

# Milestone 1: Shared Checkpoint theta_1 badge (centered directly above marker with down-arrow)
ax1.plot(15.0, 60.07, 'o', color=C_DARK, markersize=7.5, zorder=7)
ax1.annotate(r"$\theta_1 = 60.07\%$", xy=(15.0, 60.07), xytext=(15.0, 65.0),
             arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.1),
             fontsize=7.6, fontweight='bold', color=C_DARK, ha='center',
             bbox=dict(boxstyle="round,pad=0.25", facecolor='#FFFFFF', edgecolor=C_DARK, lw=0.9, alpha=0.95))

# Callout 1: Target Amnesia at 20k (Placed in open space under slope at x=16.4, y=20.65 with horizontal arrow)
ax1.annotate("Target Amnesia\nDrop to 20.65%\n" + r"($\Delta = -39.42$ pp)",
             xy=(20.0, 20.65), xytext=(16.4, 20.65),
             arrowprops=dict(arrowstyle="->", color=C_ROSE, lw=1.2),
             fontsize=7.4, fontweight='bold', color=C_ROSE, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFFFFF', edgecolor=C_ROSE, lw=1.0, alpha=0.95))

# Callout 2: Plastic Recovery Rebound (Placed in open space below rising slope at x=28.0, y=22.0, pointing UP to slope)
ax1.annotate("Plastic Recovery Rebound\n" + r"$+41.31$ pp Rebound" + "\n(Reaches 61.96% Parity)",
             xy=(28.0, 42.68), xytext=(28.0, 22.0),
             arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.2),
             fontsize=7.4, fontweight='bold', color=C_GREEN, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFFFFF', edgecolor=C_GREEN, lw=1.0, alpha=0.95))

ax1.set_xlim(13.8, 36.5)
ax1.set_ylim(10.0, 76.0)
ax1.set_xticks([15, 20, 25, 30, 35])
ax1.set_xticklabels(["15k\n(" + r"$\theta_1$" + ")", "20k\n(End P2)", "25k", "30k", "35k\n(Terminal)"], fontsize=8.0)
ax1.set_xlabel("Training Updates (Thousands)", fontsize=9.0, fontweight='bold', color=C_DARK)
ax1.set_ylabel("Matched Linear Probe Accuracy (%)", fontsize=9.0, fontweight='bold', color=C_DARK)
ax1.set_title(r"(a) Target Decodability Trajectory ($W_{\rm probe} \in \mathbb{R}^{100 \times 512}$)", fontsize=9.8, fontweight='bold', color=C_DARK, pad=10)
ax1.grid(True, linestyle=':', alpha=0.6, color='#BDC3C7')

# =========================================================================
# PANEL B: Geometric Dimensionality (Effective Rank via SVD)
# =========================================================================
add_phase_shading(ax2, 105.0, 208.0)

ax2.plot(steps, eff_c0, color=C_BLUE, lw=2.4, marker='o', markersize=6.5, zorder=5)
ax2.fill_between(steps, eff_c0 - eff_c0_sd, eff_c0 + eff_c0_sd, color=C_BLUE, alpha=0.15, zorder=3)

ax2.plot(steps, eff_c3, color=C_PURPLE, lw=2.2, marker='s', markersize=6.0, zorder=5)
ax2.fill_between(steps, eff_c3 - eff_c3_sd, eff_c3 + eff_c3_sd, color=C_PURPLE, alpha=0.15, zorder=3)

ax2.plot(steps, eff_s1, color=C_ORANGE, lw=2.6, marker='^', markersize=7.0, zorder=6)
ax2.fill_between(steps, eff_s1 - eff_s1_sd, eff_s1 + eff_s1_sd, color=C_ORANGE, alpha=0.20, zorder=4)

# Milestone 1: Shared Checkpoint theta_1 badge
ax2.plot(15.0, 178.04, 'o', color=C_DARK, markersize=7.5, zorder=7)
ax2.annotate(r"$\theta_1 = 178.04$", xy=(15.0, 178.04), xytext=(15.0, 187.0),
             arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.1),
             fontsize=7.6, fontweight='bold', color=C_DARK, ha='center',
             bbox=dict(boxstyle="round,pad=0.25", facecolor='#FFFFFF', edgecolor=C_DARK, lw=0.9, alpha=0.95))

# Callout 1: Subspace Collapse at 20k (Placed in open space under slope at x=16.4, y=124.5 with horizontal arrow)
ax2.annotate("Subspace Collapse\n" + r"${\rm EffRank} \to 124.50$" + "\n(10-Class Alignment)",
             xy=(20.0, 124.50), xytext=(16.4, 124.50),
             arrowprops=dict(arrowstyle="->", color=C_ROSE, lw=1.2),
             fontsize=7.4, fontweight='bold', color=C_ROSE, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFFFFF', edgecolor=C_ROSE, lw=1.0, alpha=0.95))

# Callout 2: Subspace Re-expansion (Placed in open space below rising slope at x=28.0, y=120.0, pointing UP to slope)
ax2.annotate("Subspace Re-expansion\n" + r"${\rm EffRank} \to 153.69$" + "\n(Capacity Recovers)",
             xy=(28.0, 141.04), xytext=(28.0, 120.0),
             arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.2),
             fontsize=7.4, fontweight='bold', color=C_GREEN, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFFFFF', edgecolor=C_GREEN, lw=1.0, alpha=0.95))

ax2.set_xlim(13.8, 36.5)
ax2.set_ylim(105.0, 208.0)
ax2.set_xticks([15, 20, 25, 30, 35])
ax2.set_xticklabels(["15k\n(" + r"$\theta_1$" + ")", "20k\n(End P2)", "25k", "30k", "35k\n(Terminal)"], fontsize=8.0)
ax2.set_xlabel("Training Updates (Thousands)", fontsize=9.0, fontweight='bold', color=C_DARK)
ax2.set_ylabel(r"Effective Rank ($\mathrm{EffRank}(Z) \in [1, 512]$)", fontsize=9.0, fontweight='bold', color=C_DARK)
ax2.set_title(r"(b) Representation Dimensionality ($Z = U \Sigma V^\top$)", fontsize=9.8, fontweight='bold', color=C_DARK, pad=10)
ax2.grid(True, linestyle=':', alpha=0.6, color='#BDC3C7')

# Unified horizontal legend centered below panels (a) and (b)
fig.legend([line_c0, line_c3, line_s1],
           [r"Continuous Target Baseline ($C_0$)", r"On-Target Reset Control ($C_3$)", r"Auxiliary STL-10 Interval ($S_1$)"],
           loc='lower center', bbox_to_anchor=(0.50, 0.02), ncol=3, frameon=True, framealpha=0.95,
           fontsize=8.2, edgecolor='#BDC3C7')

# Save figures
png_out = os.path.join(FIGURES_DIR, "figure_longitudinal_representation_dynamics.png")
pdf_out = os.path.join(FIGURES_DIR, "figure_longitudinal_representation_dynamics.pdf")

plt.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.18, wspace=0.22)
plt.savefig(png_out, dpi=300)
plt.savefig(pdf_out)
plt.close()

print(f"✓ Saved Refined Longitudinal Dynamics PNG (2 panels): {png_out} ({os.path.getsize(png_out)/1024:.1f} KB)")
print(f"✓ Saved Refined Longitudinal Dynamics PDF (2 panels): {pdf_out} ({os.path.getsize(pdf_out)/1024:.1f} KB)")
