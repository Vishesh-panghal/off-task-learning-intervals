#!/usr/bin/env python3
"""
Generate publication-grade Figure 1: Controlled Experimental Protocol and Schedule Architecture.
Displays both Regime 1 (From-Scratch CIFAR-100 / STL-10, 35k updates) and
Regime 2 (Applied Transfer ISIC 2019 / Imagenette, 12k updates) as clean timelines.
Designed at native 6.5-inch column width with generous padding and 1:1 publication typography:
- Titles: 9.5 pt bold
- Milestone markers: 9.0 pt bold
- Phase labels: 8.0 pt bold
- Arm labels: 7.5 pt bold
- Annotations: 7.0 pt
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Publication typography matching TMLR
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "sans-serif"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "cm"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
CLEAN_FIGURES_DIR = os.path.join(BASE_DIR, "clean_repo", "figures")
for d in [FIGURES_DIR, CLEAN_FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# Harmonized Palette
C_BLUE = "#1F77B4"      # C0 Continuous Baseline
C_GREEN = "#2CA02C"     # C3 Procedural Reset Control
C_RED = "#D62728"       # S1 Auxiliary Detour
C_DARK = "#0F172A"      # Slate 900
C_TEXT = "#1E293B"      # Slate 800
C_MUTED = "#475569"     # Slate 600
C_LINE = "#64748B"      # Slate 500

fig, ax = plt.subplots(figsize=(6.5, 2.25), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

def draw_regime(ax, y_base, box_h, title, budget_tag, task_a, task_b, arch_tag,
                u1, ub, u2, st_1, st_2):
    # Outer card with generous inner padding
    card = patches.FancyBboxPatch(
        (1.2, y_base), 97.6, box_h,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=0.9, zorder=1
    )
    ax.add_patch(card)

    # Title header line with generous top padding
    ax.text(3.4, y_base + box_h - 6.0, title, fontsize=9.2, fontweight="bold", color=C_DARK, va="center", zorder=3)
    ax.text(96.6, y_base + box_h - 6.0, f"Budget: {budget_tag}  ({arch_tag})", ha="right", va="center",
            fontsize=7.8, fontweight="bold", color=C_MUTED, zorder=3)

    # Horizontal coordinate anchors with generous spacing
    x_p1_s, x_p1_e = 3.4, 21.0
    x_th1 = 25.2
    x_fork = 28.8
    x_p2_s, x_p2_e = 33.0, 69.2
    x_conv = 72.8
    x_th2 = 76.5
    x_p3_s, x_p3_e = 79.8, 94.6
    x_th3 = 96.8

    y_mid = y_base + (box_h * 0.41)
    y_c0 = y_mid + 7.5
    y_c3 = y_mid
    y_s1 = y_mid - 7.5

    # Phase 1 card with increased vertical height and padding
    p1 = patches.FancyBboxPatch((x_p1_s, y_mid - 6.5), (x_p1_e - x_p1_s), 13.0,
                               boxstyle="round,pad=0.2,rounding_size=0.8",
                               facecolor="#FFFFFF", edgecolor="#94A3B8", lw=0.8, zorder=2)
    ax.add_patch(p1)
    ax.text((x_p1_s + x_p1_e)/2.0, y_mid + 2.5, f"Phase 1: {u1}", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color=C_DARK, zorder=3)
    ax.text((x_p1_s + x_p1_e)/2.0, y_mid - 2.5, f"{task_a} (Head $h_A$)", ha="center", va="center",
            fontsize=7.2, color=C_MUTED, zorder=3)

    # Line to theta1
    ax.plot([x_p1_e, x_th1], [y_mid, y_mid], color=C_LINE, lw=1.1, zorder=2)
    ax.scatter(x_th1, y_mid, s=36, marker="D", color=C_DARK, edgecolor="white", lw=0.9, zorder=5)
    # Generous padding above and below milestone diamonds
    ax.text(x_th1, y_mid + 4.8, r"$\boldsymbol{\theta}_1$", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color=C_DARK, zorder=6)
    ax.text(x_th1, y_mid - 4.8, f"Step {st_1}", ha="center", va="top", fontsize=6.8, color=C_MUTED, zorder=6)

    # Line to fork
    ax.plot([x_th1, x_fork], [y_mid, y_mid], color=C_LINE, lw=1.1, zorder=2)
    ax.scatter(x_fork, y_mid, s=12, color=C_LINE, zorder=4)

    # Branching lines
    ax.plot([x_fork, x_p2_s], [y_mid, y_c0], color=C_BLUE, lw=1.2, zorder=3)
    ax.plot([x_fork, x_p2_s], [y_mid, y_c3], color=C_GREEN, lw=1.2, zorder=3)
    ax.plot([x_fork, x_p2_s], [y_mid, y_s1], color=C_RED, lw=1.2, zorder=3)

    # Phase 2 arms with increased height (5.2) and generous internal padding
    arm_h = 5.2
    r_c0 = patches.FancyBboxPatch((x_p2_s, y_c0 - arm_h/2.0), (x_p2_e - x_p2_s), arm_h,
                                 boxstyle="round,pad=0.2,rounding_size=0.6",
                                 facecolor="#EBF5FB", edgecolor=C_BLUE, lw=0.9, zorder=3)
    ax.add_patch(r_c0)
    ax.text(x_p2_s + 1.8, y_c0, r"$\mathbf{C_0}$: Continuous Target", va="center", fontsize=7.4, fontweight="bold", color=C_BLUE, zorder=4)
    ax.text(x_p2_e - 1.8, y_c0, r"Head $h_A$, cont $\mathbf{v}$", ha="right", va="center", fontsize=7.0, color=C_BLUE, zorder=4)

    r_c3 = patches.FancyBboxPatch((x_p2_s, y_c3 - arm_h/2.0), (x_p2_e - x_p2_s), arm_h,
                                 boxstyle="round,pad=0.2,rounding_size=0.6",
                                 facecolor="#EAFAF1", edgecolor=C_GREEN, lw=0.9, zorder=3)
    ax.add_patch(r_c3)
    ax.text(x_p2_s + 1.8, y_c3, r"$\mathbf{C_3}$: Procedural Reset", va="center", fontsize=7.4, fontweight="bold", color=C_GREEN, zorder=4)
    ax.text(x_p2_e - 1.8, y_c3, r"Temp $h_{\mathrm{temp}}, \mathbf{v}\leftarrow\mathbf{0}$", ha="right", va="center", fontsize=7.0, color=C_GREEN, zorder=4)

    r_s1 = patches.FancyBboxPatch((x_p2_s, y_s1 - arm_h/2.0), (x_p2_e - x_p2_s), arm_h,
                                 boxstyle="round,pad=0.2,rounding_size=0.6",
                                 facecolor="#FDEDEC", edgecolor=C_RED, lw=0.9, zorder=3)
    ax.add_patch(r_s1)
    ax.text(x_p2_s + 1.8, y_s1, rf"$\mathbf{{S_1}}$: Detour ({task_b})", va="center", fontsize=7.4, fontweight="bold", color=C_RED, zorder=4)
    ax.text(x_p2_e - 1.8, y_s1, r"Aux $h_B, \mathbf{v}\leftarrow\mathbf{0}$", ha="right", va="center", fontsize=7.0, color=C_RED, zorder=4)

    # Phase 2 span label above with generous padding
    ax.text((x_p2_s + x_p2_e)/2.0, y_c0 + 4.4, f"Phase 2: {ub} Intervention Window", ha="center", va="bottom",
            fontsize=7.2, fontweight="bold", color=C_DARK, zorder=4)

    # Convergence lines
    ax.plot([x_p2_e, x_conv], [y_c0, y_mid], color=C_BLUE, lw=1.2, zorder=3)
    ax.plot([x_p2_e, x_conv], [y_c3, y_mid], color=C_GREEN, lw=1.2, zorder=3)
    ax.plot([x_p2_e, x_conv], [y_s1, y_mid], color=C_RED, lw=1.2, zorder=3)
    ax.scatter(x_conv, y_mid, s=12, color=C_LINE, zorder=4)
    ax.plot([x_conv, x_th2], [y_mid, y_mid], color=C_LINE, lw=1.1, zorder=2)

    # Milestone theta2
    ax.scatter(x_th2, y_mid, s=36, marker="D", color=C_DARK, edgecolor="white", lw=0.9, zorder=5)
    ax.text(x_th2, y_mid + 4.8, r"$\boldsymbol{\theta}_2$", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color=C_DARK, zorder=6)
    ax.text(x_th2, y_mid - 4.8, f"Step {st_2}", ha="center", va="top", fontsize=6.8, color=C_MUTED, zorder=6)

    # Phase 3 card with increased vertical height and padding
    ax.plot([x_th2, x_p3_s], [y_mid, y_mid], color=C_LINE, lw=1.1, zorder=2)
    p3 = patches.FancyBboxPatch((x_p3_s, y_mid - 6.5), (x_p3_e - x_p3_s), 13.0,
                               boxstyle="round,pad=0.2,rounding_size=0.8",
                               facecolor="#FFFFFF", edgecolor="#94A3B8", lw=0.8, zorder=2)
    ax.add_patch(p3)
    ax.text((x_p3_s + x_p3_e)/2.0, y_mid + 2.5, f"Phase 3: {u2}", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color=C_DARK, zorder=3)
    ax.text((x_p3_s + x_p3_e)/2.0, y_mid - 2.5, r"Relearning ($h_A$)", ha="center", va="center",
            fontsize=7.0, color=C_MUTED, zorder=3)

    # Final theta3
    ax.plot([x_p3_e, x_th3], [y_mid, y_mid], color=C_LINE, lw=1.1, zorder=2)
    ax.scatter(x_th3, y_mid, s=36, marker="D", color=C_DARK, edgecolor="white", lw=0.9, zorder=5)
    ax.text(x_th3, y_mid + 4.8, r"$\boldsymbol{\theta}_3$", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color=C_DARK, zorder=6)
    ax.text(x_th3, y_mid - 4.8, "Final", ha="center", va="top", fontsize=6.8, color=C_MUTED, zorder=6)

# Generous 6-unit vertical separation between Regime 1 and Regime 2
draw_regime(ax, y_base=53.0, box_h=45.0,
            title="Regime 1: From-Scratch Pretraining and Detour",
            budget_tag="35,000 Updates",
            task_a="CIFAR-100", task_b="STL-10",
            arch_tag="GroupNorm ResNet-18",
            u1="15k", ub="5k", u2="15k",
            st_1="15k", st_2="20k")

draw_regime(ax, y_base=2.0, box_h=45.0,
            title="Regime 2: Applied Vision Transfer in Clinical Dermoscopy",
            budget_tag="12,000 Updates",
            task_a="ISIC 2019", task_b="Imagenette",
            arch_tag="ImageNet ResNet-50",
            u1="5k", ub="2k", u2="5k",
            st_1="5k", st_2="7k")

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

pdf_out = os.path.join(FIGURES_DIR, "figure_edafnet_style_framework.pdf")
png_out = os.path.join(FIGURES_DIR, "figure_edafnet_style_framework.png")
fig.savefig(pdf_out, bbox_inches="tight", pad_inches=0.05)
fig.savefig(png_out, bbox_inches="tight", pad_inches=0.05, dpi=300)

for d in [CLEAN_FIGURES_DIR]:
    fig.savefig(os.path.join(d, "figure_edafnet_style_framework.pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(os.path.join(d, "figure_edafnet_style_framework.png"), bbox_inches="tight", pad_inches=0.05, dpi=300)

plt.close(fig)
print(f"Figure 1 successfully generated: {pdf_out}")
