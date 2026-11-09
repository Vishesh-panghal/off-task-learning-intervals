#!/usr/bin/env python3
"""
Generate publication-grade Figure 1: Controlled Experimental Protocol and Schedule Architecture.
Displays both Regime 1 (From-Scratch CIFAR-100 / STL-10, 35k updates) and
Regime 2 (Applied Transfer ISIC 2019 / Imagenette, 12k updates) as a clean, color-coded timeline.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Styling configuration
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Color Palette (consistent across all manuscript figures)
C_BLUE = '#1F77B4'      # C0 Continuous Target Baseline
C_GREEN = '#2CA02C'     # C3 Procedural Reset Control
C_RED = '#D62728'       # S1 Auxiliary Detour
C_PURPLE = '#9467BD'    # S1-shuf Control
C_DARK = '#1C2833'      # Dark charcoal for primary text
C_MUTED = '#4A5568'     # Slate gray for annotations

fig, ax = plt.subplots(figsize=(15.2, 9.6), dpi=300)
ax.set_xlim(-1, 101)
ax.set_ylim(-1, 65)
ax.axis('off')

def draw_badge(ax, x, y, text, color, bg_color='#FFFFFF', fontsize=8.2, border_lw=1.1):
    bbox = dict(boxstyle="round,pad=0.25,rounding_size=0.3", facecolor=bg_color, edgecolor=color, lw=border_lw)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=color, bbox=bbox, zorder=6)

def draw_regime_timeline(ax, y_base, title, regime_tag, arch_info, total_budget,
                         p1_range, p2_range, p3_range, 
                         theta1_text, theta2_text, theta3_text,
                         has_shuf=False):
    box_h = 28.0
    rect = patches.FancyBboxPatch(
        (0.5, y_base - 1.5), 99.0, box_h,
        boxstyle="round,pad=0.3,rounding_size=0.8",
        facecolor='#FCFDFE', edgecolor='#CBD5E1', linewidth=1.2, zorder=1
    )
    ax.add_patch(rect)
    
    # Title header banner
    ax.text(2.2, y_base + box_h - 3.2, title, fontsize=12.2, fontweight='bold', color=C_DARK, zorder=3)
    ax.text(97.8, y_base + box_h - 3.2, f"Total Budget Ceiling: {total_budget}", ha='right', fontsize=10.2, fontweight='bold', color='#2C3E50', zorder=3)
    ax.text(2.2, y_base + box_h - 5.2, arch_info, fontsize=9.0, color=C_MUTED, zorder=3)
    
    x1_s, x1_e = p1_range
    x2_s, x2_e = p2_range
    x3_s, x3_e = p3_range
    
    y_mid = y_base + 11.2
    y_top = y_mid + 4.8
    y_bot = y_mid - 4.8
    y_shuf = y_mid - 8.6 if has_shuf else None
    
    # --- PHASE 1 (Target Exposure) ---
    p1_rect = patches.FancyBboxPatch(
        (x1_s, y_mid - 2.0), (x1_e - x1_s), 4.0,
        boxstyle="round,pad=0.1,rounding_size=0.4",
        facecolor='#F1F5F9', edgecolor='#94A3B8', linewidth=1.2, zorder=2
    )
    ax.add_patch(p1_rect)
    ax.text((x1_s + x1_e)/2.0, y_mid + 0.4, "Phase 1: Target Warmup" if 'Regime 1' in title else "Phase 1: Target Fine-Tuning", 
            ha='center', va='center', fontsize=9.4, fontweight='bold', color=C_DARK, zorder=3)
    ax.text((x1_s + x1_e)/2.0, y_mid - 1.0, f"Target Task A ({'ISIC 2019' if 'Regime 2' in title else 'CIFAR-100'}) | Head $h_A$", 
            ha='center', va='center', fontsize=8.2, color=C_MUTED, zorder=3)
    
    # Milestone theta1 (Diamond)
    m1_x = (x1_e + x2_s) / 2.0 - 2.5
    ax.scatter(m1_x, y_mid, s=120, marker='D', color=C_DARK, edgecolor='white', lw=1.5, zorder=5)
    ax.text(m1_x, y_mid + 2.0, r"$\boldsymbol{\theta}_1$", ha='center', va='bottom', fontsize=10.0, fontweight='bold', color=C_DARK, zorder=6)
    ax.text(m1_x, y_mid - 2.2, theta1_text, ha='center', va='top', fontsize=7.8, color=C_MUTED, zorder=6)
    
    # Line from Phase 1 to milestone 1, then branching to Phase 2
    ax.plot([x1_e, m1_x], [y_mid, y_mid], color='#94A3B8', lw=1.5, linestyle='-', zorder=2)
    ax.plot([m1_x, x2_s], [y_mid, y_top], color=C_BLUE, lw=1.8, linestyle='-', zorder=3)
    ax.plot([m1_x, x2_s], [y_mid, y_mid], color=C_GREEN, lw=1.8, linestyle='-', zorder=3)
    ax.plot([m1_x, x2_s], [y_mid, y_bot], color=C_RED, lw=1.8, linestyle='-', zorder=3)
    if has_shuf and y_shuf is not None:
        ax.plot([m1_x, x2_s], [y_mid, y_shuf], color=C_PURPLE, lw=1.8, linestyle='--', zorder=3)
    
    # Arm C0 (Continuous Baseline)
    r_c0 = patches.FancyBboxPatch((x2_s, y_top - 1.4), (x2_e - x2_s), 2.8, boxstyle="round,pad=0.1,rounding_size=0.3",
                                 facecolor='#EBF5FB', edgecolor=C_BLUE, lw=1.4, zorder=3)
    ax.add_patch(r_c0)
    ax.text(x2_s + 1.2, y_top, r"$\mathbf{C_0}$: Continuous Target", va='center', fontsize=8.8, fontweight='bold', color=C_BLUE, zorder=4)
    ax.text(x2_e - 1.2, y_top, "Head $h_A$ Intact | Momentum $\mathbf{v}$ Cont.", ha='right', va='center', fontsize=8.0, color='#2874A6', zorder=4)
    
    # Arm C3 (On-Target Reset Control)
    r_c3 = patches.FancyBboxPatch((x2_s, y_mid - 1.4), (x2_e - x2_s), 2.8, boxstyle="round,pad=0.1,rounding_size=0.3",
                                 facecolor='#EAFAF1', edgecolor=C_GREEN, lw=1.4, zorder=3)
    ax.add_patch(r_c3)
    ax.text(x2_s + 1.2, y_mid, r"$\mathbf{C_3}$: Procedural Reset Control", va='center', fontsize=8.8, fontweight='bold', color=C_GREEN, zorder=4)
    draw_badge(ax, x2_s + 20.8, y_mid, r"Head $h_{\mathrm{temp}}$", C_GREEN, '#FFFFFF', fontsize=7.8)
    draw_badge(ax, x2_s + 28.6, y_mid, r"$\mathbf{v} \leftarrow \mathbf{0}$", C_GREEN, '#FFFFFF', fontsize=7.8)
    
    # Arm S1 (Auxiliary Detour)
    r_s1 = patches.FancyBboxPatch((x2_s, y_bot - 1.4), (x2_e - x2_s), 2.8, boxstyle="round,pad=0.1,rounding_size=0.3",
                                 facecolor='#FDEDEC', edgecolor=C_RED, lw=1.4, zorder=3)
    ax.add_patch(r_s1)
    ax.text(x2_s + 1.2, y_bot, r"$\mathbf{S_1}$: Auxiliary Detour (" + ('Imagenette' if 'Regime 2' in title else 'STL-10') + ")", va='center', fontsize=8.8, fontweight='bold', color=C_RED, zorder=4)
    draw_badge(ax, x2_s + 23.2, y_bot, r"Aux Head $h_B$", C_RED, '#FFFFFF', fontsize=7.8)
    draw_badge(ax, x2_s + 30.2, y_bot, r"$\mathbf{v} \leftarrow \mathbf{0}$", C_RED, '#FFFFFF', fontsize=7.8)
    
    # Optional Shuffled Detour
    if has_shuf and y_shuf is not None:
        r_shuf = patches.FancyBboxPatch((x2_s, y_shuf - 1.2), (x2_e - x2_s), 2.4, boxstyle="round,pad=0.1,rounding_size=0.3",
                                       facecolor='#F4ECF7', edgecolor=C_PURPLE, lw=1.2, linestyle='--', zorder=3)
        ax.add_patch(r_shuf)
        ax.text(x2_s + 1.2, y_shuf, r"$\mathbf{S_1\text{-shuf}}$: Label-Shuffled Detour", va='center', fontsize=8.2, fontweight='bold', color=C_PURPLE, zorder=4)
        ax.text(x2_e - 1.2, y_shuf, "Permuted Labels | Flushed $\mathbf{v}$", ha='right', va='center', fontsize=7.8, color=C_PURPLE, zorder=4)

    # Milestone theta2 (Diamond)
    m2_x = 71.5
    ax.scatter(m2_x, y_mid, s=120, marker='D', color=C_DARK, edgecolor='white', lw=1.5, zorder=5)
    ax.text(m2_x, y_mid + 2.0, r"$\boldsymbol{\theta}_2$", ha='center', va='bottom', fontsize=10.0, fontweight='bold', color=C_DARK, zorder=6)
    ax.text(m2_x, y_mid - 2.2, theta2_text, ha='center', va='top', fontsize=7.8, color=C_MUTED, zorder=6)
    
    # Connectors from Phase 2 into milestone 2, then into Phase 3
    ax.plot([x2_e, m2_x], [y_top, y_mid], color=C_BLUE, lw=1.8, linestyle='-', zorder=3)
    ax.plot([x2_e, m2_x], [y_mid, y_mid], color=C_GREEN, lw=1.8, linestyle='-', zorder=3)
    ax.plot([x2_e, m2_x], [y_bot, y_mid], color=C_RED, lw=1.8, linestyle='-', zorder=3)
    if has_shuf and y_shuf is not None:
        ax.plot([x2_e, m2_x], [y_shuf, y_mid], color=C_PURPLE, lw=1.8, linestyle='--', zorder=3)
    ax.plot([m2_x, x3_s], [y_mid, y_mid], color='#94A3B8', lw=1.5, linestyle='-', zorder=2)

    # --- PHASE 3 (Target Relearning) ---
    p3_rect = patches.FancyBboxPatch(
        (x3_s, y_mid - 2.0), (x3_e - x3_s), 4.0,
        boxstyle="round,pad=0.1,rounding_size=0.4",
        facecolor='#F1F5F9', edgecolor='#94A3B8', linewidth=1.2, zorder=2
    )
    ax.add_patch(p3_rect)
    ax.text((x3_s + x3_e)/2.0, y_mid + 0.4, "Phase 3: Target Relearning (All Arms)", ha='center', va='center', fontsize=9.4, fontweight='bold', color=C_DARK, zorder=3)
    ax.text((x3_s + x3_e)/2.0, y_mid - 1.0, r"Return to Target Task A | Restored Head $h_A$", ha='center', va='center', fontsize=8.2, color=C_MUTED, zorder=3)

    # Intermediate milestone if applicable (e.g. theta_10k equal-target comparator)
    if 'Regime 2' in title:
        x_10k = x3_s + (x3_e - x3_s) * (3.0 / 5.0)
        ax.scatter(x_10k, y_mid - 4.5, s=90, marker='o', color='#7D3C98', edgecolor='white', lw=1.2, zorder=5)
        ax.plot([x_10k, x_10k], [y_mid - 2.0, y_mid - 4.5], color='#7D3C98', lw=1.2, linestyle=':', zorder=4)
        ax.text(x_10k, y_mid - 6.2, r"$\boldsymbol{\theta}_{\mathrm{10k}}$ (Equal Target)", ha='center', va='top', fontsize=7.8, fontweight='bold', color='#7D3C98', zorder=6)

    # Terminal evaluation theta3
    m3_x = x3_e + 2.8
    ax.plot([x3_e, m3_x], [y_mid, y_mid], color='#94A3B8', lw=1.5, linestyle='-', zorder=2)
    ax.scatter(m3_x, y_mid, s=130, marker='D', color='#B03A2E', edgecolor='white', lw=1.5, zorder=5)
    ax.text(m3_x, y_mid + 2.0, r"$\boldsymbol{\theta}_3$", ha='center', va='bottom', fontsize=10.0, fontweight='bold', color='#B03A2E', zorder=6)
    ax.text(m3_x, y_mid - 2.2, theta3_text, ha='center', va='top', fontsize=7.8, fontweight='bold', color='#B03A2E', zorder=6)


# Draw Regime 1 (Top Half, Y = 34 to 62)
draw_regime_timeline(
    ax, y_base=34.0,
    title="Regime 1: From-Scratch Pretraining and Exploration (CIFAR-100 & STL-10)",
    regime_tag="Regime 1",
    arch_info="Backbone: ResNet-18 with GroupNorm (G=32, d=512) | Constant SGD (eta=0.1, mu=0.9, wd=5e-4) | N=20 Confirmatory Seeds",
    total_budget="35,000 updates",
    p1_range=(3.0, 29.0),     # 15k / 35k
    p2_range=(37.0, 67.0),    # 5k detour
    p3_range=(76.0, 93.0),    # 15k relearning
    theta1_text="Step 15k\n(Stash $h_A$)",
    theta2_text="Step 20k\n(Audit Collapse)",
    theta3_text="Terminal 35k\n(H1, H2, RQ3)",
    has_shuf=False
)

# Draw Regime 2 (Bottom Half, Y = 3 to 31)
draw_regime_timeline(
    ax, y_base=3.0,
    title="Regime 2: Applied Vision Transfer Fine-Tuning in Clinical Diagnostics (ISIC 2019 & Imagenette)",
    regime_tag="Regime 2",
    arch_info="Backbone: ImageNet-pretrained ResNet-50 (d=2048) | Constant SGD (AMP, eta=0.01, mu=0.9, wd=1e-4) | Primary N=8 Seeds",
    total_budget="12,000 updates",
    p1_range=(3.0, 29.0),     # 5k / 12k
    p2_range=(37.0, 67.0),    # 2k detour
    p3_range=(76.0, 93.0),    # 5k relearning
    theta1_text="Step 5k\n(Stash $h_A$)",
    theta2_text="Step 7k\n(Audit Collapse)",
    theta3_text="Terminal 12k\n(Parity & Recovery)",
    has_shuf=True
)

pdf_path = os.path.join(FIGURES_DIR, "figure_edafnet_style_framework.pdf")
png_path = os.path.join(FIGURES_DIR, "figure_edafnet_style_framework.png")
plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
plt.savefig(png_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"Refined Figure 1 successfully:\n  {pdf_path}\n  {png_path}")
