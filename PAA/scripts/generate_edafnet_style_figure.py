#!/usr/bin/env python3
"""
Generate a simplified, highly readable executive architecture and experimental
schedule figure for Figure 1, optimized for publication standards in PAA.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Canvas styling
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../figures"))
os.makedirs(FIGURES_DIR, exist_ok=True)

# 1. Load sample CIFAR-100 images
cifar_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data_cache/cifar-100-python/train"))
sample_images = []
if os.path.exists(cifar_path):
    with open(cifar_path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
        raw_data = d[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        sample_images = [raw_data[4], raw_data[17], raw_data[29]]
else:
    sample_images = [
        np.ones((32, 32, 3), dtype=np.uint8) * 130,
        np.ones((32, 32, 3), dtype=np.uint8) * 160,
        np.ones((32, 32, 3), dtype=np.uint8) * 190
    ]

# 2. Canvas Setup (19.0 x 8.0 inches), Y from 0 to 26
fig, ax = plt.subplots(figsize=(19.0, 8.0), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 26)
ax.axis('off')

# Color Palette: Clean, professional, high-contrast
C_DARK = '#0F172A'       # Slate 900
C_TEXT = '#1E293B'       # Slate 800
C_MUTED = '#475569'      # Slate 600
C_BORDER = '#94A3B8'     # Slate 400
C_LINE = '#334155'       # Slate 700

# Arm colors
C_C3_FILL = '#FAF5FF'    # Soft Purple
C_C3_BORDER = '#7E22CE'  # Purple 700
C_C0_FILL = '#F0F9FF'    # Soft Blue
C_C0_BORDER = '#0284C7'  # Sky 600
C_S1_FILL = '#FFF7ED'    # Soft Amber / Orange
C_S1_BORDER = '#EA580C'  # Orange 600

# Diagnostic & neutral colors
C_BADGE_GREEN = '#15803D'
C_BADGE_RED = '#B91C1C'

# Track Y positions (Bilateral symmetry)
Y_TOP = 19.3   # C3
Y_MID = 13.0   # C0
Y_BOT = 6.7    # S1

# Helper: Draw rounded card
def draw_card(ax, xy, width, height, title, items=None, fill='#FFFFFF', border=C_BORDER, 
              title_color=C_DARK, title_size=11.0, item_size=9.2, pad=0.35, lw=1.5, zorder=3):
    rect = patches.FancyBboxPatch(
        xy, width, height,
        boxstyle=f"round,pad={pad},rounding_size=0.6",
        facecolor=fill, edgecolor=border, linewidth=lw, zorder=zorder
    )
    ax.add_patch(rect)
    cx = xy[0] + width / 2.0
    
    if items:
        ty = xy[1] + height - 0.95
        ax.text(cx, ty, title, ha='center', va='center', fontsize=title_size, 
                fontweight='bold', color=title_color, zorder=zorder+1)
        sep_y = ty - 0.65
        ax.plot([xy[0] + 0.4, xy[0] + width - 0.4], [sep_y, sep_y], color=border, lw=0.8, alpha=0.6, zorder=zorder+1)
        
        n_items = len(items)
        spacing = (height - 2.2) / max(n_items, 1)
        start_y = sep_y - 0.65
        for i, item in enumerate(items):
            iy = start_y - i * spacing
            ax.text(xy[0] + 0.5, iy, item, ha='left', va='center', fontsize=item_size, 
                    color=C_TEXT, zorder=zorder+1)
    else:
        cy = xy[1] + height / 2.0
        ax.text(cx, cy, title, ha='center', va='center', fontsize=title_size, 
                fontweight='bold', color=title_color, zorder=zorder+1)
    return rect

# Helper: Orthogonal connection
def draw_conn(ax, points, color=C_LINE, lw=1.6, arrow=True, style='-'):
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i+1]
        is_last = (i == len(points) - 2)
        if is_last and arrow:
            ax.annotate(
                '', xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=12.0),
                zorder=2
            )
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle=style, color=color, lw=lw, zorder=2)

# =========================================================================
# --- STAGE HEADERS (Y = 24.0) ---
# =========================================================================
ax.text(15.5, 24.2, "Phase 1: Initial Target Exposure [0 to 15,000 updates]", ha='center', va='center', fontsize=11.2, fontweight='bold', color=C_DARK)
ax.text(45.5, 24.2, "Phase 2: Schedule Interruption [15,000 to 20,000 updates]", ha='center', va='center', fontsize=11.2, fontweight='bold', color=C_DARK)
ax.text(69.8, 24.2, "Phase 3: Target Relearning [20,000 to 35,000 updates]", ha='center', va='center', fontsize=11.2, fontweight='bold', color=C_DARK)
ax.text(90.8, 24.2, "Confirmatory Synthesis & Diagnostics", ha='center', va='center', fontsize=11.2, fontweight='bold', color=C_DARK)

# =========================================================================
# --- STAGE 1: PHASE 1 SHARED INITIALIZATION (X: 0.8 to 31.0) ---
# =========================================================================
# Target CIFAR-100 card (Thumbnails positioned neatly at bottom inside)
draw_card(ax, (0.8, 14.2), 7.2, 7.8, "CIFAR-100 (Target)", 
          items=[
              r"$\bullet$ $K_A = 100$ cls ($50\mathrm{k}$ train)",
              r"$\bullet$ Size: $32 \times 32 \times 3$"
          ],
          fill=C_C0_FILL, border=C_C0_BORDER, title_color=C_C0_BORDER, title_size=10.0, item_size=8.8)
ab1 = AnnotationBbox(OffsetImage(sample_images[0], zoom=1.1), (2.8, 15.6), frameon=True, pad=0.12,
                     bboxprops=dict(edgecolor=C_C0_BORDER, linewidth=1.0, facecolor='white'), zorder=5)
ab2 = AnnotationBbox(OffsetImage(sample_images[1], zoom=1.1), (5.8, 15.6), frameon=True, pad=0.12,
                     bboxprops=dict(edgecolor=C_C0_BORDER, linewidth=1.0, facecolor='white'), zorder=5)
ax.add_artist(ab1); ax.add_artist(ab2)

# Auxiliary STL-10 card
draw_card(ax, (0.8, 4.0), 7.2, 7.8, "STL-10 (Auxiliary)", 
          items=[
              r"$\bullet$ $K_B = 10$ cls ($5\mathrm{k}$ labeled)",
              r"$\bullet$ Resized: $96 \to 32 \times 32$"
          ],
          fill=C_S1_FILL, border=C_S1_BORDER, title_color=C_S1_BORDER, title_size=10.0, item_size=8.8)
ab3 = AnnotationBbox(OffsetImage(sample_images[2], zoom=1.1), (4.4, 5.4), frameon=True, pad=0.12,
                     bboxprops=dict(edgecolor=C_S1_BORDER, linewidth=1.0, facecolor='white'), zorder=5)
ax.add_artist(ab3)

# Connect Inputs to Shared Backbone
draw_conn(ax, [(8.0, 18.1), (9.8, 18.1), (9.8, Y_MID + 0.8), (10.6, Y_MID + 0.8)], arrow=False)
draw_conn(ax, [(8.0, 7.9), (9.8, 7.9), (9.8, Y_MID - 0.8), (10.6, Y_MID - 0.8)], arrow=False)
draw_conn(ax, [(9.8, Y_MID), (10.6, Y_MID)], arrow=True)

# Shared Backbone Card
draw_card(ax, (10.6, Y_MID - 4.4), 9.6, 8.8, "Shared Feature Backbone",
          items=[
              r"$\bullet$ ResNet-18 Architecture",
              r"$\bullet$ GroupNorm ($G=32$ groups)",
              r"$\bullet$ Zero batch-stat drift",
              r"$\bullet$ Feature dim: $d = 512$",
              r"$\bullet$ SGD: $\eta = 0.1, \mu = 0.9$"
          ],
          fill='#FFFFFF', border=C_LINE, title_color=C_DARK, title_size=10.8, item_size=9.0)

# Connect Backbone to Checkpoint theta_1
draw_conn(ax, [(20.2, Y_MID), (22.2, Y_MID)], arrow=True)

# Checkpoint theta_1 Card
draw_card(ax, (22.2, Y_MID - 4.4), 8.8, 8.8, r"Checkpoint $\theta_1$ (15k)",
          items=[
              r"$\bullet$ Step $15{,}000$ on CIFAR-100",
              r"$\bullet$ Shared pre-interruption state",
              r"$\bullet$ Stashed target head $h_A$",
              r"$\bullet$ Probe: $60.07\%$ ($N=6$)",
              r"$\bullet$ EffRank: $178.04$ ($N=6$)"
          ],
          fill='#FEFCE8', border='#CA8A04', title_color='#854D0E', title_size=10.8, item_size=9.2)

# =========================================================================
# --- STAGE 2: PHASE 2 SCHEDULE MODULATION (X: 33.5 to 57.5) ---
# =========================================================================
# Background container for Phase 2
rect_p2 = patches.FancyBboxPatch(
    (33.5, 3.8), 24.0, 19.4,
    boxstyle="round,pad=0.4,rounding_size=0.8",
    facecolor='#FAFAFA', edgecolor='#E2E8F0', linewidth=1.2, linestyle='--', zorder=1
)
ax.add_patch(rect_p2)

# Symmetrical 1-to-3 Fork from Checkpoint theta_1
draw_conn(ax, [(31.0, Y_MID), (33.5, Y_MID)], arrow=False)
draw_conn(ax, [(33.5, Y_MID), (33.5, Y_TOP), (35.0, Y_TOP)], arrow=True)
draw_conn(ax, [(33.5, Y_MID), (35.0, Y_MID)], arrow=True)
draw_conn(ax, [(33.5, Y_MID), (33.5, Y_BOT), (35.0, Y_BOT)], arrow=True)
ax.plot(33.5, Y_MID, 'o', color=C_LINE, markersize=5.0, zorder=4)

# Top Track: C3 On-Target Reset Control
draw_card(ax, (35.0, Y_TOP - 2.8), 21.2, 5.6, r"On-Target Reset Control ($C_3$)",
          items=[
              r"$\bullet$ Data: CIFAR-100 (Target) | Head: Temporary $h_{\rm temp}$ ($100$ classes)",
              r"$\bullet$ Boundary Resets: Velocity zeroed ($\mathbf{v} \leftarrow \mathbf{0}$) at $15\mathrm{k}$ and $20\mathrm{k}$",
              r"$\bullet$ Post-Intervention State: Probe decodability = $\mathbf{56.12\%}$ ($N=6$)"
          ],
          fill=C_C3_FILL, border=C_C3_BORDER, title_color=C_C3_BORDER, title_size=10.8, item_size=9.0)

# Center Track: C0 Continuous Target Baseline
draw_card(ax, (35.0, Y_MID - 2.8), 21.2, 5.6, r"Continuous Target Training ($C_0$ Baseline)",
          items=[
              r"$\bullet$ Data: CIFAR-100 (Target) | Head: Original $h_A$ active throughout",
              r"$\bullet$ Optimization: Continuous SGD momentum ($\mathbf{v}_t$ uninterrupted)",
              r"$\bullet$ Uninterrupted State: Probe decodability = $\mathbf{61.85\%}$ ($N=6$)"
          ],
          fill=C_C0_FILL, border=C_C0_BORDER, title_color=C_C0_BORDER, title_size=10.8, item_size=9.0)

# Bottom Track: S1 Auxiliary Learning Interval
draw_card(ax, (35.0, Y_BOT - 2.8), 21.2, 5.6, r"Auxiliary Learning Interval ($S_1$)",
          items=[
              r"$\bullet$ Data: STL-10 (Auxiliary) | Head: Auxiliary $h_B$ ($10$ classes)",
              r"$\bullet$ Boundary Resets: Velocity zeroed ($\mathbf{v} \leftarrow \mathbf{0}$) at $15\mathrm{k}$ and $20\mathrm{k}$",
              r"$\bullet$ Target Disruption Shock: Probe decodability = $\mathbf{20.65\%}$ ($N=6$)"
          ],
          fill=C_S1_FILL, border=C_S1_BORDER, title_color=C_S1_BORDER, title_size=10.8, item_size=9.0)

# =========================================================================
# --- STAGE 3: PHASE 3 RELEARNING & DIAGNOSTICS (X: 59.5 to 80.0) ---
# =========================================================================
# Background container for Phase 3
rect_p3 = patches.FancyBboxPatch(
    (59.5, 3.8), 20.5, 19.4,
    boxstyle="round,pad=0.4,rounding_size=0.8",
    facecolor='#FAFAFA', edgecolor='#E2E8F0', linewidth=1.2, linestyle='--', zorder=1
)
ax.add_patch(rect_p3)

# Connect Phase 2 to Phase 3
draw_conn(ax, [(56.2, Y_TOP), (60.5, Y_TOP)], arrow=True)
draw_conn(ax, [(56.2, Y_MID), (60.5, Y_MID)], arrow=True)
draw_conn(ax, [(56.2, Y_BOT), (60.5, Y_BOT)], arrow=True)

# Top Track Outcome (C3)
draw_card(ax, (60.5, Y_TOP - 2.8), 18.5, 5.6, r"$C_3$ Terminal Relearning (35k)",
          items=[
              r"$\bullet$ Target head $h_A$ restored; momentum reset",
              r"$\bullet$ Final Test Accuracy: $\mathbf{56.29\%}$ (SD: $1.59\%$, $N=20$)",
              r"$\bullet$ Probe: $62.14\%$ | Effective Rank: $157.80$ ($N=6$)"
          ],
          fill=C_C3_FILL, border=C_C3_BORDER, title_color=C_C3_BORDER, title_size=10.8, item_size=9.0)

# Center Track Outcome (C0)
draw_card(ax, (60.5, Y_MID - 2.8), 18.5, 5.6, r"$C_0$ Continuous Terminal (35k)",
          items=[
              r"$\bullet$ Continuous target optimization to $35\mathrm{k}$",
              r"$\bullet$ Final Test Accuracy: $\mathbf{56.69\%}$ (SD: $0.75\%$, $N=20$)",
              r"$\bullet$ Probe: $63.01\%$ | Effective Rank: $162.45$ ($N=6$)"
          ],
          fill=C_C0_FILL, border=C_C0_BORDER, title_color=C_C0_BORDER, title_size=10.8, item_size=9.0)

# Bottom Track Outcome (S1)
draw_card(ax, (60.5, Y_BOT - 2.8), 18.5, 5.6, r"$S_1$ Auxiliary Terminal (35k)",
          items=[
              r"$\bullet$ Target head $h_A$ restored; momentum reset",
              r"$\bullet$ Final Test Accuracy: $\mathbf{56.26\%}$ (SD: $1.51\%$, $N=20$)",
              r"$\bullet$ Probe: $61.96\%$ | Effective Rank: $153.69$ ($N=6$)"
          ],
          fill=C_S1_FILL, border=C_S1_BORDER, title_color=C_S1_BORDER, title_size=10.8, item_size=9.0)

# =========================================================================
# --- STAGE 4: EVALUATIVE SYNTHESIS & CONTRASTS (X: 82.5 to 99.2) ---
# =========================================================================
# Connect Phase 3 into Synthesis Card
draw_conn(ax, [(79.0, Y_TOP), (81.0, Y_TOP), (81.0, Y_MID + 1.5), (82.5, Y_MID + 1.5)], arrow=True)
draw_conn(ax, [(79.0, Y_MID), (82.5, Y_MID)], arrow=True)
draw_conn(ax, [(79.0, Y_BOT), (81.0, Y_BOT), (81.0, Y_MID - 1.5), (82.5, Y_MID - 1.5)], arrow=True)

# Synthesis Card
rect_syn = patches.FancyBboxPatch(
    (82.5, 3.8), 16.5, 18.2,
    boxstyle="round,pad=0.4,rounding_size=0.8",
    facecolor='#FFFFFF', edgecolor=C_DARK, linewidth=1.8, zorder=3
)
ax.add_patch(rect_syn)

cx_syn = 90.75
ax.text(cx_syn, 20.8, "Confirmatory Synthesis", ha='center', va='center', fontsize=11.2, fontweight='bold', color=C_DARK, zorder=4)
ax.plot([83.5, 98.0], [20.0, 20.0], color='#CBD5E1', lw=1.0, zorder=4)

# Contrast 1: H1 (Primary)
ax.text(cx_syn, 18.8, "Primary Contrast: Hypothesis H1 (N=20)", ha='center', va='center', fontsize=9.6, fontweight='bold', color=C_BADGE_RED, zorder=4)
ax.text(cx_syn, 17.6, r"$\Delta_{\rm total} = \mathrm{Acc}(S_1) - \mathrm{Acc}(C_0)$", ha='center', va='center', fontsize=9.2, color=C_DARK, zorder=4)
ax.text(cx_syn, 16.4, r"$\mathbf{-0.43}$ pp (95% CI: [$-0.94$, $+0.07$])", ha='center', va='center', fontsize=9.8, fontweight='bold', color=C_DARK, zorder=4)
ax.text(cx_syn, 15.3, r"$t(19) = -1.79, p = 0.090$ (perm $p = 0.091$)", ha='center', va='center', fontsize=8.6, color=C_MUTED, zorder=4)
ax.text(cx_syn, 14.3, r"$\rightarrow$ Statistically uncertain", ha='center', va='center', fontsize=9.0, fontweight='bold', color=C_BADGE_RED, zorder=4)

ax.plot([83.5, 98.0], [13.4, 13.4], color='#CBD5E1', lw=1.0, zorder=4)

# Contrast 2: H2 (Procedural Reset)
ax.text(cx_syn, 12.3, "Procedural Control: Hypothesis H2 (N=20)", ha='center', va='center', fontsize=9.6, fontweight='bold', color='#7C3AED', zorder=4)
ax.text(cx_syn, 11.2, r"$\Delta_{\rm proc} = \mathrm{Acc}(C_3) - \mathrm{Acc}(C_0)$", ha='center', va='center', fontsize=9.2, color=C_DARK, zorder=4)
ax.text(cx_syn, 10.1, r"$\mathbf{-0.40}$ pp (95% CI: [$-1.14$, $+0.34$])", ha='center', va='center', fontsize=9.8, fontweight='bold', color=C_DARK, zorder=4)
ax.text(cx_syn, 9.1, r"$t(19) = -1.14, p = 0.269$ (Invariance not rejected)", ha='center', va='center', fontsize=8.6, color=C_MUTED, zorder=4)

ax.plot([83.5, 98.0], [8.1, 8.1], color='#CBD5E1', lw=1.0, zorder=4)

# Representation Dynamics: RQ3
ax.text(cx_syn, 7.0, "Representation Dynamics: RQ3 (N=6)", ha='center', va='center', fontsize=9.6, fontweight='bold', color=C_BADGE_GREEN, zorder=4)
ax.text(cx_syn, 5.9, r"Probe Decodability: $20.65\% \to 61.96\%$", ha='center', va='center', fontsize=9.0, color=C_DARK, zorder=4)
ax.text(cx_syn, 4.8, r"EffRank Rebound: $124.50 \to 153.69$", ha='center', va='center', fontsize=9.0, color=C_DARK, zorder=4)

# =========================================================================
# --- BOTTOM TIMELINE & PROTOCOL SUMMARY (Y: 0.6 to 2.4) ---
# =========================================================================
y_tline = 1.8
# Phase 1 bracket
ax.plot([0.8, 31.0], [y_tline, y_tline], color=C_DARK, lw=2.0)
ax.plot([0.8, 0.8], [y_tline - 0.35, y_tline + 0.35], color=C_DARK, lw=2.0)
ax.plot([31.0, 31.0], [y_tline - 0.35, y_tline + 0.35], color=C_DARK, lw=2.0)
ax.text(15.9, y_tline - 0.9, "Phase 1: Initial Target Training [0 to 15,000 updates]", ha='center', va='center', fontsize=9.5, fontweight='bold', color=C_DARK)

# Phase 2 bracket
ax.plot([33.5, 57.5], [y_tline, y_tline], color=C_DARK, lw=2.0)
ax.plot([33.5, 33.5], [y_tline - 0.35, y_tline + 0.35], color=C_DARK, lw=2.0)
ax.plot([57.5, 57.5], [y_tline - 0.35, y_tline + 0.35], color=C_DARK, lw=2.0)
ax.text(45.5, y_tline - 0.9, "Phase 2: Schedule Interruption [15,000 to 20,000 updates]", ha='center', va='center', fontsize=9.5, fontweight='bold', color=C_DARK)

# Phase 3 bracket
ax.plot([59.5, 99.0], [y_tline, y_tline], color=C_DARK, lw=2.0)
ax.plot([59.5, 59.5], [y_tline - 0.35, y_tline + 0.35], color=C_DARK, lw=2.0)
ax.plot([99.0, 99.0], [y_tline - 0.35, y_tline + 0.35], color=C_DARK, lw=2.0)
ax.text(79.25, y_tline - 0.9, "Phase 3: Target Relearning & Synthesis [20,000 to 35,000 updates | Budget = 35k total updates]", ha='center', va='center', fontsize=9.5, fontweight='bold', color=C_DARK)

# Save figures
png_out = os.path.join(FIGURES_DIR, "figure_edafnet_style_framework.png")
pdf_out = os.path.join(FIGURES_DIR, "figure_edafnet_style_framework.pdf")

plt.tight_layout()
plt.savefig(png_out, dpi=300, bbox_inches='tight')
plt.savefig(pdf_out, bbox_inches='tight')
plt.close()

print(f"✓ Saved refined figure PNG: {png_out} ({os.path.getsize(png_out)/1024:.1f} KB)")
print(f"✓ Saved refined figure PDF: {pdf_out} ({os.path.getsize(pdf_out)/1024:.1f} KB)")
