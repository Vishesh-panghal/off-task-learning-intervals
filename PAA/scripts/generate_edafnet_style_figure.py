#!/usr/bin/env python3
"""
Generate an executive, publication-grade experimental architecture and schedule figure
for Figure 1, specifically optimized for maximum legibility on standard paper / small screens.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Canvas typography
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../figures"))
os.makedirs(FIGURES_DIR, exist_ok=True)

# 1. Load sample CIFAR-100 images
cifar_path = "/Users/visheshpanghal/Downloads/Chapters&Papers/TMLR/data_cache/cifar-100-python/train"
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

# 2. Canvas Setup (12.4 x 6.8 inches)
# Coordinate system: X: [0, 100], Y: [0, 50]
fig, ax = plt.subplots(figsize=(12.4, 6.8), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.axis('off')

# Color Palette: Modern, high-contrast, executive
C_DARK = '#0F172A'       # Slate 900
C_TEXT = '#1E293B'       # Slate 800
C_MUTED = '#475569'      # Slate 600
C_BORDER = '#94A3B8'     # Slate 400
C_LINE = '#334155'       # Slate 700

# Arm colors
C_C3_FILL = '#FAF5FF'    # Soft Purple
C_C3_BORDER = '#7E22CE'  # Purple 700
C_C3_TEXT = '#581C87'    # Purple 900

C_C0_FILL = '#F0F9FF'    # Soft Sky
C_C0_BORDER = '#0284C7'  # Sky 600
C_C0_TEXT = '#0369A1'    # Sky 700

C_S1_FILL = '#FFF7ED'    # Soft Amber / Orange
C_S1_BORDER = '#EA580C'  # Orange 600
C_S1_TEXT = '#C2410C'    # Orange 700

# Badges & Accents
C_GREEN_BG = '#DCFCE7'
C_GREEN_BORDER = '#16A34A'
C_GREEN_TEXT = '#14532D'

C_RED_BG = '#FEE2E2'
C_RED_BORDER = '#DC2626'
C_RED_TEXT = '#7F1D1D'

C_AMBER_BG = '#FEF3C7'
C_AMBER_BORDER = '#D97706'
C_AMBER_TEXT = '#78350F'

# Helper: Draw Phase Header Pill
def draw_phase_header(ax, xy, width, height, title, subtitle, fill='#F8FAFC', border='#94A3B8', text_color=C_DARK):
    rect = patches.FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.2,rounding_size=0.8",
        facecolor=fill, edgecolor=border, linewidth=1.4, zorder=2
    )
    ax.add_patch(rect)
    cx = xy[0] + width / 2.0
    ax.text(cx, xy[1] + height * 0.65, title, ha='center', va='center', fontsize=11.2, fontweight='bold', color=text_color, zorder=3)
    ax.text(cx, xy[1] + height * 0.25, subtitle, ha='center', va='center', fontsize=8.6, fontweight='medium', color=C_MUTED, zorder=3)

# Helper: Orthogonal connection
def draw_conn(ax, points, color=C_LINE, lw=1.8, arrow=True, style='-'):
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i+1]
        is_last = (i == len(points) - 2)
        if is_last and arrow:
            ax.annotate(
                '', xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=14.0),
                zorder=2
            )
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle=style, color=color, lw=lw, zorder=2)

# =========================================================================
# --- PHASE TOP BANNERS (Y = 44.8 to 49.0) ---
# =========================================================================
draw_phase_header(ax, (1.0, 44.8), 30.6, 4.2, "Phase 1: Initial Target Exposure", "0 to 15,000 updates | CIFAR-100", fill='#F8FAFC', border='#94A3B8')
draw_phase_header(ax, (32.4, 44.8), 21.0, 4.2, "Phase 2: Schedule Interruption", "15,000 to 20,000 updates | 3 Arms", fill='#F8FAFC', border='#94A3B8')
draw_phase_header(ax, (54.2, 44.8), 21.0, 4.2, "Phase 3: Target Relearning", "20,000 to 35,000 updates | Budget = 35k", fill='#F8FAFC', border='#94A3B8')
draw_phase_header(ax, (76.0, 44.8), 23.0, 4.2, "Confirmatory Synthesis", "Hypothesis Testing & Diagnostics", fill='#F1F5F9', border=C_DARK, text_color=C_DARK)

# Track Y centers
Y_TOP = 36.55   # C3
Y_MID = 23.05   # C0
Y_BOT = 9.55    # S1
TRACK_H = 11.5

# =========================================================================
# --- STAGE 1: PHASE 1 SHARED INITIALIZATION (X: 1.0 to 31.6) ---
# =========================================================================
rect_p1 = patches.FancyBboxPatch(
    (1.0, 2.5), 30.6, 41.0,
    boxstyle="round,pad=0.2,rounding_size=1.0",
    facecolor='#FAFAFA', edgecolor='#E2E8F0', linewidth=1.2, linestyle='--', zorder=1
)
ax.add_patch(rect_p1)

# CIFAR-100 Input Card (Width = 7.0)
rect_cifar = patches.FancyBboxPatch(
    (1.8, 23.5), 7.0, 18.8,
    boxstyle="round,pad=0.2,rounding_size=0.8",
    facecolor=C_C0_FILL, edgecolor=C_C0_BORDER, linewidth=1.5, zorder=3
)
ax.add_patch(rect_cifar)
ax.text(5.3, 40.8, "CIFAR-100", ha='center', va='center', fontsize=10.2, fontweight='bold', color=C_C0_TEXT, zorder=4)
ax.plot([2.4, 8.2], [39.6, 39.6], color=C_C0_BORDER, lw=0.8, alpha=0.5, zorder=4)
ax.text(5.3, 38.0, "Target Dataset", ha='center', va='center', fontsize=8.4, fontweight='bold', color=C_DARK, zorder=4)
ax.text(5.3, 36.3, r"$K_A = 100$ classes", ha='center', va='center', fontsize=8.0, color=C_TEXT, zorder=4)
ax.text(5.3, 34.6, r"$50\mathrm{k}$ train (32$\times$32)", ha='center', va='center', fontsize=7.8, color=C_MUTED, zorder=4)

ab1 = AnnotationBbox(OffsetImage(sample_images[0], zoom=1.35), (3.6, 28.5), frameon=True, pad=0.08,
                     bboxprops=dict(edgecolor=C_C0_BORDER, linewidth=1.0, facecolor='white'), zorder=5)
ab2 = AnnotationBbox(OffsetImage(sample_images[1], zoom=1.35), (7.0, 28.5), frameon=True, pad=0.08,
                     bboxprops=dict(edgecolor=C_C0_BORDER, linewidth=1.0, facecolor='white'), zorder=5)
ax.add_artist(ab1); ax.add_artist(ab2)

# STL-10 Input Card (Width = 7.0)
rect_stl = patches.FancyBboxPatch(
    (1.8, 3.8), 7.0, 18.5,
    boxstyle="round,pad=0.2,rounding_size=0.8",
    facecolor=C_S1_FILL, edgecolor=C_S1_BORDER, linewidth=1.5, zorder=3
)
ax.add_patch(rect_stl)
ax.text(5.3, 20.8, "STL-10", ha='center', va='center', fontsize=10.2, fontweight='bold', color=C_S1_TEXT, zorder=4)
ax.plot([2.4, 8.2], [19.6, 19.6], color=C_S1_BORDER, lw=0.8, alpha=0.5, zorder=4)
ax.text(5.3, 18.0, "Auxiliary Dataset", ha='center', va='center', fontsize=8.4, fontweight='bold', color=C_DARK, zorder=4)
ax.text(5.3, 16.3, r"$K_B = 10$ classes", ha='center', va='center', fontsize=8.0, color=C_TEXT, zorder=4)
ax.text(5.3, 14.6, r"$5\mathrm{k}$ train (32$\times$32)", ha='center', va='center', fontsize=7.8, color=C_MUTED, zorder=4)

ab3 = AnnotationBbox(OffsetImage(sample_images[2], zoom=1.5), (5.3, 8.8), frameon=True, pad=0.08,
                     bboxprops=dict(edgecolor=C_S1_BORDER, linewidth=1.0, facecolor='white'), zorder=5)
ax.add_artist(ab3)

# Connect Inputs to Shared Backbone
draw_conn(ax, [(8.8, 32.9), (9.6, 32.9), (9.6, Y_MID + 1.2), (10.2, Y_MID + 1.2)], arrow=False)
draw_conn(ax, [(8.8, 13.05), (9.6, 13.05), (9.6, Y_MID - 1.2), (10.2, Y_MID - 1.2)], arrow=False)
draw_conn(ax, [(9.6, Y_MID), (10.2, Y_MID)], arrow=True)

# Shared Backbone Card (Width = 10.0)
rect_bb = patches.FancyBboxPatch(
    (10.2, Y_MID - 8.8), 10.0, 17.6,
    boxstyle="round,pad=0.2,rounding_size=0.8",
    facecolor='#FFFFFF', edgecolor=C_LINE, linewidth=1.6, zorder=3
)
ax.add_patch(rect_bb)
ax.text(15.2, Y_MID + 7.4, "Shared Backbone", ha='center', va='center', fontsize=10.4, fontweight='bold', color=C_DARK, zorder=4)
ax.plot([10.8, 19.6], [Y_MID + 6.2, Y_MID + 6.2], color=C_BORDER, lw=0.8, alpha=0.5, zorder=4)

bb_items = [
    r"$\bullet$ ResNet-18 Backbone",
    r"$\bullet$ GroupNorm ($G=32$)",
    r"$\bullet$ Zero batch-stat drift",
    r"$\bullet$ Feature dim: $d = 512$",
    r"$\bullet$ SGD: $\eta=0.1, \mu=0.9$",
    r"$\bullet$ Target: $15\mathrm{k}$ updates"
]
for i, item in enumerate(bb_items):
    ax.text(10.8, Y_MID + 4.8 - i * 2.1, item, fontsize=8.0, color=C_TEXT, zorder=4)

# Connect Backbone to Checkpoint theta_1
draw_conn(ax, [(20.2, Y_MID), (21.4, Y_MID)], arrow=True)

# Checkpoint theta_1 Card (Width = 9.4)
rect_cp = patches.FancyBboxPatch(
    (21.4, Y_MID - 8.8), 9.4, 17.6,
    boxstyle="round,pad=0.2,rounding_size=0.8",
    facecolor=C_AMBER_BG, edgecolor=C_AMBER_BORDER, linewidth=1.6, zorder=3
)
ax.add_patch(rect_cp)
ax.text(26.1, Y_MID + 7.4, r"Checkpoint $\theta_1$", ha='center', va='center', fontsize=10.4, fontweight='bold', color=C_AMBER_TEXT, zorder=4)
ax.plot([22.0, 30.2], [Y_MID + 6.2, Y_MID + 6.2], color=C_AMBER_BORDER, lw=0.8, alpha=0.5, zorder=4)

cp_items = [
    r"$\bullet$ Step $15{,}000$ (CIFAR)",
    r"$\bullet$ Shared initial state",
    r"$\bullet$ Head $h_A$ stashed",
    r"$\bullet$ Common arm origin"
]
for i, item in enumerate(cp_items):
    ax.text(22.0, Y_MID + 4.8 - i * 2.1, item, fontsize=8.0, color=C_TEXT, zorder=4)

# Dedicated Checkpoint theta_1 Metric Badge
rect_cp_badge = patches.FancyBboxPatch(
    (21.8, Y_MID - 8.0), 8.6, 4.6,
    boxstyle="round,pad=0.1,rounding_size=0.6",
    facecolor='#FFFFFF', edgecolor=C_AMBER_BORDER, linewidth=1.2, zorder=4
)
ax.add_patch(rect_cp_badge)
ax.text(26.1, Y_MID - 4.6, r"Probe: $\mathbf{60.07\%}$", ha='center', va='center', fontsize=9.0, fontweight='bold', color=C_AMBER_TEXT, zorder=5)
ax.text(26.1, Y_MID - 6.1, r"EffRank: $\mathbf{178.04}$", ha='center', va='center', fontsize=8.4, color=C_DARK, zorder=5)
ax.text(26.1, Y_MID - 7.3, r"($N=6$ paired cohort)", ha='center', va='center', fontsize=7.4, color=C_MUTED, zorder=5)

# =========================================================================
# --- STAGE 2: PHASE 2 SCHEDULE MODULATION (X: 32.4 to 53.4) ---
# =========================================================================
rect_p2 = patches.FancyBboxPatch(
    (32.4, 2.5), 21.0, 41.0,
    boxstyle="round,pad=0.2,rounding_size=1.0",
    facecolor='#FAFAFA', edgecolor='#E2E8F0', linewidth=1.2, linestyle='--', zorder=1
)
ax.add_patch(rect_p2)

# Symmetrical 1-to-3 Fork from Checkpoint theta_1
draw_conn(ax, [(30.8, Y_MID), (31.8, Y_MID)], arrow=False)
draw_conn(ax, [(31.8, Y_MID), (31.8, Y_TOP), (33.0, Y_TOP)], arrow=True)
draw_conn(ax, [(31.8, Y_MID), (33.0, Y_MID)], arrow=True)
draw_conn(ax, [(31.8, Y_MID), (31.8, Y_BOT), (33.0, Y_BOT)], arrow=True)
ax.plot(31.8, Y_MID, 'o', color=C_LINE, markersize=5.0, zorder=4)

# Function to draw Phase 2 Card
def draw_phase2_card(ax, xy, width, height, title, data_str, head_str, reset_str, probe_str, sub_probe, fill, border, text_color):
    rect = patches.FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.2,rounding_size=0.8",
        facecolor=fill, edgecolor=border, linewidth=1.5, zorder=3
    )
    ax.add_patch(rect)
    cx = xy[0] + width / 2.0
    
    # Title at the very top
    ax.text(cx, xy[1] + height - 1.5, title, ha='center', va='center', fontsize=10.2, fontweight='bold', color=text_color, zorder=4)
    ax.plot([xy[0] + 0.8, xy[0] + width - 0.8], [xy[1] + height - 2.6, xy[1] + height - 2.6], color=border, lw=0.8, alpha=0.5, zorder=4)
    
    # Specs
    ax.text(xy[0] + 0.8, xy[1] + height - 4.0, f"Data: {data_str}", fontsize=8.6, fontweight='bold', color=C_DARK, zorder=4)
    ax.text(xy[0] + 0.8, xy[1] + height - 5.5, f"Head: {head_str}", fontsize=8.4, color=C_TEXT, zorder=4)
    ax.text(xy[0] + 0.8, xy[1] + height - 7.0, f"Resets: {reset_str}", fontsize=8.4, color=C_TEXT, zorder=4)
    
    # Highlight Result Badge at Bottom
    rect_b = patches.FancyBboxPatch(
        (xy[0] + 0.6, xy[1] + 0.7), width - 1.2, 3.0,
        boxstyle="round,pad=0.1,rounding_size=0.6",
        facecolor='#FFFFFF', edgecolor=border, linewidth=1.2, zorder=4
    )
    ax.add_patch(rect_b)
    ax.text(cx, xy[1] + 2.6, probe_str, ha='center', va='center', fontsize=9.2, fontweight='bold', color=text_color, zorder=5)
    ax.text(cx, xy[1] + 1.3, sub_probe, ha='center', va='center', fontsize=7.8, color=C_MUTED, zorder=5)

# C3 (Top)
draw_phase2_card(ax, (33.0, Y_TOP - TRACK_H/2.0), 19.8, TRACK_H,
                 r"$C_3$: On-Target Reset Control",
                 "CIFAR-100 (Target) | 100 cls", r"Temp $h_{\rm temp}$ (Fresh init)", r"$\mathbf{v} \leftarrow \mathbf{0}$ at 15k, 20k",
                 r"Step 20k Probe: $\mathbf{61.03\%}$",
                 r"Procedural Reset Retention ($N=6$)",
                 fill=C_C3_FILL, border=C_C3_BORDER, text_color=C_C3_TEXT)

# C0 (Mid)
draw_phase2_card(ax, (33.0, Y_MID - TRACK_H/2.0), 19.8, TRACK_H,
                 r"$C_0$: Continuous Baseline",
                 "CIFAR-100 (Target) | 100 cls", r"Target $h_A$ Active Throughout", r"None (Continuous Momentum $\mathbf{v}$)",
                 r"Step 20k Probe: $\mathbf{61.24\%}$",
                 r"Continuous Benchmark ($N=6$)",
                 fill=C_C0_FILL, border=C_C0_BORDER, text_color=C_C0_TEXT)

# S1 (Bot)
draw_phase2_card(ax, (33.0, Y_BOT - TRACK_H/2.0), 19.8, TRACK_H,
                 r"$S_1$: Auxiliary Detour Interval",
                 "STL-10 (Auxiliary) | 10 cls", r"Aux $h_B$ (Fresh init)", r"$\mathbf{v} \leftarrow \mathbf{0}$ at 15k, 20k",
                 r"Shock Probe: $\mathbf{20.65\%}$",
                 r"Target Feature Amnesia ($-40.59$ pp)",
                 fill=C_S1_FILL, border=C_S1_BORDER, text_color=C_S1_TEXT)

# =========================================================================
# --- STAGE 3: PHASE 3 RELEARNING (X: 54.2 to 75.2) ---
# =========================================================================
rect_p3 = patches.FancyBboxPatch(
    (54.2, 2.5), 21.0, 41.0,
    boxstyle="round,pad=0.2,rounding_size=1.0",
    facecolor='#FAFAFA', edgecolor='#E2E8F0', linewidth=1.2, linestyle='--', zorder=1
)
ax.add_patch(rect_p3)

# Connect Phase 2 to Phase 3
draw_conn(ax, [(52.8, Y_TOP), (54.8, Y_TOP)], arrow=True)
draw_conn(ax, [(52.8, Y_MID), (54.8, Y_MID)], arrow=True)
draw_conn(ax, [(52.8, Y_BOT), (54.8, Y_BOT)], arrow=True)

# Function to draw Phase 3 Card
def draw_phase3_card(ax, xy, width, height, title, sub_title, acc_str, probe_str, effrank_str, fill, border, text_color):
    rect = patches.FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.2,rounding_size=0.8",
        facecolor=fill, edgecolor=border, linewidth=1.5, zorder=3
    )
    ax.add_patch(rect)
    cx = xy[0] + width / 2.0
    
    # Title at the very top
    ax.text(cx, xy[1] + height - 1.5, title, ha='center', va='center', fontsize=10.2, fontweight='bold', color=text_color, zorder=4)
    ax.plot([xy[0] + 0.8, xy[0] + width - 0.8], [xy[1] + height - 2.6, xy[1] + height - 2.6], color=border, lw=0.8, alpha=0.5, zorder=4)
    
    # Subtitle protocol
    ax.text(cx, xy[1] + height - 3.7, sub_title, ha='center', va='center', fontsize=8.2, color=C_MUTED, zorder=4)
    
    # Test Accuracy Box (Terminal Outcome)
    rect_acc = patches.FancyBboxPatch(
        (xy[0] + 0.6, xy[1] + 4.2), width - 1.2, 3.0,
        boxstyle="round,pad=0.1,rounding_size=0.6",
        facecolor='#FFFFFF', edgecolor=border, linewidth=1.3, zorder=4
    )
    ax.add_patch(rect_acc)
    ax.text(cx, xy[1] + 6.0, acc_str, ha='center', va='center', fontsize=9.4, fontweight='bold', color=C_DARK, zorder=5)
    ax.text(cx, xy[1] + 4.8, r"Terminal Test Acc ($N=20$ seeds)", ha='center', va='center', fontsize=7.6, color=C_MUTED, zorder=5)
    
    # Longitudinal Representation Box
    rect_rep = patches.FancyBboxPatch(
        (xy[0] + 0.6, xy[1] + 0.7), width - 1.2, 3.0,
        boxstyle="round,pad=0.1,rounding_size=0.6",
        facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1.0, zorder=4
    )
    ax.add_patch(rect_rep)
    ax.text(cx, xy[1] + 2.5, probe_str, ha='center', va='center', fontsize=8.6, fontweight='bold', color=text_color, zorder=5)
    ax.text(cx, xy[1] + 1.3, effrank_str, ha='center', va='center', fontsize=8.0, color=C_TEXT, zorder=5)

# C3 Terminal
draw_phase3_card(ax, (54.8, Y_TOP - TRACK_H/2.0), 19.8, TRACK_H,
                 r"$C_3$: Relearning Terminal (35k)",
                 r"Target $h_A$ Restored $\cdot$ Momentum Reset",
                 r"Acc: $\mathbf{56.29\%}$ ($\pm 1.59\%$)",
                 r"Probe: $\mathbf{62.52\%}$ ($N=6$)",
                 r"Effective Rank: $\mathbf{162.37}$",
                 fill=C_C3_FILL, border=C_C3_BORDER, text_color=C_C3_TEXT)

# C0 Terminal
draw_phase3_card(ax, (54.8, Y_MID - TRACK_H/2.0), 19.8, TRACK_H,
                 r"$C_0$: Continuous Terminal (35k)",
                 r"Continuous Target SGD to 35,000 updates",
                 r"Acc: $\mathbf{56.69\%}$ ($\pm 0.75\%$)",
                 r"Probe: $\mathbf{63.01\%}$ ($N=6$)",
                 r"Effective Rank: $\mathbf{162.45}$",
                 fill=C_C0_FILL, border=C_C0_BORDER, text_color=C_C0_TEXT)

# S1 Terminal
draw_phase3_card(ax, (54.8, Y_BOT - TRACK_H/2.0), 19.8, TRACK_H,
                 r"$S_1$: Auxiliary Terminal (35k)",
                 r"Target $h_A$ Restored $\cdot$ Momentum Reset",
                 r"Acc: $\mathbf{56.26\%}$ ($\pm 1.51\%$)",
                 r"Probe: $\mathbf{61.96\%}$ ($N=6$)",
                 r"Effective Rank: $\mathbf{153.69}$",
                 fill=C_S1_FILL, border=C_S1_BORDER, text_color=C_S1_TEXT)

# =========================================================================
# --- STAGE 4: EVALUATIVE SYNTHESIS & CONTRASTS (X: 76.0 to 99.0) ---
# =========================================================================
# Connect Phase 3 into Synthesis Panels
draw_conn(ax, [(74.6, Y_TOP), (76.8, Y_TOP)], arrow=True)
draw_conn(ax, [(74.6, Y_MID), (76.8, Y_MID)], arrow=True)
draw_conn(ax, [(74.6, Y_BOT), (76.8, Y_BOT)], arrow=True)

# Helper: Draw Synthesis Panel
def draw_syn_panel(ax, xy, width, height, title, subtitle, estimand_math, stat_str, verdict_str, bg_fill, border_color, verdict_color):
    rect = patches.FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.2,rounding_size=0.8",
        facecolor=bg_fill, edgecolor=border_color, linewidth=1.5, zorder=3
    )
    ax.add_patch(rect)
    cx = xy[0] + width / 2.0
    
    # Title & Subtitle at top
    ax.text(cx, xy[1] + height - 1.5, title, ha='center', va='center', fontsize=10.0, fontweight='bold', color=C_DARK, zorder=4)
    ax.text(cx, xy[1] + height - 2.8, subtitle, ha='center', va='center', fontsize=8.0, color=C_MUTED, zorder=4)
    
    # Estimand Box
    rect_est = patches.FancyBboxPatch(
        (xy[0] + 0.8, xy[1] + 3.2), width - 1.6, 5.0,
        boxstyle="round,pad=0.1,rounding_size=0.6",
        facecolor='#FFFFFF', edgecolor=border_color, linewidth=1.2, zorder=4
    )
    ax.add_patch(rect_est)
    ax.text(cx, xy[1] + 6.8, estimand_math, ha='center', va='center', fontsize=9.4, fontweight='bold', color=C_DARK, zorder=5)
    ax.text(cx, xy[1] + 5.2, stat_str, ha='center', va='center', fontsize=8.2, color=C_TEXT, zorder=5)
    ax.text(cx, xy[1] + 3.9, r"Paired $t$-test ($N=20$ completed seeds)", ha='center', va='center', fontsize=7.4, color=C_MUTED, zorder=5)
    
    # Verdict Pill at bottom
    rect_v = patches.FancyBboxPatch(
        (xy[0] + 0.8, xy[1] + 0.7), width - 1.6, 2.0,
        boxstyle="round,pad=0.1,rounding_size=0.6",
        facecolor=verdict_color, edgecolor=verdict_color, linewidth=1.0, zorder=4
    )
    ax.add_patch(rect_v)
    ax.text(cx, xy[1] + 1.7, verdict_str, ha='center', va='center', fontsize=8.8, fontweight='bold', color='#FFFFFF', zorder=5)

# Panel 1: Hypothesis H1 (Auxiliary vs Continuous)
draw_syn_panel(ax, (76.8, Y_TOP - TRACK_H/2.0), 22.2, TRACK_H,
               "Primary Contrast: Hypothesis H1",
               r"$S_1 \text{ vs. } C_0 \text{ (Equal Budget, } 35\mathrm{k}\text{)}$",
               r"$\Delta_{\rm total} = \mathbf{-0.43\,\mathrm{pp}}$",
               r"$95\%\text{ CI: } [-0.94, +0.07] \mid p = 0.090$",
               "Statistically Uncertain (No Benefit)",
               bg_fill='#FFF5F5', border_color=C_RED_BORDER, verdict_color=C_RED_BORDER)

# Panel 2: Hypothesis H2 (Procedural Reset Control)
draw_syn_panel(ax, (76.8, Y_MID - TRACK_H/2.0), 22.2, TRACK_H,
               "Procedural Control: Hypothesis H2",
               r"$C_3 \text{ vs. } C_0 \text{ (Reset Shock, } 35\mathrm{k}\text{)}$",
               r"$\Delta_{\rm proc} = \mathbf{-0.40\,\mathrm{pp}}$",
               r"$95\%\text{ CI: } [-1.14, +0.34] \mid p = 0.269$",
               "Invariance Not Rejected",
               bg_fill='#FAF5FF', border_color=C_C3_BORDER, verdict_color=C_C3_BORDER)

# Panel 3: RQ3 Representation Recovery
rect_rq3 = patches.FancyBboxPatch(
    (76.8, Y_BOT - TRACK_H/2.0), 22.2, TRACK_H,
    boxstyle="round,pad=0.2,rounding_size=0.8",
    facecolor='#F0FDF4', edgecolor=C_GREEN_BORDER, linewidth=1.5, zorder=3
)
ax.add_patch(rect_rq3)
cx_rq3 = 76.8 + 22.2 / 2.0
ax.text(cx_rq3, Y_BOT - TRACK_H/2.0 + TRACK_H - 1.5, "Representation Dynamics: RQ3", ha='center', va='center', fontsize=10.0, fontweight='bold', color=C_DARK, zorder=4)
ax.text(cx_rq3, Y_BOT - TRACK_H/2.0 + TRACK_H - 2.8, r"Longitudinal Probe & Rank Cohort ($N=6$)", ha='center', va='center', fontsize=8.0, color=C_MUTED, zorder=4)

rect_rq3_est = patches.FancyBboxPatch(
    (76.8 + 0.8, Y_BOT - TRACK_H/2.0 + 3.2), 22.2 - 1.6, 5.0,
    boxstyle="round,pad=0.1,rounding_size=0.6",
    facecolor='#FFFFFF', edgecolor=C_GREEN_BORDER, linewidth=1.2, zorder=4
)
ax.add_patch(rect_rq3_est)
ax.text(cx_rq3, Y_BOT - TRACK_H/2.0 + 6.8, r"Probe Decodability: $20.65\% \to \mathbf{61.96\%}$", ha='center', va='center', fontsize=8.8, fontweight='bold', color=C_DARK, zorder=5)
ax.text(cx_rq3, Y_BOT - TRACK_H/2.0 + 5.2, r"Effective Rank: $124.50 \to \mathbf{153.69}$", ha='center', va='center', fontsize=8.6, color=C_TEXT, zorder=5)
ax.text(cx_rq3, Y_BOT - TRACK_H/2.0 + 3.9, r"Rapid Recovery Upon Re-Exposure", ha='center', va='center', fontsize=7.8, color=C_MUTED, zorder=5)

rect_rq3_v = patches.FancyBboxPatch(
    (76.8 + 0.8, Y_BOT - TRACK_H/2.0 + 0.7), 22.2 - 1.6, 2.0,
    boxstyle="round,pad=0.1,rounding_size=0.6",
    facecolor=C_GREEN_BORDER, edgecolor=C_GREEN_BORDER, linewidth=1.0, zorder=4
)
ax.add_patch(rect_rq3_v)
ax.text(cx_rq3, Y_BOT - TRACK_H/2.0 + 1.7, "Full Target-Feature Rebound", ha='center', va='center', fontsize=8.8, fontweight='bold', color='#FFFFFF', zorder=5)

# Save figures
png_out = os.path.join(FIGURES_DIR, "figure_edafnet_style_framework.png")
pdf_out = os.path.join(FIGURES_DIR, "figure_edafnet_style_framework.pdf")

plt.tight_layout()
plt.savefig(png_out, dpi=300, bbox_inches='tight')
plt.savefig(pdf_out, bbox_inches='tight')
plt.close()

print(f"✓ Saved refined figure PNG: {png_out} ({os.path.getsize(png_out)/1024:.1f} KB)")
print(f"✓ Saved refined figure PDF: {pdf_out} ({os.path.getsize(pdf_out)/1024:.1f} KB)")
