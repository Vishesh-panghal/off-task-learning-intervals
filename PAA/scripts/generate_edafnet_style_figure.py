#!/usr/bin/env python3
"""
Generate Mermaid-style Figure 1:
- Smooth cubic-Bezier S-curve connectors
- Distinct Mermaid node headers (pill header bands with accent tints)
- Metric badge chips (rounded pill badges for key results)
- Subgraph column containers with clean header badges
- Crisp high-contrast typography, zero nested rectangles
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Typography
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
        sample_images = [raw_data[4], raw_data[17]]
else:
    sample_images = [
        np.ones((32, 32, 3), dtype=np.uint8) * 130,
        np.ones((32, 32, 3), dtype=np.uint8) * 170
    ]

# 2. Canvas Setup (14.4 x 9.2 inches -> 1.56:1 aspect ratio)
fig, ax = plt.subplots(figsize=(14.4, 9.2), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis('off')

# Color Palette: Clean Mermaid Modern Palette
C_DARK = '#0F172A'       # Slate 900
C_TEXT = '#1E293B'       # Slate 800
C_MUTED = '#475569'      # Slate 600
C_BORDER = '#CBD5E1'     # Slate 300
C_LINE = '#475569'       # Slate 600 (flow lines)

# Arm Colors (Header Tint, Border, Text)
# C3 Purple
BG_C3 = '#F5F3FF'        # Purple 50
HDR_C3 = '#EDE9FE'       # Purple 100
BDR_C3 = '#7E22CE'       # Purple 700
TXT_C3 = '#581C87'       # Purple 900

# C0 Sky Blue
BG_C0 = '#F0F9FF'        # Sky 50
HDR_C0 = '#E0F2FE'       # Sky 100
BDR_C0 = '#0284C7'       # Sky 600
TXT_C0 = '#0369A1'       # Sky 800

# S1 Orange
BG_S1 = '#FFF7ED'        # Orange 50
HDR_S1 = '#FFEDD5'       # Orange 100
BDR_S1 = '#EA580C'       # Orange 600
TXT_S1 = '#9A3412'       # Orange 900

# Synthesis Contrasts
# H1 Red
BG_H1 = '#FEF2F2'
HDR_H1 = '#FEE2E2'
BDR_H1 = '#DC2626'
TXT_H1 = '#991B1B'

# RQ3 Green
BG_RQ3 = '#F0FDF4'
HDR_RQ3 = '#DCFCE7'
BDR_RQ3 = '#16A34A'
TXT_RQ3 = '#166534'

# Neutral Cards
BG_NEUT = '#F8FAFC'
HDR_NEUT = '#F1F5F9'
BDR_NEUT = '#64748B'
TXT_NEUT = '#0F172A'

# 4 Column Geometry
COL_W = 22.8
X_COL1 = 1.6
X_COL2 = 26.4
X_COL3 = 51.0
X_COL4 = 75.6

# 3 Track Centers
TRACK_H = 15.2
Y_TOP = 43.8   # Track 1 Center
Y_MID = 26.8   # Track 2 Center
Y_BOT = 9.8    # Track 3 Center

# Helper: Draw Cubic-Bezier Spline Connection (Mermaid S-curve)
def draw_bezier_conn(ax, p1, p2, color=C_LINE, lw=1.8, arrow=True):
    x1, y1 = p1
    x2, y2 = p2
    dx = (x2 - x1) * 0.52
    ctrl1 = (x1 + dx, y1)
    ctrl2 = (x2 - dx, y2)
    path = Path([p1, ctrl1, ctrl2, p2], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    patch = patches.PathPatch(path, facecolor='none', edgecolor=color, lw=lw, zorder=2)
    ax.add_patch(patch)
    if arrow:
        ax.annotate(
            '', xy=p2, xytext=(p2[0] - 0.15, p2[1]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=13.0),
            zorder=3
        )

# Helper: Draw Mermaid Subgraph Container
def draw_subgraph(ax, x, width, title, subtitle):
    # Outer container
    rect = patches.FancyBboxPatch(
        (x, 1.8), width, 57.0,
        boxstyle="round,pad=0.2,rounding_size=1.0",
        facecolor='#FAFAFA', edgecolor='#E2E8F0', linewidth=1.2, linestyle='--', zorder=1
    )
    ax.add_patch(rect)
    
    # Subgraph Header Tab (Mermaid pill style)
    hdr_box = patches.FancyBboxPatch(
        (x + 0.8, 54.0), width - 1.6, 4.4,
        boxstyle="round,pad=0.15,rounding_size=0.6",
        facecolor='#F1F5F9', edgecolor='#CBD5E1', linewidth=1.0, zorder=2
    )
    ax.add_patch(hdr_box)
    
    cx = x + width / 2.0
    ax.text(cx, 56.6, title, ha='center', va='center', fontsize=13.0, fontweight='bold', color=C_DARK, zorder=3)
    ax.text(cx, 54.9, subtitle, ha='center', va='center', fontsize=10.2, fontweight='medium', color=C_MUTED, zorder=3)

# Helper: Draw Metric Badge Pill
def draw_badge_pill(ax, cx, cy, width, height, text, bg_color, border_color, text_color, fontsize=12.5):
    pill = patches.FancyBboxPatch(
        (cx - width/2.0, cy - height/2.0), width, height,
        boxstyle="round,pad=0.1,rounding_size=0.6",
        facecolor=bg_color, edgecolor=border_color, linewidth=1.1, zorder=4
    )
    ax.add_patch(pill)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=text_color, zorder=5)

# =========================================================================
# --- SUBGRAPH CONTAINERS ---
# =========================================================================
draw_subgraph(ax, X_COL1, COL_W, "Phase 1: Target Exposure", "0 to 15,000 updates | CIFAR-100")
draw_subgraph(ax, X_COL2, COL_W, "Phase 2: Interruption", "15,000 to 20,000 updates | 3 Arms")
draw_subgraph(ax, X_COL3, COL_W, "Phase 3: Relearning", "20,000 to 35,000 updates | Budget = 35k")
draw_subgraph(ax, X_COL4, COL_W, "Confirmatory Synthesis", "Hypotheses H1, H2 & RQ3")

# =========================================================================
# --- COLUMN 1: PHASE 1 (TARGET TRAINING & CHECKPOINT) ---
# =========================================================================
cx1 = X_COL1 + COL_W / 2.0
card_w = COL_W - 1.6

# Top Card: Initial Target Training (Y = 28.4 to 52.4, Height = 24.0)
rect_p1_top = patches.FancyBboxPatch(
    (X_COL1 + 0.8, 28.4), card_w, 24.0,
    boxstyle="round,pad=0.2,rounding_size=0.8",
    facecolor='#FFFFFF', edgecolor=BDR_NEUT, linewidth=1.4, zorder=3
)
ax.add_patch(rect_p1_top)

# Header Pill
hdr_p1_top = patches.FancyBboxPatch(
    (X_COL1 + 1.4, 49.0), card_w - 1.2, 2.8,
    boxstyle="round,pad=0.1,rounding_size=0.5",
    facecolor=HDR_NEUT, edgecolor=BDR_NEUT, linewidth=1.0, zorder=4
)
ax.add_patch(hdr_p1_top)
ax.text(cx1, 50.4, "Initial Target Training", ha='center', va='center', fontsize=12.5, fontweight='bold', color=C_DARK, zorder=5)

ax.text(cx1, 46.5, "Data: CIFAR-100 (100 classes)", ha='center', va='center', fontsize=11.4, fontweight='bold', color=C_TEXT, zorder=4)
ax.text(cx1, 44.2, r"ResNet-18 $\cdot$ GroupNorm ($G=32$)", ha='center', va='center', fontsize=11.0, color=C_TEXT, zorder=4)
ax.text(cx1, 42.0, "Feature dim: d = 512 (Zero batch drift)", ha='center', va='center', fontsize=10.6, color=C_TEXT, zorder=4)
ax.text(cx1, 39.8, r"SGD: $\eta = 0.1, \mu = 0.9$ $\cdot$ 15k updates", ha='center', va='center', fontsize=10.6, color=C_MUTED, zorder=4)

ab1 = AnnotationBbox(OffsetImage(sample_images[0], zoom=1.55), (cx1 - 3.2, 34.2), frameon=True, pad=0.08,
                     bboxprops=dict(edgecolor=BDR_C0, linewidth=1.0, facecolor='white'), zorder=5)
ab2 = AnnotationBbox(OffsetImage(sample_images[1], zoom=1.55), (cx1 + 3.2, 34.2), frameon=True, pad=0.08,
                     bboxprops=dict(edgecolor=BDR_C0, linewidth=1.0, facecolor='white'), zorder=5)
ax.add_artist(ab1); ax.add_artist(ab2)
ax.text(cx1, 30.8, "CIFAR-100 Sample Inputs", ha='center', va='center', fontsize=9.4, color=C_MUTED, zorder=4)

# Vertical Arrow to Checkpoint with clear shaft
draw_bezier_conn(ax, (cx1, 28.4), (cx1, 25.0), lw=1.8)

# Bottom Card: Checkpoint theta_1 (Y = 2.0 to 25.0, Height = 23.0)
rect_p1_bot = patches.FancyBboxPatch(
    (X_COL1 + 0.8, 2.0), card_w, 23.0,
    boxstyle="round,pad=0.2,rounding_size=0.8",
    facecolor='#FFFFFF', edgecolor=BDR_S1, linewidth=1.5, zorder=3
)
ax.add_patch(rect_p1_bot)

hdr_p1_bot = patches.FancyBboxPatch(
    (X_COL1 + 1.4, 21.6), card_w - 1.2, 2.8,
    boxstyle="round,pad=0.1,rounding_size=0.5",
    facecolor=HDR_S1, edgecolor=BDR_S1, linewidth=1.0, zorder=4
)
ax.add_patch(hdr_p1_bot)
ax.text(cx1, 23.0, r"Checkpoint $\theta_1$ (Step 15k)", ha='center', va='center', fontsize=12.5, fontweight='bold', color=TXT_S1, zorder=5)

ax.text(cx1, 19.6, "Shared Origin for All 3 Arms", ha='center', va='center', fontsize=11.4, fontweight='bold', color=C_DARK, zorder=4)
ax.text(cx1, 17.4, r"Target Head $h_A$ Stashed for Phase 3", ha='center', va='center', fontsize=10.8, color=C_TEXT, zorder=4)

# Badge Pill for Probe Decodability
draw_badge_pill(ax, cx1, 13.4, card_w - 2.8, 3.2, "Probe Decodability: 60.07%", HDR_S1, BDR_S1, TXT_S1, fontsize=13.0)

ax.text(cx1, 9.8, "Effective Rank: 178.04", ha='center', va='center', fontsize=12.2, fontweight='bold', color=C_DARK, zorder=4)
ax.text(cx1, 7.3, "(N = 6 Paired Cohort Baseline)", ha='center', va='center', fontsize=10.0, color=C_MUTED, zorder=4)
ax.text(cx1, 4.8, "Full Target Decodability Pre-Shock", ha='center', va='center', fontsize=9.4, color=C_MUTED, zorder=4)

# =========================================================================
# --- MERMAID S-CURVE FORK INTO PHASE 2 ---
# =========================================================================
# Route from Checkpoint theta_1 right edge through an S-curve fan-out
P1_EXIT = (X_COL1 + card_w + 0.8, 13.8)
FORK_X = 24.8
FORK_MID = (FORK_X, Y_MID)

# Trunk from Checkpoint to Fork center
draw_bezier_conn(ax, P1_EXIT, FORK_MID, color=C_LINE, lw=1.8, arrow=False)

# Milestone dot / badge at fork
ax.plot(FORK_X, Y_MID, 'o', color=C_LINE, markersize=5.0, zorder=4)

# Smooth cubic-Bezier splines to 3 arms in Phase 2
P2_ENTRY_C3 = (X_COL2 + 0.8, Y_TOP)
P2_ENTRY_C0 = (X_COL2 + 0.8, Y_MID)
P2_ENTRY_S1 = (X_COL2 + 0.8, Y_BOT)

draw_bezier_conn(ax, FORK_MID, P2_ENTRY_C3, color=BDR_C3, lw=1.8, arrow=True)
draw_bezier_conn(ax, FORK_MID, P2_ENTRY_C0, color=BDR_C0, lw=1.8, arrow=True)
draw_bezier_conn(ax, FORK_MID, P2_ENTRY_S1, color=BDR_S1, lw=1.8, arrow=True)

# Helper: Draw Mermaid Node Card
def draw_mermaid_node(ax, x, y_center, title, data_str, head_str, reset_str, badge_text, note_str,
                      hdr_bg, border_color, text_color):
    y_bot = y_center - TRACK_H / 2.0
    rect = patches.FancyBboxPatch(
        (x + 0.8, y_bot), card_w, TRACK_H,
        boxstyle="round,pad=0.2,rounding_size=0.8",
        facecolor='#FFFFFF', edgecolor=border_color, linewidth=1.5, zorder=3
    )
    ax.add_patch(rect)
    cx = x + COL_W / 2.0
    
    # Header Pill
    hdr = patches.FancyBboxPatch(
        (x + 1.4, y_bot + TRACK_H - 3.2), card_w - 1.2, 2.6,
        boxstyle="round,pad=0.1,rounding_size=0.5",
        facecolor=hdr_bg, edgecolor=border_color, linewidth=1.0, zorder=4
    )
    ax.add_patch(hdr)
    ax.text(cx, y_bot + TRACK_H - 1.9, title, ha='center', va='center', fontsize=12.5, fontweight='bold', color=text_color, zorder=5)
    
    # Body lines
    ax.text(cx, y_bot + TRACK_H - 4.9, data_str, ha='center', va='center', fontsize=11.2, fontweight='bold', color=C_DARK, zorder=4)
    ax.text(cx, y_bot + TRACK_H - 6.8, head_str, ha='center', va='center', fontsize=10.6, color=C_TEXT, zorder=4)
    if reset_str:
        ax.text(cx, y_bot + TRACK_H - 8.6, reset_str, ha='center', va='center', fontsize=10.4, color=C_TEXT, zorder=4)
    
    # Metric Badge Pill
    badge_y = y_bot + 3.2
    draw_badge_pill(ax, cx, badge_y, card_w - 2.8, 2.8, badge_text, hdr_bg, border_color, text_color, fontsize=12.5)
    
    # Subtitle note
    ax.text(cx, y_bot + 1.1, note_str, ha='center', va='center', fontsize=9.6, color=C_MUTED, zorder=4)

# =========================================================================
# --- COLUMN 2: PHASE 2 NODES ---
# =========================================================================
# C3 (Top)
draw_mermaid_node(ax, X_COL2, Y_TOP,
                  r"$C_3$: On-Target Reset Control",
                  "Data: CIFAR-100 (Target)",
                  r"Head: Temp $h_{\rm temp}$ (Fresh init)",
                  r"Velocity Reset: $\mathbf{v} \leftarrow \mathbf{0}$ at 15k, 20k",
                  "Step 20k Probe: 61.03%",
                  "Procedural Control (N = 6)",
                  HDR_C3, BDR_C3, TXT_C3)

# C0 (Mid)
draw_mermaid_node(ax, X_COL2, Y_MID,
                  r"$C_0$: Continuous Baseline",
                  "Data: CIFAR-100 (Target)",
                  r"Head: Target $h_A$ Active Throughout",
                  "Optimizer: Momentum Preserved",
                  "Step 20k Probe: 61.24%",
                  "Continuous Benchmark (N = 6)",
                  HDR_C0, BDR_C0, TXT_C0)

# S1 (Bot)
draw_mermaid_node(ax, X_COL2, Y_BOT,
                  r"$S_1$: Auxiliary Detour Interval",
                  "Data: STL-10 (Auxiliary)",
                  r"Head: Aux $h_B$ (Fresh init)",
                  r"Velocity Reset: $\mathbf{v} \leftarrow \mathbf{0}$ at 15k, 20k",
                  "Shock Probe: 20.65%",
                  "Target Amnesia Shock (-40.59 pp)",
                  HDR_S1, BDR_S1, TXT_S1)

# =========================================================================
# --- COLUMN 3: PHASE 3 NODES ---
# =========================================================================
# Connect Phase 2 to Phase 3 with smooth lines
draw_bezier_conn(ax, (X_COL2 + card_w + 0.8, Y_TOP), (X_COL3 + 0.8, Y_TOP), color=BDR_C3, lw=1.8)
draw_bezier_conn(ax, (X_COL2 + card_w + 0.8, Y_MID), (X_COL3 + 0.8, Y_MID), color=BDR_C0, lw=1.8)
draw_bezier_conn(ax, (X_COL2 + card_w + 0.8, Y_BOT), (X_COL3 + 0.8, Y_BOT), color=BDR_S1, lw=1.8)

# Helper: Draw Phase 3 Relearning Node
def draw_phase3_node(ax, x, y_center, title, protocol_str, acc_str, probe_rank_str,
                     hdr_bg, border_color, text_color):
    y_bot = y_center - TRACK_H / 2.0
    rect = patches.FancyBboxPatch(
        (x + 0.8, y_bot), card_w, TRACK_H,
        boxstyle="round,pad=0.2,rounding_size=0.8",
        facecolor='#FFFFFF', edgecolor=border_color, linewidth=1.5, zorder=3
    )
    ax.add_patch(rect)
    cx = x + COL_W / 2.0
    
    # Header Pill
    hdr = patches.FancyBboxPatch(
        (x + 1.4, y_bot + TRACK_H - 3.2), card_w - 1.2, 2.6,
        boxstyle="round,pad=0.1,rounding_size=0.5",
        facecolor=hdr_bg, edgecolor=border_color, linewidth=1.0, zorder=4
    )
    ax.add_patch(hdr)
    ax.text(cx, y_bot + TRACK_H - 1.9, title, ha='center', va='center', fontsize=12.5, fontweight='bold', color=text_color, zorder=5)
    
    # Protocol line
    ax.text(cx, y_bot + TRACK_H - 4.9, protocol_str, ha='center', va='center', fontsize=11.0, color=C_MUTED, zorder=4)
    
    # Big Test Accuracy Pill Badge
    draw_badge_pill(ax, cx, y_bot + TRACK_H - 8.2, card_w - 2.8, 3.2, acc_str, hdr_bg, border_color, C_DARK, fontsize=13.5)
    ax.text(cx, y_bot + TRACK_H - 10.4, "Terminal Test Acc (N = 20 paired seeds)", ha='center', va='center', fontsize=9.8, color=C_MUTED, zorder=4)
    
    # Longitudinal Representation Line
    ax.plot([x + 2.0, x + COL_W - 2.0], [y_bot + 3.8, y_bot + 3.8], color=C_BORDER, lw=0.8, alpha=0.5, zorder=4)
    ax.text(cx, y_bot + 2.0, probe_rank_str, ha='center', va='center', fontsize=11.2, fontweight='bold', color=text_color, zorder=4)

# C3 Terminal
draw_phase3_node(ax, X_COL3, Y_TOP,
                 r"$C_3$: Relearning Terminal",
                 r"Restored Target Head $h_A$ $\cdot$ Relearning",
                 "Acc: 56.29% (±1.59%)",
                 "Probe: 62.52%  |  EffRank: 162.37",
                 HDR_C3, BDR_C3, TXT_C3)

# C0 Terminal
draw_phase3_node(ax, X_COL3, Y_MID,
                 r"$C_0$: Continuous Terminal",
                 "Continuous Target Training to 35k",
                 "Acc: 56.69% (±0.75%)",
                 "Probe: 63.01%  |  EffRank: 162.45",
                 HDR_C0, BDR_C0, TXT_C0)

# S1 Terminal
draw_phase3_node(ax, X_COL3, Y_BOT,
                 r"$S_1$: Auxiliary Terminal",
                 r"Restored Target Head $h_A$ $\cdot$ Relearning",
                 "Acc: 56.26% (±1.51%)",
                 "Probe: 61.96%  |  EffRank: 153.69",
                 HDR_S1, BDR_S1, TXT_S1)

# =========================================================================
# --- COLUMN 4: CONFIRMATORY SYNTHESIS NODES ---
# =========================================================================
# Connect Phase 3 into Synthesis Nodes
draw_bezier_conn(ax, (X_COL3 + card_w + 0.8, Y_TOP), (X_COL4 + 0.8, Y_TOP), color=BDR_H1, lw=1.8)
draw_bezier_conn(ax, (X_COL3 + card_w + 0.8, Y_MID), (X_COL4 + 0.8, Y_MID), color=BDR_C3, lw=1.8)
draw_bezier_conn(ax, (X_COL3 + card_w + 0.8, Y_BOT), (X_COL4 + 0.8, Y_BOT), color=BDR_RQ3, lw=1.8)

# Helper: Draw Synthesis Panel
def draw_synthesis_node(ax, x, y_center, title, subtitle, estimand_math, stat_str, verdict_l1, verdict_l2,
                        hdr_bg, border_color, text_color):
    y_bot = y_center - TRACK_H / 2.0
    rect = patches.FancyBboxPatch(
        (x + 0.8, y_bot), card_w, TRACK_H,
        boxstyle="round,pad=0.2,rounding_size=0.8",
        facecolor='#FFFFFF', edgecolor=border_color, linewidth=1.5, zorder=3
    )
    ax.add_patch(rect)
    cx = x + COL_W / 2.0
    
    # Header Pill
    hdr = patches.FancyBboxPatch(
        (x + 1.4, y_bot + TRACK_H - 3.2), card_w - 1.2, 2.6,
        boxstyle="round,pad=0.1,rounding_size=0.5",
        facecolor=hdr_bg, edgecolor=border_color, linewidth=1.0, zorder=4
    )
    ax.add_patch(hdr)
    ax.text(cx, y_bot + TRACK_H - 1.9, title, ha='center', va='center', fontsize=12.2, fontweight='bold', color=text_color, zorder=5)
    
    # Subtitle
    ax.text(cx, y_bot + TRACK_H - 4.7, subtitle, ha='center', va='center', fontsize=10.4, color=C_MUTED, zorder=4)
    
    # Estimand Badge Pill
    draw_badge_pill(ax, cx, y_bot + TRACK_H - 7.6, card_w - 2.8, 2.8, estimand_math, hdr_bg, border_color, C_DARK, fontsize=13.5)
    ax.text(cx, y_bot + TRACK_H - 9.8, stat_str, ha='center', va='center', fontsize=10.8, color=C_TEXT, zorder=4)
    
    # Verdict text lines at bottom
    ax.plot([x + 2.0, x + COL_W - 2.0], [y_bot + 4.2, y_bot + 4.2], color=C_BORDER, lw=0.8, alpha=0.5, zorder=4)
    ax.text(cx, y_bot + 2.8, verdict_l1, ha='center', va='center', fontsize=11.4, fontweight='bold', color=text_color, zorder=4)
    ax.text(cx, y_bot + 1.2, verdict_l2, ha='center', va='center', fontsize=10.0, color=text_color, zorder=4)

# Panel 1: Hypothesis H1 (Auxiliary vs Continuous)
draw_synthesis_node(ax, X_COL4, Y_TOP,
                    "Primary Contrast: Hypothesis H1",
                    r"$S_1 \text{ vs. } C_0 \text{ (Budget 35k, } N=20\text{)}$",
                    r"$\Delta_{\rm total} = \mathbf{-0.43\,\mathrm{pp}}$",
                    r"$95\%\text{ CI: } [-0.94, +0.07] \mid p = 0.090$",
                    "Result: Statistically Uncertain",
                    "(No Generalization Benefit)",
                    HDR_H1, BDR_H1, TXT_H1)

# Panel 2: Hypothesis H2 (Procedural Reset Control)
draw_synthesis_node(ax, X_COL4, Y_MID,
                    "Procedural Control: Hypothesis H2",
                    r"$C_3 \text{ vs. } C_0 \text{ (Reset Shock, } N=20\text{)}$",
                    r"$\Delta_{\rm proc} = \mathbf{-0.40\,\mathrm{pp}}$",
                    r"$95\%\text{ CI: } [-1.14, +0.34] \mid p = 0.269$",
                    "Result: Invariance Not Rejected",
                    "(Reset Impact Overcome by 35k)",
                    HDR_C3, BDR_C3, TXT_C3)

# Panel 3: RQ3 Representation Recovery
draw_synthesis_node(ax, X_COL4, Y_BOT,
                    "Representation Dynamics: RQ3",
                    "Longitudinal Cohort (N = 6 paired seeds)",
                    r"Probe: $20.65\% \to \mathbf{61.96\%}$",
                    r"EffRank: $124.50 \to \mathbf{153.69}$",
                    "Result: Full Feature Rebound",
                    "(Decodability Fully Restored)",
                    HDR_RQ3, BDR_RQ3, TXT_RQ3)

# Save figures with tightly trimmed padding
png_out = os.path.join(FIGURES_DIR, "figure_edafnet_style_framework.png")
pdf_out = os.path.join(FIGURES_DIR, "figure_edafnet_style_framework.pdf")

plt.tight_layout()
plt.savefig(png_out, dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.savefig(pdf_out, bbox_inches='tight', pad_inches=0.01)
plt.close()

print(f"✓ Saved Mermaid-style figure PNG: {png_out} ({os.path.getsize(png_out)/1024:.1f} KB)")
print(f"✓ Saved Mermaid-style figure PDF: {pdf_out} ({os.path.getsize(pdf_out)/1024:.1f} KB)")
