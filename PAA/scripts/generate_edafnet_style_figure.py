#!/usr/bin/env python3
"""
Generate the redesigned Figure 1 adhering strictly to peer-review requirements:
1. Increase internal font size by ~30–40% (Phase labels, accuracy values, statistical results legible).
2. Reduce information inside each box by 25–30% (retain arm, intervention, interval, principal result).
3. Use more vertical space (taller boxes, 1.54:1 aspect ratio, 4 balanced columns).
4. Remove unnecessary nested rectangles (clean single card per track, high typography hierarchy).
5. Trim white space with tight bounding box (pad_inches=0.01).
6. High contrast: white backgrounds, thin colored borders, dark text, no saturated color bars.
7. Uniform clean Helvetica typography throughout (no awkward font mixing on numbers).
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Set typography
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

# 2. Canvas Setup (14.2 x 9.2 inches -> 1.54:1 aspect ratio)
fig, ax = plt.subplots(figsize=(14.2, 9.2), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis('off')

# Color Palette: Clean, journal-grade, high-contrast
C_DARK = '#0F172A'       # Slate 900 (primary dark text)
C_TEXT = '#1E293B'       # Slate 800 (body text)
C_MUTED = '#475569'      # Slate 600 (subtitles and notes)
C_BORDER = '#CBD5E1'     # Slate 300 (subtle divider lines)
C_LINE = '#334155'       # Slate 700 (connector arrows)

# Arm border and accent colors (thin, elegant)
C_C3_BORDER = '#7E22CE'  # Purple 700
C_C3_TITLE = '#6B21A8'   # Purple 800

C_C0_BORDER = '#0284C7'  # Sky 600
C_C0_TITLE = '#0369A1'   # Sky 700

C_S1_BORDER = '#EA580C'  # Orange 600
C_S1_TITLE = '#C2410C'   # Orange 800

# Diagnostic & Contrast colors
C_H1_BORDER = '#DC2626'  # Red 600
C_H1_TEXT = '#B91C1C'

C_RQ3_BORDER = '#16A34A' # Green 600
C_RQ3_TEXT = '#15803D'

# 4 Balanced Columns Geometry (Width = 22.7 each, Gaps = 2.0)
COL_W = 22.7
X_COL1 = 1.6
X_COL2 = 26.3
X_COL3 = 51.0
X_COL4 = 75.7

# Vertical Track Dimensions (3 tracks in Cols 2, 3, 4)
TRACK_H = 15.4
Y_TOP = 43.9   # Track 1 Center (36.2 to 51.6)
Y_MID = 26.8   # Track 2 Center (19.1 to 34.5)
Y_BOT = 9.7    # Track 3 Center (2.0 to 17.4)

# Helper: Draw Phase Header Banner
def draw_phase_header(ax, x, width, title, subtitle):
    rect = patches.FancyBboxPatch(
        (x, 53.4), width, 5.2,
        boxstyle="round,pad=0.2,rounding_size=0.8",
        facecolor='#F8FAFC', edgecolor='#94A3B8', linewidth=1.2, zorder=2
    )
    ax.add_patch(rect)
    cx = x + width / 2.0
    ax.text(cx, 56.6, title, ha='center', va='center', fontsize=13.5, fontweight='bold', color=C_DARK, zorder=3)
    ax.text(cx, 54.7, subtitle, ha='center', va='center', fontsize=10.5, fontweight='medium', color=C_MUTED, zorder=3)

# Helper: Orthogonal connection
def draw_conn(ax, points, color=C_LINE, lw=1.8, arrow=True, style='-'):
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i+1]
        is_last = (i == len(points) - 2)
        if is_last and arrow:
            ax.annotate(
                '', xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=13.0),
                zorder=2
            )
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle=style, color=color, lw=lw, zorder=2)

# =========================================================================
# --- TOP PHASE HEADERS (Y = 53.4 to 58.6) ---
# =========================================================================
draw_phase_header(ax, X_COL1, COL_W, "Phase 1: Target Exposure", "0 to 15,000 updates | CIFAR-100")
draw_phase_header(ax, X_COL2, COL_W, "Phase 2: Interruption", "15,000 to 20,000 updates | 3 Arms")
draw_phase_header(ax, X_COL3, COL_W, "Phase 3: Relearning", "20,000 to 35,000 updates | Budget = 35k")
draw_phase_header(ax, X_COL4, COL_W, "Confirmatory Synthesis", "Hypotheses H1, H2 & RQ3")

# Soft background band per column (borderless, purely for visual grouping)
for col_x in [X_COL1, X_COL2, X_COL3, X_COL4]:
    bg = patches.FancyBboxPatch(
        (col_x, 2.0), COL_W, 49.6,
        boxstyle="round,pad=0.2,rounding_size=1.0",
        facecolor='#F8FAFC', edgecolor='none', zorder=1
    )
    ax.add_patch(bg)

# =========================================================================
# --- COLUMN 1: PHASE 1 TARGET EXPOSURE & CHECKPOINT ---
# =========================================================================
cx1 = X_COL1 + COL_W / 2.0

# Top Card: Initial Target Training (Y = 27.6 to 51.6, Height = 24.0)
rect_p1_top = patches.FancyBboxPatch(
    (X_COL1 + 0.8, 27.6), COL_W - 1.6, 23.2,
    boxstyle="round,pad=0.2,rounding_size=0.8",
    facecolor='#FFFFFF', edgecolor=C_LINE, linewidth=1.4, zorder=3
)
ax.add_patch(rect_p1_top)

ax.text(cx1, 49.0, "Initial Target Training", ha='center', va='center', fontsize=13.5, fontweight='bold', color=C_DARK, zorder=4)
ax.plot([X_COL1 + 2.0, X_COL1 + COL_W - 2.0], [47.4, 47.4], color=C_BORDER, lw=0.8, alpha=0.6, zorder=4)

ax.text(cx1, 45.2, "Data: CIFAR-100 (100 classes)", ha='center', va='center', fontsize=11.6, fontweight='bold', color=C_TEXT, zorder=4)
ax.text(cx1, 42.8, r"ResNet-18 $\cdot$ GroupNorm ($G=32$)", ha='center', va='center', fontsize=11.2, color=C_TEXT, zorder=4)
ax.text(cx1, 40.5, "Feature dim: d = 512 (Zero batch drift)", ha='center', va='center', fontsize=10.8, color=C_TEXT, zorder=4)
ax.text(cx1, 38.2, r"SGD: $\eta = 0.1, \mu = 0.9$ $\cdot$ 15k updates", ha='center', va='center', fontsize=10.8, color=C_MUTED, zorder=4)

# Embedded CIFAR-100 thumbnails
ab1 = AnnotationBbox(OffsetImage(sample_images[0], zoom=1.55), (cx1 - 3.2, 32.5), frameon=True, pad=0.08,
                     bboxprops=dict(edgecolor=C_C0_BORDER, linewidth=1.0, facecolor='white'), zorder=5)
ab2 = AnnotationBbox(OffsetImage(sample_images[1], zoom=1.55), (cx1 + 3.2, 32.5), frameon=True, pad=0.08,
                     bboxprops=dict(edgecolor=C_C0_BORDER, linewidth=1.0, facecolor='white'), zorder=5)
ax.add_artist(ab1); ax.add_artist(ab2)
ax.text(cx1, 29.2, "CIFAR-100 Sample Inputs", ha='center', va='center', fontsize=9.6, color=C_MUTED, zorder=4)

# Vertical Arrow from Top Card to Bottom Card
draw_conn(ax, [(cx1, 27.6), (cx1, 25.2)], arrow=True, lw=1.8)

# Bottom Card: Checkpoint theta_1 (Y = 2.0 to 25.0, Height = 23.0)
rect_p1_bot = patches.FancyBboxPatch(
    (X_COL1 + 0.8, 2.0), COL_W - 1.6, 23.0,
    boxstyle="round,pad=0.2,rounding_size=0.8",
    facecolor='#FFFFFF', edgecolor=C_S1_BORDER, linewidth=1.5, zorder=3
)
ax.add_patch(rect_p1_bot)

ax.text(cx1, 23.0, r"Checkpoint $\theta_1$ (Step 15k)", ha='center', va='center', fontsize=13.5, fontweight='bold', color=C_S1_TITLE, zorder=4)
ax.plot([X_COL1 + 2.0, X_COL1 + COL_W - 2.0], [21.4, 21.4], color=C_S1_BORDER, lw=0.8, alpha=0.5, zorder=4)

ax.text(cx1, 19.3, "Shared Origin for All 3 Arms", ha='center', va='center', fontsize=11.6, fontweight='bold', color=C_DARK, zorder=4)
ax.text(cx1, 17.0, r"Target Head $h_A$ Stashed for Phase 3", ha='center', va='center', fontsize=11.0, color=C_TEXT, zorder=4)

ax.plot([X_COL1 + 2.0, X_COL1 + COL_W - 2.0], [14.8, 14.8], color=C_BORDER, lw=0.8, alpha=0.6, zorder=4)

ax.text(cx1, 12.4, "Probe Decodability: 60.07%", ha='center', va='center', fontsize=14.0, fontweight='bold', color=C_S1_TITLE, zorder=4)
ax.text(cx1, 9.4, "Effective Rank: 178.04", ha='center', va='center', fontsize=12.5, fontweight='bold', color=C_DARK, zorder=4)
ax.text(cx1, 6.8, "(N = 6 Paired Cohort Baseline)", ha='center', va='center', fontsize=10.2, color=C_MUTED, zorder=4)
ax.text(cx1, 4.4, "Full Target Decodability Pre-Shock", ha='center', va='center', fontsize=9.6, color=C_MUTED, zorder=4)

# =========================================================================
# --- COLUMN 2: PHASE 2 SCHEDULE INTERRUPTION (3 ARMS) ---
# =========================================================================
# Clean fork from Checkpoint theta_1 into the 3 arms
X_FORK = 24.3
draw_conn(ax, [(X_COL1 + COL_W - 0.8, 13.5), (X_FORK, 13.5), (X_FORK, Y_MID)], arrow=False, lw=1.8)
draw_conn(ax, [(X_FORK, Y_MID), (X_FORK, Y_TOP), (X_COL2 + 0.8, Y_TOP)], arrow=True, lw=1.8)
draw_conn(ax, [(X_FORK, Y_MID), (X_COL2 + 0.8, Y_MID)], arrow=True, lw=1.8)
draw_conn(ax, [(X_FORK, Y_MID), (X_FORK, Y_BOT), (X_COL2 + 0.8, Y_BOT)], arrow=True, lw=1.8)
ax.plot(X_FORK, Y_MID, 'o', color=C_LINE, markersize=4.5, zorder=4)

# Function to draw Phase 2 Card (White background, single card, NO nested rectangles)
def draw_phase2_card(ax, x, y_center, title, data_str, head_str, reset_str, probe_str, note_str, border, title_color):
    y_bot = y_center - TRACK_H / 2.0
    rect = patches.FancyBboxPatch(
        (x + 0.8, y_bot), COL_W - 1.6, TRACK_H,
        boxstyle="round,pad=0.2,rounding_size=0.8",
        facecolor='#FFFFFF', edgecolor=border, linewidth=1.5, zorder=3
    )
    ax.add_patch(rect)
    cx = x + COL_W / 2.0
    
    # Title
    ax.text(cx, y_bot + TRACK_H - 1.9, title, ha='center', va='center', fontsize=13.5, fontweight='bold', color=title_color, zorder=4)
    ax.plot([x + 2.0, x + COL_W - 2.0], [y_bot + TRACK_H - 3.3, y_bot + TRACK_H - 3.3], color=border, lw=0.8, alpha=0.5, zorder=4)
    
    # Intervention details
    ax.text(cx, y_bot + TRACK_H - 5.3, data_str, ha='center', va='center', fontsize=11.6, fontweight='bold', color=C_DARK, zorder=4)
    ax.text(cx, y_bot + TRACK_H - 7.3, head_str, ha='center', va='center', fontsize=11.0, color=C_TEXT, zorder=4)
    ax.text(cx, y_bot + TRACK_H - 9.3, reset_str, ha='center', va='center', fontsize=10.6, color=C_TEXT, zorder=4)
    
    # Principal Result (High-contrast, prominent)
    ax.plot([x + 2.0, x + COL_W - 2.0], [y_bot + 4.6, y_bot + 4.6], color=C_BORDER, lw=0.8, alpha=0.6, zorder=4)
    ax.text(cx, y_bot + 3.0, probe_str, ha='center', va='center', fontsize=14.0, fontweight='bold', color=title_color, zorder=4)
    ax.text(cx, y_bot + 1.3, note_str, ha='center', va='center', fontsize=10.0, color=C_MUTED, zorder=4)

# C3 (Top)
draw_phase2_card(ax, X_COL2, Y_TOP,
                 r"$C_3$: On-Target Reset Control",
                 "Data: CIFAR-100 (Target)",
                 r"Head: Temp $h_{\rm temp}$ (Fresh init)",
                 r"Velocity Reset: $\mathbf{v} \leftarrow \mathbf{0}$ at 15k, 20k",
                 "Step 20k Probe: 61.03%",
                 "Procedural Control (N = 6)",
                 border=C_C3_BORDER, title_color=C_C3_TITLE)

# C0 (Mid)
draw_phase2_card(ax, X_COL2, Y_MID,
                 r"$C_0$: Continuous Baseline",
                 "Data: CIFAR-100 (Target)",
                 r"Head: Target $h_A$ Active Throughout",
                 "Optimizer: Momentum Preserved",
                 "Step 20k Probe: 61.24%",
                 "Continuous Benchmark (N = 6)",
                 border=C_C0_BORDER, title_color=C_C0_TITLE)

# S1 (Bot)
draw_phase2_card(ax, X_COL2, Y_BOT,
                 r"$S_1$: Auxiliary Detour Interval",
                 "Data: STL-10 (Auxiliary)",
                 r"Head: Aux $h_B$ (Fresh init)",
                 r"Velocity Reset: $\mathbf{v} \leftarrow \mathbf{0}$ at 15k, 20k",
                 "Shock Probe: 20.65%",
                 "Target Amnesia Shock (-40.59 pp)",
                 border=C_S1_BORDER, title_color=C_S1_TITLE)

# =========================================================================
# --- COLUMN 3: PHASE 3 TARGET RELEARNING (BUDGET = 35k) ---
# =========================================================================
# Horizontal Connections from Phase 2 to Phase 3
draw_conn(ax, [(X_COL2 + COL_W - 0.8, Y_TOP), (X_COL3 + 0.8, Y_TOP)], arrow=True, lw=1.8)
draw_conn(ax, [(X_COL2 + COL_W - 0.8, Y_MID), (X_COL3 + 0.8, Y_MID)], arrow=True, lw=1.8)
draw_conn(ax, [(X_COL2 + COL_W - 0.8, Y_BOT), (X_COL3 + 0.8, Y_BOT)], arrow=True, lw=1.8)

# Function to draw Phase 3 Card (White background, single card, NO nested rectangles)
def draw_phase3_card(ax, x, y_center, title, protocol_str, acc_str, probe_rank_str, border, title_color):
    y_bot = y_center - TRACK_H / 2.0
    rect = patches.FancyBboxPatch(
        (x + 0.8, y_bot), COL_W - 1.6, TRACK_H,
        boxstyle="round,pad=0.2,rounding_size=0.8",
        facecolor='#FFFFFF', edgecolor=border, linewidth=1.5, zorder=3
    )
    ax.add_patch(rect)
    cx = x + COL_W / 2.0
    
    # Title
    ax.text(cx, y_bot + TRACK_H - 1.9, title, ha='center', va='center', fontsize=13.5, fontweight='bold', color=title_color, zorder=4)
    ax.plot([x + 2.0, x + COL_W - 2.0], [y_bot + TRACK_H - 3.3, y_bot + TRACK_H - 3.3], color=border, lw=0.8, alpha=0.5, zorder=4)
    
    # Protocol line
    ax.text(cx, y_bot + TRACK_H - 5.3, protocol_str, ha='center', va='center', fontsize=11.2, color=C_MUTED, zorder=4)
    
    # Terminal Test Accuracy (Primary result: giant, bold, clear)
    ax.text(cx, y_bot + TRACK_H - 8.1, acc_str, ha='center', va='center', fontsize=14.5, fontweight='bold', color=C_DARK, zorder=4)
    ax.text(cx, y_bot + TRACK_H - 10.1, "Terminal Test Acc (N = 20 paired seeds)", ha='center', va='center', fontsize=10.2, color=C_MUTED, zorder=4)
    
    # Longitudinal Representation Metrics
    ax.plot([x + 2.0, x + COL_W - 2.0], [y_bot + 4.2, y_bot + 4.2], color=C_BORDER, lw=0.8, alpha=0.6, zorder=4)
    ax.text(cx, y_bot + 2.2, probe_rank_str, ha='center', va='center', fontsize=11.8, fontweight='bold', color=title_color, zorder=4)

# C3 Terminal
draw_phase3_card(ax, X_COL3, Y_TOP,
                 r"$C_3$: Relearning Terminal",
                 r"Restored Target Head $h_A$ $\cdot$ Relearning",
                 "Acc: 56.29% (±1.59%)",
                 "Probe: 62.52%  |  EffRank: 162.37",
                 border=C_C3_BORDER, title_color=C_C3_TITLE)

# C0 Terminal
draw_phase3_card(ax, X_COL3, Y_MID,
                 r"$C_0$: Continuous Terminal",
                 "Continuous Target Training to 35k",
                 "Acc: 56.69% (±0.75%)",
                 "Probe: 63.01%  |  EffRank: 162.45",
                 border=C_C0_BORDER, title_color=C_C0_TITLE)

# S1 Terminal
draw_phase3_card(ax, X_COL3, Y_BOT,
                 r"$S_1$: Auxiliary Terminal",
                 r"Restored Target Head $h_A$ $\cdot$ Relearning",
                 "Acc: 56.26% (±1.51%)",
                 "Probe: 61.96%  |  EffRank: 153.69",
                 border=C_S1_BORDER, title_color=C_S1_TITLE)

# =========================================================================
# --- COLUMN 4: CONFIRMATORY SYNTHESIS (H1, H2, RQ3) ---
# =========================================================================
# Horizontal Connections from Phase 3 to Confirmatory Synthesis
draw_conn(ax, [(X_COL3 + COL_W - 0.8, Y_TOP), (X_COL4 + 0.8, Y_TOP)], arrow=True, lw=1.8)
draw_conn(ax, [(X_COL3 + COL_W - 0.8, Y_MID), (X_COL4 + 0.8, Y_MID)], arrow=True, lw=1.8)
draw_conn(ax, [(X_COL3 + COL_W - 0.8, Y_BOT), (X_COL4 + 0.8, Y_BOT)], arrow=True, lw=1.8)

# Helper: Draw Synthesis Panel (White background, thin border, NO saturated bars)
def draw_syn_panel(ax, x, y_center, title, subtitle, estimand_math, stat_str, verdict_l1, verdict_l2, border_color, verdict_color):
    y_bot = y_center - TRACK_H / 2.0
    rect = patches.FancyBboxPatch(
        (x + 0.8, y_bot), COL_W - 1.6, TRACK_H,
        boxstyle="round,pad=0.2,rounding_size=0.8",
        facecolor='#FFFFFF', edgecolor=border_color, linewidth=1.5, zorder=3
    )
    ax.add_patch(rect)
    cx = x + COL_W / 2.0
    
    # Title & Subtitle
    ax.text(cx, y_bot + TRACK_H - 1.9, title, ha='center', va='center', fontsize=13.5, fontweight='bold', color=C_DARK, zorder=4)
    ax.text(cx, y_bot + TRACK_H - 3.7, subtitle, ha='center', va='center', fontsize=10.6, color=C_MUTED, zorder=4)
    ax.plot([x + 2.0, x + COL_W - 2.0], [y_bot + TRACK_H - 4.9, y_bot + TRACK_H - 4.9], color=border_color, lw=0.8, alpha=0.5, zorder=4)
    
    # Estimand & Statistics (Prominent, high-contrast)
    ax.text(cx, y_bot + TRACK_H - 7.4, estimand_math, ha='center', va='center', fontsize=14.5, fontweight='bold', color=C_DARK, zorder=4)
    ax.text(cx, y_bot + TRACK_H - 9.6, stat_str, ha='center', va='center', fontsize=11.2, color=C_TEXT, zorder=4)
    
    # Conclusion (Two clean, well-spaced lines with high-contrast text, NO saturated solid bar!)
    ax.plot([x + 2.0, x + COL_W - 2.0], [y_bot + 4.4, y_bot + 4.4], color=C_BORDER, lw=0.8, alpha=0.6, zorder=4)
    ax.text(cx, y_bot + 2.9, verdict_l1, ha='center', va='center', fontsize=11.8, fontweight='bold', color=verdict_color, zorder=4)
    ax.text(cx, y_bot + 1.3, verdict_l2, ha='center', va='center', fontsize=10.2, color=verdict_color, zorder=4)

# Panel 1: Hypothesis H1 (Auxiliary vs Continuous)
draw_syn_panel(ax, X_COL4, Y_TOP,
               "Primary Contrast: Hypothesis H1",
               r"$S_1 \text{ vs. } C_0 \text{ (Budget 35k, } N=20\text{)}$",
               r"$\Delta_{\rm total} = \mathbf{-0.43\,\mathrm{pp}}$",
               r"$95\%\text{ CI: } [-0.94, +0.07] \mid p = 0.090$",
               "Result: Statistically Uncertain",
               "(No Generalization Benefit)",
               border_color=C_H1_BORDER, verdict_color=C_H1_TEXT)

# Panel 2: Hypothesis H2 (Procedural Reset Control)
draw_syn_panel(ax, X_COL4, Y_MID,
               "Procedural Control: Hypothesis H2",
               r"$C_3 \text{ vs. } C_0 \text{ (Reset Shock, } N=20\text{)}$",
               r"$\Delta_{\rm proc} = \mathbf{-0.40\,\mathrm{pp}}$",
               r"$95\%\text{ CI: } [-1.14, +0.34] \mid p = 0.269$",
               "Result: Invariance Not Rejected",
               "(Reset Impact Overcome by 35k)",
               border_color=C_C3_BORDER, verdict_color=C_C3_TITLE)

# Panel 3: RQ3 Representation Recovery
draw_syn_panel(ax, X_COL4, Y_BOT,
               "Representation Dynamics: RQ3",
               "Longitudinal Cohort (N = 6 paired seeds)",
               r"Probe: $20.65\% \to \mathbf{61.96\%}$",
               r"EffRank: $124.50 \to \mathbf{153.69}$",
               "Result: Full Feature Rebound",
               "(Decodability Fully Restored)",
               border_color=C_RQ3_BORDER, verdict_color=C_RQ3_TEXT)

# Save figures with tightly trimmed padding
png_out = os.path.join(FIGURES_DIR, "figure_edafnet_style_framework.png")
pdf_out = os.path.join(FIGURES_DIR, "figure_edafnet_style_framework.pdf")

plt.tight_layout()
plt.savefig(png_out, dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.savefig(pdf_out, bbox_inches='tight', pad_inches=0.01)
plt.close()

print(f"✓ Saved refined figure PNG: {png_out} ({os.path.getsize(png_out)/1024:.1f} KB)")
print(f"✓ Saved refined figure PDF: {pdf_out} ({os.path.getsize(pdf_out)/1024:.1f} KB)")
