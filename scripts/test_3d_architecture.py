import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, FancyBboxPatch

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'mathtext.fontset': 'cm',
})

fig = plt.figure(figsize=(15.2, 9.2), dpi=300)
# Create a 2-row layout: Top row for 3D Isometric Architecture, Bottom row for Training Schedule
gs = fig.add_gridspec(2, 1, height_ratios=[1.38, 1.0], hspace=0.28)
ax_arch = fig.add_subplot(gs[0])
ax_sched = fig.add_subplot(gs[1])

ax_arch.set_xlim(-6, 116)
ax_arch.set_ylim(-15, 65)
ax_arch.axis('off')

ax_sched.set_xlim(0, 106)
ax_sched.set_ylim(-1, 26)
ax_sched.axis('off')

# -----------------------------------------------------------------
# 3D ISOMETRIC BOX DRAWING FUNCTION
# -----------------------------------------------------------------
def draw_3d_box(ax, x, y_center, width, height, depth, color_front, color_top, color_side, 
                edge_color='#1E293B', lw=1.1, alpha=1.0, angle=38, zorder=3):
    rad = np.radians(angle)
    dx = depth * np.cos(rad)
    dy = depth * np.sin(rad)
    y = y_center - height / 2.0
    
    # Front face
    front = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
    ax.add_patch(Polygon(front, closed=True, facecolor=color_front, edgecolor=edge_color, 
                         linewidth=lw, alpha=alpha, zorder=zorder))
    
    # Top face
    top = [(x, y + height), (x + width, y + height), 
           (x + width + dx, y + height + dy), (x + dx, y + height + dy)]
    ax.add_patch(Polygon(top, closed=True, facecolor=color_top, edgecolor=edge_color, 
                         linewidth=lw, alpha=alpha, zorder=zorder+1))
    
    # Side face (right)
    side = [(x + width, y), (x + width + dx, y + dy), 
            (x + width + dx, y + height + dy), (x + width, y + height)]
    ax.add_patch(Polygon(side, closed=True, facecolor=color_side, edgecolor=edge_color, 
                         linewidth=lw, alpha=alpha, zorder=zorder))
    
    return (x + width + dx, y_center + dy/2.0)

# Colors for 3D blocks (matching reference image style)
C_CONV_FRONT = '#93C5FD' # sky blue
C_CONV_TOP   = '#DBEAFE' # light ice blue
C_CONV_SIDE  = '#3B82F6' # dark blue shadow

C_DOWN_FRONT = '#FCA5A5' # soft red/coral
C_DOWN_TOP   = '#FEE2E2' # light pink
C_DOWN_SIDE  = '#EF4444' # dark red

C_GAP_FRONT  = '#CBD5E1' # slate/silver
C_GAP_TOP    = '#F1F5F9'
C_GAP_SIDE   = '#94A3B8'

C_HEADA_FRONT = '#86EFAC' # green
C_HEADA_TOP   = '#DCFCE7'
C_HEADA_SIDE  = '#22C55E'

C_HEADB_FRONT = '#FCD34D' # amber
C_HEADB_TOP   = '#FEF3C7'
C_HEADB_SIDE  = '#F59E0B'

# -----------------------------------------------------------------
# PANEL A: 3D ISOMETRIC ARCHITECTURE (ResNet-18 + GroupNorm Backbone)
# -----------------------------------------------------------------
ax_arch.text(-5.0, 61.5, "A. 3D Isometric Network Architecture: GroupNorm ResNet-18 Backbone & Dual Task Heads", 
             fontsize=12, fontweight='bold', color='#0F172A')

Y_CENTER = 22.0

# 1. Input Image (in perspective)
img_x = -3.5
img_w = 1.0
img_h = 42.0
img_d = 7.5
rad_img = np.radians(38)
img_y = Y_CENTER - img_h / 2.0

img_poly = [(img_x, img_y), (img_x, img_y + img_h), 
            (img_x + img_d*np.cos(rad_img), img_y + img_h + img_d*np.sin(rad_img)), 
            (img_x + img_d*np.cos(rad_img), img_y + img_d*np.sin(rad_img))]
ax_arch.add_patch(Polygon(img_poly, closed=True, facecolor='#F8FAFC', edgecolor='#334155', linewidth=1.5, zorder=2))

# Gradient / grid lines inside input frame to represent image slice
for s in [0.2, 0.4, 0.6, 0.8]:
    p1 = (img_x + s*img_d*np.cos(rad_img), img_y + s*img_d*np.sin(rad_img))
    p2 = (img_x + s*img_d*np.cos(rad_img), img_y + img_h + s*img_d*np.sin(rad_img))
    ax_arch.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#94A3B8', lw=0.7, zorder=3)
for s in [0.25, 0.5, 0.75]:
    p1 = (img_x, img_y + s*img_h)
    p2 = (img_x + img_d*np.cos(rad_img), img_y + s*img_h + img_d*np.sin(rad_img))
    ax_arch.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#94A3B8', lw=0.7, zorder=3)

ax_arch.text(img_x - 0.5, img_y - 3.5, "Input Image\n$32 \\times 32 \\times 3$", fontsize=8.5, ha='center', va='top', color='#1E293B')

# Arrow from image to Conv1
ax_arch.annotate('', xy=(3.8, Y_CENTER), xytext=(1.2, Y_CENTER),
                 arrowprops=dict(arrowstyle="->", color='#475569', lw=1.6))

# 2. Stage 1: Conv1 + Initial GroupNorm (32x32x64)
# Two blue slabs
x_pos = 5.0
draw_3d_box(ax_arch, x_pos, Y_CENTER, 1.8, 38.0, 5.0, C_CONV_FRONT, C_CONV_TOP, C_CONV_SIDE)
x_pos += 2.8
draw_3d_box(ax_arch, x_pos, Y_CENTER, 1.8, 38.0, 5.0, C_CONV_FRONT, C_CONV_TOP, C_CONV_SIDE)
s1_center = x_pos - 0.9
ax_arch.text(s1_center, 51.0, "conv1 + GN\n(Stage 1)", fontsize=8.5, ha='center', va='bottom', color='#1E293B', fontweight='bold')
ax_arch.text(s1_center, 0.5, "$32 \\times 32 \\times 64$", fontsize=8.0, ha='center', va='top', color='#475569')

# Transition: Downsample 1
x_pos += 4.0
draw_3d_box(ax_arch, x_pos, Y_CENTER, 1.6, 30.0, 5.5, C_DOWN_FRONT, C_DOWN_TOP, C_DOWN_SIDE)

# 3. Stage 2: Residual Blocks (16x16x128)
x_pos += 4.0
draw_3d_box(ax_arch, x_pos, Y_CENTER, 2.2, 28.0, 6.0, C_CONV_FRONT, C_CONV_TOP, C_CONV_SIDE)
x_pos += 3.2
draw_3d_box(ax_arch, x_pos, Y_CENTER, 2.2, 28.0, 6.0, C_CONV_FRONT, C_CONV_TOP, C_CONV_SIDE)
s2_center = x_pos - 1.1
ax_arch.text(s2_center, 47.0, "Stage 2\n(ResBlocks)", fontsize=8.5, ha='center', va='bottom', color='#1E293B', fontweight='bold')
ax_arch.text(s2_center, 4.5, "$16 \\times 16 \\times 128$", fontsize=8.0, ha='center', va='top', color='#475569')

# Transition: Downsample 2
x_pos += 4.2
draw_3d_box(ax_arch, x_pos, Y_CENTER, 1.6, 22.0, 6.8, C_DOWN_FRONT, C_DOWN_TOP, C_DOWN_SIDE)

# 4. Stage 3: Residual Blocks (8x8x256)
x_pos += 4.0
draw_3d_box(ax_arch, x_pos, Y_CENTER, 2.6, 20.0, 7.5, C_CONV_FRONT, C_CONV_TOP, C_CONV_SIDE)
x_pos += 3.6
draw_3d_box(ax_arch, x_pos, Y_CENTER, 2.6, 20.0, 7.5, C_CONV_FRONT, C_CONV_TOP, C_CONV_SIDE)
s3_center = x_pos - 1.3
ax_arch.text(s3_center, 44.0, "Stage 3\n(ResBlocks)", fontsize=8.5, ha='center', va='bottom', color='#1E293B', fontweight='bold')
ax_arch.text(s3_center, 8.5, "$8 \\times 8 \\times 256$", fontsize=8.0, ha='center', va='top', color='#475569')

# Transition: Downsample 3
x_pos += 4.5
draw_3d_box(ax_arch, x_pos, Y_CENTER, 1.6, 16.0, 8.2, C_DOWN_FRONT, C_DOWN_TOP, C_DOWN_SIDE)

# 5. Stage 4: Residual Blocks (4x4x512)
x_pos += 4.0
draw_3d_box(ax_arch, x_pos, Y_CENTER, 3.0, 14.0, 9.0, C_CONV_FRONT, C_CONV_TOP, C_CONV_SIDE)
x_pos += 4.0
draw_3d_box(ax_arch, x_pos, Y_CENTER, 3.0, 14.0, 9.0, C_CONV_FRONT, C_CONV_TOP, C_CONV_SIDE)
s4_center = x_pos - 1.5
ax_arch.text(s4_center, 41.0, "Stage 4\n(ResBlocks)", fontsize=8.5, ha='center', va='bottom', color='#1E293B', fontweight='bold')
ax_arch.text(s4_center, 11.5, "$4 \\times 4 \\times 512$", fontsize=8.0, ha='center', va='top', color='#475569')

# Global Average Pooling (GAP) -> 1x1x512
x_pos += 6.0
gap_exit = draw_3d_box(ax_arch, x_pos, Y_CENTER, 1.8, 8.0, 8.5, C_GAP_FRONT, C_GAP_TOP, C_GAP_SIDE)
gap_center = x_pos + 1.2
ax_arch.text(gap_center, 38.0, "GAP\nPool", fontsize=8.5, ha='center', va='bottom', color='#0F172A', fontweight='bold')
ax_arch.text(gap_center, 14.5, "$1 \\times 1 \\times 512$", fontsize=7.8, ha='center', va='top', color='#475569')

# Bracket / Callout: Shared Backbone w
backbone_start = 5.0
backbone_end = x_pos + 3.0
ax_arch.plot([backbone_start, backbone_end], [-3.5, -3.5], color='#2563EB', lw=2.0)
ax_arch.plot([backbone_start, backbone_start], [-2.0, -3.5], color='#2563EB', lw=2.0)
ax_arch.plot([backbone_end, backbone_end], [-2.0, -3.5], color='#2563EB', lw=2.0)
ax_arch.text((backbone_start + backbone_end)/2, -6.5, 
             r"$\mathbf{Shared\;Backbone\;w}$ (GroupNorm ResNet-18, $G=32$ groups)", 
             fontsize=9.5, fontweight='bold', ha='center', va='top', color='#1D4ED8')

# -----------------------------------------------------------------
# FORKING INTO DUAL HEADS (Head A and Head B)
# -----------------------------------------------------------------
fork_x = gap_exit[0] + 1.0
fork_y = gap_exit[1]

head_a_x = 76.0
head_a_y = 38.0
head_b_x = 76.0
head_b_y = 10.0

# Upper Fork Arrow: to Target Head A (CIFAR-100)
ax_arch.annotate('', xy=(head_a_x - 1.0, head_a_y), xytext=(fork_x, fork_y),
                 arrowprops=dict(arrowstyle="->", color='#15803D', lw=2.0, connectionstyle="arc3,rad=-0.16"))

# Lower Fork Arrow: to Auxiliary Head B (STL-10)
ax_arch.annotate('', xy=(head_b_x - 1.0, head_b_y), xytext=(fork_x, fork_y),
                 arrowprops=dict(arrowstyle="->", color='#D97706', lw=2.0, connectionstyle="arc3,rad=0.16"))

# --- Target Head A Box ---
draw_3d_box(ax_arch, head_a_x, head_a_y, 4.0, 9.5, 3.5, C_HEADA_FRONT, C_HEADA_TOP, C_HEADA_SIDE, lw=1.2)
ax_arch.text(head_a_x + 8.5, head_a_y + 8.0, r"$\mathbf{Target\;Head\;} h_A$ (100 classes)", 
             fontsize=9.8, fontweight='bold', va='center', color='#15803D')
card_a_text = (
    "• Phase 1: Trained on CIFAR-100 ($u_1 = 15\\mathrm{k}$ updates)\n"
    "• Phase 2: Frozen / stashed in memory (0 grad updates)\n"
    "• Phase 3: Restored & relearned ($u_2 = 15\\mathrm{k}$ updates)"
)
ax_arch.text(head_a_x + 8.5, head_a_y + 5.0, card_a_text, 
             fontsize=8.0, va='top', color='#1E293B', linespacing=1.35)

# --- Auxiliary Head B Box ---
draw_3d_box(ax_arch, head_b_x, head_b_y, 3.5, 7.5, 3.0, C_HEADB_FRONT, C_HEADB_TOP, C_HEADB_SIDE, lw=1.2)
ax_arch.text(head_b_x + 8.5, head_b_y + 6.0, r"$\mathbf{Auxiliary\;Head\;} h_B$ (10 classes)", 
             fontsize=9.8, fontweight='bold', va='center', color='#B45309')
card_b_text = (
    "• Phase 2 only: Trained on STL-10 ($u_B = 5\\mathrm{k}$ updates)\n"
    "• Checkpoint $\\theta_2$: Discarded (no target transfer)"
)
ax_arch.text(head_b_x + 8.5, head_b_y + 3.2, card_b_text, 
             fontsize=8.0, va='top', color='#1E293B', linespacing=1.35)

# -----------------------------------------------------------------
# 3D LEGEND (Stacked neatly in Lower Right of Panel A, like reference image)
# -----------------------------------------------------------------
leg_x = 76.0
leg_y1 = -4.0
draw_3d_box(ax_arch, leg_x, leg_y1, 2.0, 3.2, 1.8, C_CONV_FRONT, C_CONV_TOP, C_CONV_SIDE, lw=0.8)
ax_arch.text(leg_x + 4.5, leg_y1, "Convolution + GroupNorm + ReLU", fontsize=8.0, va='center', color='#1E293B')

leg_y2 = -8.0
draw_3d_box(ax_arch, leg_x, leg_y2, 2.0, 3.2, 1.8, C_DOWN_FRONT, C_DOWN_TOP, C_DOWN_SIDE, lw=0.8)
ax_arch.text(leg_x + 4.5, leg_y2, "Residual Downsample (Stride 2)", fontsize=8.0, va='center', color='#1E293B')

leg_y3 = -12.0
draw_3d_box(ax_arch, leg_x, leg_y3, 2.0, 3.2, 1.8, C_HEADA_FRONT, C_HEADA_TOP, C_HEADA_SIDE, lw=0.8)
ax_arch.text(leg_x + 4.5, leg_y3, "Classification Heads (Head A / Head B)", fontsize=8.0, va='center', color='#1E293B')


# -----------------------------------------------------------------
# PANEL B: SOTLI TRAINING SCHEDULE & COMPARATORS
# -----------------------------------------------------------------
ax_sched.text(1.0, 24.5, "B. SOTLI Sequential Training Protocol vs. Continuous Baseline Comparators", 
              fontsize=11.5, fontweight='bold', color='#0F172A')

def draw_card_sched(ax, x, y, w, h, bg_color, border_color, linestyle='-', linewidth=1.2):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8",
                         facecolor=bg_color, edgecolor=border_color, 
                         linestyle=linestyle, linewidth=linewidth, zorder=2)
    ax.add_patch(box)
    return box

# --- Track 1: SOTLI 3-Phase Schedule ---
y_s1 = 14.5
h_s1 = 7.5

# Phase 1 Box
draw_card_sched(ax_sched, 2.0, y_s1, 24.5, h_s1, '#EFF6FF', '#2563EB')
ax_sched.text(14.2, y_s1 + 5.2, "Phase 1: Initial Target", fontsize=9.2, fontweight='bold', ha='center', color='#1D4ED8')
ax_sched.text(14.2, y_s1 + 3.2, r"Target A ($\mathcal{D}_A$) • $u_1 = 15\mathrm{k}$ updates", fontsize=8.0, ha='center', color='#1E293B')
ax_sched.text(14.2, y_s1 + 1.4, r"Active: $\mathbf{w}_1 + h_A^{(1)}$", fontsize=7.5, ha='center', color='#475569')

# Arrow P1 -> theta1
ax_sched.annotate('', xy=(29.0, y_s1 + h_s1/2), xytext=(26.8, y_s1 + h_s1/2),
                  arrowprops=dict(arrowstyle="->", color='#334155', lw=1.5))

# Node theta1
ax_sched.plot(31.2, y_s1 + h_s1/2, 'o', markersize=10, color='#1E3A8A', zorder=5)
ax_sched.plot(31.2, y_s1 + h_s1/2, 'o', markersize=4.5, color='#FFFFFF', zorder=6)
ax_sched.text(31.2, y_s1 + h_s1/2 + 2.4, r"Checkpoint $\theta_1$", fontsize=8.5, fontweight='bold', ha='center', color='#1E3A8A')
ax_sched.text(31.2, y_s1 + h_s1/2 - 2.2, r"$(\mathbf{w}_1, h_A^{(1)})$", fontsize=7.2, ha='center', color='#475569')

# Arrow theta1 -> P2
ax_sched.annotate('', xy=(35.5, y_s1 + h_s1/2), xytext=(33.2, y_s1 + h_s1/2),
                  arrowprops=dict(arrowstyle="->", color='#334155', lw=1.5))

# Phase 2 Box
draw_card_sched(ax_sched, 35.5, y_s1, 26.5, h_s1, '#F0FDF4', '#16A34A')
ax_sched.text(48.7, y_s1 + 5.2, "Phase 2: Auxiliary Interval", fontsize=9.2, fontweight='bold', ha='center', color='#15803D')
ax_sched.text(48.7, y_s1 + 3.2, r"Auxiliary B ($\mathcal{D}_B$) • $u_B = 5\mathrm{k}$ updates", fontsize=8.0, ha='center', color='#1E293B')
ax_sched.text(48.7, y_s1 + 1.4, r"Active: $\mathbf{w}_2 + h_B$ • $h_A^{(1)}$ [Frozen]", fontsize=7.5, ha='center', color='#166534')

# Arrow P2 -> theta2
ax_sched.annotate('', xy=(64.5, y_s1 + h_s1/2), xytext=(62.2, y_s1 + h_s1/2),
                  arrowprops=dict(arrowstyle="->", color='#334155', lw=1.5))

# Node theta2
ax_sched.plot(66.5, y_s1 + h_s1/2, 'o', markersize=10, color='#15803D', zorder=5)
ax_sched.plot(66.5, y_s1 + h_s1/2, 'o', markersize=4.5, color='#FFFFFF', zorder=6)
ax_sched.text(66.5, y_s1 + h_s1/2 + 2.4, r"Checkpoint $\theta_2$", fontsize=8.5, fontweight='bold', ha='center', color='#15803D')
ax_sched.text(66.5, y_s1 + h_s1/2 - 2.2, r"$(\mathbf{w}_2, h_A^{(1)})$", fontsize=7.2, ha='center', color='#475569')

# Arrow theta2 -> P3
ax_sched.annotate('', xy=(71.0, y_s1 + h_s1/2), xytext=(68.5, y_s1 + h_s1/2),
                  arrowprops=dict(arrowstyle="->", color='#334155', lw=1.5))

# Phase 3 Box
draw_card_sched(ax_sched, 71.0, y_s1, 24.5, h_s1, '#EFF6FF', '#2563EB')
ax_sched.text(83.2, y_s1 + 5.2, "Phase 3: Target Relearning", fontsize=9.2, fontweight='bold', ha='center', color='#1D4ED8')
ax_sched.text(83.2, y_s1 + 3.2, r"Target A ($\mathcal{D}_A$) • $u_2 = 15\mathrm{k}$ updates", fontsize=8.0, ha='center', color='#1E293B')
ax_sched.text(83.2, y_s1 + 1.4, r"Active: $\mathbf{w}_3 + h_A^{(3)}$ ($\mathbf{v}_{\mathbf{w}} \equiv \mathbf{0}$)", fontsize=7.5, ha='center', color='#DC2626')

# Arrow P3 -> theta3
ax_sched.annotate('', xy=(97.8, y_s1 + h_s1/2), xytext=(95.8, y_s1 + h_s1/2),
                  arrowprops=dict(arrowstyle="->", color='#334155', lw=1.5))

# Node theta3
ax_sched.plot(100.0, y_s1 + h_s1/2, 'o', markersize=11, color='#DC2626', zorder=5)
ax_sched.plot(100.0, y_s1 + h_s1/2, 'o', markersize=5.0, color='#FFFFFF', zorder=6)
ax_sched.text(100.0, y_s1 + h_s1/2 + 2.4, r"Final $\theta_3$", fontsize=9.0, fontweight='bold', ha='center', color='#DC2626')
ax_sched.text(100.0, y_s1 + h_s1/2 - 2.2, r"$(\mathbf{w}_3, h_A^{(3)})$", fontsize=7.2, ha='center', color='#475569')

# --- Track 2: Continuous Baseline Timeline ---
y_s2 = 1.0
h_s2 = 7.5
draw_card_sched(ax_sched, 2.0, y_s2, 101.5, h_s2, '#FEFCE8', '#CA8A04')
ax_sched.text(4.0, y_s2 + 5.5, "Continuous Single-Task Baseline (B0): Uninterrupted Training on Task A (No Resets or Head Switches)", 
              fontsize=8.5, fontweight='bold', color='#854D0E')

t_y = y_s2 + 2.5
ax_sched.plot([4.0, 100.0], [t_y, t_y], color='#CBD5E1', lw=2.0, zorder=3)

# Milestones
ax_sched.plot(4.0, t_y, '|', color='#334155', markersize=9, mew=1.8, zorder=4)
ax_sched.text(4.0, t_y - 1.8, "0k\n(Init)", fontsize=7.0, ha='center', color='#1E293B')

ax_sched.plot(31.2, t_y, '|', color='#334155', markersize=9, mew=1.8, zorder=4)
ax_sched.text(31.2, t_y - 1.8, "15k", fontsize=7.0, ha='center', color='#1E293B')

ax_sched.plot(66.5, t_y, '|', color='#334155', markersize=9, mew=1.8, zorder=4)
ax_sched.text(66.5, t_y - 1.8, "20k", fontsize=7.0, ha='center', color='#1E293B')

# 30k (Matched Target Comparator theta_C^A)
ctrl_30k_x = 83.2
ax_sched.plot(ctrl_30k_x, t_y, 's', markersize=8.5, color='#1D4ED8', zorder=5)
ax_sched.text(ctrl_30k_x, t_y + 1.8, r"$\theta_C^A$ (30k Target)", fontsize=7.8, fontweight='bold', ha='center', color='#1D4ED8')
ax_sched.text(ctrl_30k_x, t_y - 1.8, "30k", fontsize=7.0, ha='center', color='#1D4ED8')

# 35k (Matched Total Comparator theta_C^total)
ctrl_35k_x = 100.0
ax_sched.plot(ctrl_35k_x, t_y, 's', markersize=8.5, color='#DC2626', zorder=5)
ax_sched.text(ctrl_35k_x, t_y + 1.8, r"$\theta_C^{\mathrm{total}}$ (35k Total)", fontsize=7.8, fontweight='bold', ha='center', color='#DC2626')
ax_sched.text(ctrl_35k_x, t_y - 1.8, "35k", fontsize=7.0, ha='center', color='#DC2626')

# Comparison Arrows / Badges (positioned clearly in the gap between tracks: y = 9.0 to 14.5)
callout_y = 11.2

# Blue badge for Target-matched comparison
ax_sched.annotate('', xy=(80.0, y_s1), xytext=(ctrl_30k_x, t_y + 2.5),
                  arrowprops=dict(arrowstyle="->", color='#1D4ED8', lw=1.2, linestyle=':'))
ax_sched.text(78.5, callout_y, r"$\Delta_A = P_A(\theta_3) - P_A(\theta_C^A)$" + "\nTarget-matched: +1.57 pp ($p=0.013$)", 
              fontsize=7.2, fontweight='bold', color='#1D4ED8', ha='center', va='center',
              bbox=dict(boxstyle='round,pad=0.25', facecolor='#EFF6FF', edgecolor='#93C5FD', lw=0.7))

# Red badge for Total-budget comparison
ax_sched.annotate('', xy=(99.0, y_s1), xytext=(ctrl_35k_x, t_y + 2.5),
                  arrowprops=dict(arrowstyle="->", color='#DC2626', lw=1.2, linestyle=':'))
ax_sched.text(98.5, callout_y, r"$\Delta_{\mathrm{total}} = P_A(\theta_3) - P_A(\theta_C^{\mathrm{total}})$" + "\nBudget-matched: +0.76 pp ($p=0.249$)", 
              fontsize=7.2, fontweight='bold', color='#DC2626', ha='center', va='center',
              bbox=dict(boxstyle='round,pad=0.25', facecolor='#FEF2F2', edgecolor='#FCA5A5', lw=0.7))

fig.savefig('manuscript/figures/figure_method_schematic.png', dpi=300, bbox_inches='tight')
fig.savefig('results/figures/figure_method_schematic.png', dpi=300, bbox_inches='tight')
print("Successfully generated finalized 3D Isometric Architecture Diagram!")
