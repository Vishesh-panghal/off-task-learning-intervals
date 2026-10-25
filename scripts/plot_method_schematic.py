import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'mathtext.fontset': 'cm',
})

# Dimensions: wide format with generous breathing room
fig, ax = plt.subplots(figsize=(14.2, 6.6), dpi=300)
ax.set_xlim(0, 106)
ax.set_ylim(0, 56)
ax.axis('off')

# Color palette: subtle, modern Mermaid / academic palette
C_BG_A = '#EDF4FC'          # Soft slate blue
C_BORDER_A = '#2B6CB0'      # Deep slate blue
C_BG_B = '#EDF7ED'          # Soft sage green
C_BORDER_B = '#2E7D32'      # Deep forest green
C_BG_FROZEN = '#F8F9FA'     # Subtle neutral gray
C_BORDER_FROZEN = '#6C757D'
C_BG_CONT = '#FEF9ED'       # Warm muted amber for baseline
C_BORDER_CONT = '#B78103'
C_ARROW = '#374151'
C_TEXT = '#1F2937'
C_MUTED = '#4B5563'

# -------------------------------------------------------------
# 1. SECTION TITLES
# -------------------------------------------------------------
ax.text(1.5, 53.5, "A. Structured Off-Task Learning Intervals (SOTLI Training Schedule)", 
        fontsize=11.5, fontweight='bold', color='#1E3A8A')

ax.text(1.5, 18.5, "B. Continuous Target Baselines (Dual Computational Comparators)", 
        fontsize=11.5, fontweight='bold', color='#78350F')

# Helper function for rounded boxes
def draw_card(ax, x, y, w, h, bg_color, border_color, linestyle='-', linewidth=1.4, alpha=1.0, zorder=2):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.2",
                         facecolor=bg_color, edgecolor=border_color, 
                         linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    return box

# Helper function for badges
def draw_badge(ax, x, y, text, bg='#FFFFFF', border='#B0BEC5', text_color='#37474F', fontsize=8.5, zorder=5):
    ax.text(x, y, text, fontsize=fontsize, fontweight='bold', color=text_color,
            ha='center', va='center', zorder=zorder,
            bbox=dict(boxstyle='round,pad=0.32,rounding_size=0.8', facecolor=bg, edgecolor=border, linewidth=1.0))

# -------------------------------------------------------------
# 2. TRACK 1: SOTLI SCHEDULE (Phases 1, 2, 3)
# -------------------------------------------------------------
card_w = 24.5
card_h = 24.0
y_track1 = 26.0

# ==================== Phase 1 ====================
p1_x = 2.0
draw_card(ax, p1_x, y_track1, card_w, card_h, C_BG_A, C_BORDER_A)
ax.text(p1_x + card_w/2, y_track1 + card_h - 2.2, "Phase 1: Initial Target", 
        fontsize=10.8, fontweight='bold', ha='center', color=C_BORDER_A, zorder=3)
draw_badge(ax, p1_x + card_w/2, y_track1 + card_h - 5.2, r"Target Task A  •  $u_1 = 15\mathrm{k}$ updates", 
           bg='#FFFFFF', border=C_BORDER_A, text_color=C_BORDER_A)

# Shared Backbone w1
draw_card(ax, p1_x + 1.8, y_track1 + 9.8, card_w - 3.6, 7.0, '#FFFFFF', C_BORDER_A, linewidth=1.0)
ax.text(p1_x + card_w/2, y_track1 + 14.8, r"Shared Backbone $\mathbf{w}_1$", fontsize=10, fontweight='bold', ha='center', color=C_TEXT, zorder=3)
ax.text(p1_x + card_w/2, y_track1 + 12.6, "ResNet-18 (GroupNorm, G=32)", fontsize=8.2, ha='center', color=C_MUTED, zorder=3)
ax.text(p1_x + card_w/2, y_track1 + 10.9, r"Active SGD updates ($\mu = 0.9$)", fontsize=7.5, ha='center', color=C_MUTED, zorder=3)

# Active Head A
draw_card(ax, p1_x + 1.8, y_track1 + 1.2, card_w - 3.6, 7.2, '#E0EEFB', C_BORDER_A, linewidth=1.0)
ax.text(p1_x + card_w/2, y_track1 + 6.2, r"Target Head $h_A^{(1)}$ (Active)", fontsize=9.5, fontweight='bold', ha='center', color='#1A4971', zorder=3)
ax.text(p1_x + card_w/2, y_track1 + 4.2, "100-class linear classifier", fontsize=8.2, ha='center', color='#1A4971', zorder=3)
ax.text(p1_x + card_w/2, y_track1 + 2.5, r"Trained on $\mathcal{D}_A$ (CIFAR-100)", fontsize=7.5, ha='center', color=C_MUTED, zorder=3)


# --- Checkpoint θ1 Node ---
cp1_x = 31.0
cp_y = y_track1 + card_h/2
ax.plot(cp1_x, cp_y, 'o', markersize=13, color='#1E3A8A', zorder=5)
ax.plot(cp1_x, cp_y, 'o', markersize=6.0, color='#FFFFFF', zorder=6)
ax.text(cp1_x, cp_y + 3.2, r"Checkpoint $\theta_1$", fontsize=9.5, fontweight='bold', ha='center', color='#1E3A8A')
ax.text(cp1_x, cp_y - 3.2, r"$(\mathbf{w}_1, h_A^{(1)})$", fontsize=8.2, ha='center', color=C_TEXT)
ax.text(cp1_x, cp_y - 5.2, r"$u = 15\mathrm{k}$", fontsize=7.6, ha='center', color=C_MUTED)

# Arrow from P1 to theta1
ax.annotate('', xy=(cp1_x - 2.0, cp_y), xytext=(p1_x + card_w, cp_y),
            arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.6))
# Arrow from theta1 to P2
p2_x = 35.5
ax.annotate('', xy=(p2_x, cp_y), xytext=(cp1_x + 2.0, cp_y),
            arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.6))


# ==================== Phase 2 ====================
draw_card(ax, p2_x, y_track1, card_w, card_h, C_BG_B, C_BORDER_B)
ax.text(p2_x + card_w/2, y_track1 + card_h - 2.2, "Phase 2: Auxiliary Interval", 
        fontsize=10.8, fontweight='bold', ha='center', color=C_BORDER_B, zorder=3)
draw_badge(ax, p2_x + card_w/2, y_track1 + card_h - 5.2, r"Auxiliary Task B  •  $u_B = 5\mathrm{k}$ updates", 
           bg='#FFFFFF', border=C_BORDER_B, text_color=C_BORDER_B)

# Shared Backbone w2
draw_card(ax, p2_x + 1.8, y_track1 + 12.0, card_w - 3.6, 5.2, '#FFFFFF', C_BORDER_B, linewidth=1.0)
ax.text(p2_x + card_w/2, y_track1 + 15.3, r"Shared Backbone $\mathbf{w}_2$ (Updated)", fontsize=9.2, fontweight='bold', ha='center', color=C_TEXT, zorder=3)
ax.text(p2_x + card_w/2, y_track1 + 13.3, r"Adapted on $\mathcal{D}_B$ (STL-10, SVHN, etc.)", fontsize=7.8, ha='center', color=C_MUTED, zorder=3)

# Active Temp Head B
draw_card(ax, p2_x + 1.8, y_track1 + 6.4, card_w - 3.6, 4.8, '#DCF2DC', C_BORDER_B, linewidth=1.0)
ax.text(p2_x + card_w/2, y_track1 + 9.5, r"Temporary Head $h_B$ (Active)", fontsize=9.2, fontweight='bold', ha='center', color='#1E7D32', zorder=3)
ax.text(p2_x + card_w/2, y_track1 + 7.6, "Fresh init • Discarded at θ₂", fontsize=7.8, ha='center', color='#1E7D32', zorder=3)

# Frozen Stashed Head A
draw_card(ax, p2_x + 1.8, y_track1 + 1.0, card_w - 3.6, 4.6, C_BG_FROZEN, C_BORDER_FROZEN, linestyle='--', linewidth=1.2)
ax.text(p2_x + card_w/2, y_track1 + 4.0, r"Target Head $h_A^{(1)}$ [Frozen / Stashed]", fontsize=8.5, fontweight='bold', ha='center', color='#374151', zorder=3)
ax.text(p2_x + card_w/2, y_track1 + 2.1, "Held in memory; no Task B updates", fontsize=7.4, ha='center', color=C_MUTED, zorder=3)


# --- Checkpoint θ2 Node ---
cp2_x = 64.5
ax.plot(cp2_x, cp_y, 'o', markersize=13, color='#2E7D32', zorder=5)
ax.plot(cp2_x, cp_y, 'o', markersize=6.0, color='#FFFFFF', zorder=6)
ax.text(cp2_x, cp_y + 3.2, r"Checkpoint $\theta_2$", fontsize=9.5, fontweight='bold', ha='center', color='#2E7D32')
ax.text(cp2_x, cp_y - 3.2, r"$(\mathbf{w}_2, h_A^{(1)})$", fontsize=8.2, ha='center', color=C_TEXT)
ax.text(cp2_x, cp_y - 5.2, r"$u = 20\mathrm{k}$", fontsize=7.6, ha='center', color=C_MUTED)

# Arrow from P2 to theta2
ax.annotate('', xy=(cp2_x - 2.0, cp_y), xytext=(p2_x + card_w, cp_y),
            arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.6))
# Arrow from theta2 to P3
p3_x = 69.0
ax.annotate('', xy=(p3_x, cp_y), xytext=(cp2_x + 2.0, cp_y),
            arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.6))


# ==================== Phase 3 ====================
draw_card(ax, p3_x, y_track1, card_w, card_h, C_BG_A, C_BORDER_A)
ax.text(p3_x + card_w/2, y_track1 + card_h - 2.2, "Phase 3: Target Relearning", 
        fontsize=10.8, fontweight='bold', ha='center', color=C_BORDER_A, zorder=3)
draw_badge(ax, p3_x + card_w/2, y_track1 + card_h - 5.2, r"Target Task A  •  $u_2 = 15\mathrm{k}$ updates", 
           bg='#FFFFFF', border=C_BORDER_A, text_color=C_BORDER_A)

# Shared Backbone w3
draw_card(ax, p3_x + 1.8, y_track1 + 9.8, card_w - 3.6, 7.0, '#FFFFFF', C_BORDER_A, linewidth=1.0)
ax.text(p3_x + card_w/2, y_track1 + 14.8, r"Shared Backbone $\mathbf{w}_3$", fontsize=10, fontweight='bold', ha='center', color=C_TEXT, zorder=3)
ax.text(p3_x + card_w/2, y_track1 + 12.6, r"Re-adapted to Task A from $\mathbf{w}_2$", fontsize=8.2, ha='center', color=C_MUTED, zorder=3)
ax.text(p3_x + card_w/2, y_track1 + 10.9, r"Backbone $\mathbf{v}_{\mathbf{w}} \equiv \mathbf{0}$ at switch (SOTLI)", fontsize=7.5, ha='center', color='#C0392B', zorder=3)

# Restored Head A
draw_card(ax, p3_x + 1.8, y_track1 + 1.2, card_w - 3.6, 7.2, '#E0EEFB', C_BORDER_A, linewidth=1.0)
ax.text(p3_x + card_w/2, y_track1 + 6.2, r"Target Head $h_A^{(3)}$ (Restored)", fontsize=9.5, fontweight='bold', ha='center', color='#1A4971', zorder=3)
ax.text(p3_x + card_w/2, y_track1 + 4.2, r"Resumed from stashed $h_A^{(1)}$ state", fontsize=8.2, ha='center', color='#1A4971', zorder=3)
ax.text(p3_x + card_w/2, y_track1 + 2.5, r"Relearned on $\mathcal{D}_A$ for 15k steps", fontsize=7.5, ha='center', color=C_MUTED, zorder=3)


# --- Checkpoint θ3 Node ---
cp3_x = 98.0
ax.plot(cp3_x, cp_y, 'o', markersize=13, color='#C0392B', zorder=5)
ax.plot(cp3_x, cp_y, 'o', markersize=6.0, color='#FFFFFF', zorder=6)
ax.text(cp3_x, cp_y + 3.2, r"Final $\theta_3$", fontsize=10, fontweight='bold', ha='center', color='#C0392B')
ax.text(cp3_x, cp_y - 3.2, r"$(\mathbf{w}_3, h_A^{(3)})$", fontsize=8.2, ha='center', color=C_TEXT)
ax.text(cp3_x, cp_y - 5.2, r"$U = 35\mathrm{k}$", fontsize=7.6, ha='center', color=C_MUTED)

# Arrow from P3 to theta3
ax.annotate('', xy=(cp3_x - 2.0, cp_y), xytext=(p3_x + card_w, cp_y),
            arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.6))


# -------------------------------------------------------------
# 3. TRACK 2: CONTINUOUS BASELINES & COMPARATORS
# -------------------------------------------------------------
y_track2 = 2.0
cont_w = 101.5
cont_h = 14.2

draw_card(ax, 2.0, y_track2, cont_w, cont_h, C_BG_CONT, C_BORDER_CONT, linestyle='-', linewidth=1.2)

# Label inside continuous bar
ax.text(4.0, y_track2 + cont_h - 2.2, "Continuous Single-Task Baseline (B0): Uninterrupted Direct Target Training", 
        fontsize=9.8, fontweight='bold', color='#78350F', zorder=3)
ax.text(4.0, y_track2 + cont_h - 4.1, "Single head h_A; optimizer momentum retained continuously throughout without resets or head changes.", 
        fontsize=8.0, color='#92400E', zorder=3)

# Update axis line
axis_y = y_track2 + 4.2
ax.plot([4.0, cont_w - 1.0], [axis_y, axis_y], color='#BDBDBD', lw=2.0, zorder=3)

# Milestones on continuous axis
# Milestone 0k
ax.plot(4.0, axis_y, '|', color='#424242', markersize=11, mew=2.0, zorder=4)
ax.text(4.0, axis_y - 2.2, "0k\n(Init)", fontsize=7.5, ha='center', color=C_TEXT)

# Milestone 15k (aligned with theta1)
ax.plot(cp1_x, axis_y, '|', color='#424242', markersize=11, mew=2.0, zorder=4)
ax.text(cp1_x, axis_y - 2.2, "15k", fontsize=7.5, ha='center', color=C_TEXT)

# Milestone 20k (aligned with theta2)
ax.plot(cp2_x, axis_y, '|', color='#424242', markersize=11, mew=2.0, zorder=4)
ax.text(cp2_x, axis_y - 2.2, "20k", fontsize=7.5, ha='center', color=C_TEXT)

# Milestone 30k: Control 2 (Equal Target Updates)
ctrl2_x = 81.5
ax.plot(ctrl2_x, axis_y, 's', markersize=9.5, color='#0D47A1', zorder=5)
ax.text(ctrl2_x, axis_y + 2.2, r"$\theta_C^A$ (30k Target Steps)", fontsize=8.5, fontweight='bold', ha='center', color='#0D47A1')
ax.text(ctrl2_x, axis_y - 2.2, "30k steps\n(Matched Target)", fontsize=7.4, ha='center', color='#0D47A1')

# Milestone 35k: Control 1 (Equal Total Updates, aligned with theta3)
ctrl1_x = cp3_x
ax.plot(ctrl1_x, axis_y, 's', markersize=9.5, color='#B71C1C', zorder=5)
ax.text(ctrl1_x, axis_y + 2.2, r"$\theta_C^{\mathrm{total}}$ (35k Total Steps)", fontsize=8.5, fontweight='bold', ha='center', color='#B71C1C')
ax.text(ctrl1_x, axis_y - 2.2, "35k steps\n(Matched Total)", fontsize=7.4, ha='center', color='#B71C1C')

# --- Comparative Callouts / Deltas ---
# Bracket for Delta_A (vs 30k)
ax.annotate('', xy=(cp3_x - 1.5, y_track1 - 0.2), xytext=(ctrl2_x, axis_y + 3.6),
            arrowprops=dict(arrowstyle="<->", color='#0D47A1', lw=1.5, linestyle=':',
                            connectionstyle="arc3,rad=-0.14"))
ax.text(78.5, 20.8, r"$\Delta_A = P_A(\theta_3) - P_A(\theta_C^A)$" + "\nMatched Target Updates (+1.57 pp, p=0.013)", 
        fontsize=7.8, fontweight='bold', color='#0D47A1', ha='center',
        bbox=dict(boxstyle='round,pad=0.28', facecolor='#EBF4FC', edgecolor='#90CAF9', lw=0.9))

# Bracket for Delta_total (vs 35k)
ax.annotate('', xy=(cp3_x, y_track1 - 0.2), xytext=(ctrl1_x, axis_y + 3.6),
            arrowprops=dict(arrowstyle="<->", color='#B71C1C', lw=1.5, linestyle=':',
                            connectionstyle="arc3,rad=0.0"))
ax.text(98.5, 20.8, r"$\Delta_{\mathrm{total}} = P_A(\theta_3) - P_A(\theta_C^{\mathrm{total}})$" + "\nEqual Total Budget (+0.76 pp, p=0.249)", 
        fontsize=7.8, fontweight='bold', color='#B71C1C', ha='center',
        bbox=dict(boxstyle='round,pad=0.28', facecolor='#FEECEC', edgecolor='#EF9A9A', lw=0.9))

fig.savefig('manuscript/figures/figure_method_schematic.png', dpi=300, bbox_inches='tight')
fig.savefig('results/figures/figure_method_schematic.png', dpi=300, bbox_inches='tight')
print("Successfully generated refined figure_method_schematic.png!")
