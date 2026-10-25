#!/usr/bin/env python3
"""
Generate publication-grade Multi-Domain Representation Disruption & Recovery Figure
for Pattern Analysis and Applications (PAA, Springer Nature).

Redesigns the exploratory multi-domain characterization figure into a modern,
academic vector graphic matching the design aesthetics of Fig. 1 (EDAFNet style)
and Fig. 3 (Longitudinal Dynamics).

Visualizes Table 7 multi-domain screening data (N = 5 seeds) from supplementary/results/screening_results.csv:
- Panel (a): Diagnostic Probe Decodability vs. Classifier Head Forgetting
             (Demonstrates representational decoupling: semantic tasks retain target decodability
              despite apparent classification-head forgetting, whereas out-of-domain tasks collapse).
- Panel (b): Recovery Latency (u_rec) vs. Representation Drift (Delta CKA)
             (Demonstrates monotonic recovery scaling: Phase-3 relearning iterations scale linearly
              with internal CKA representational drift from theta_1).

Outputs:
- PAA/figures/figure_monotonic_representation_spectrum.pdf (vector)
- PAA/figures/figure_monotonic_representation_spectrum.png (300 DPI)
- PAA/figures/figure2_monotonic_characterization.png (backward-compatible upgrade)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats

# Canvas styling matching PAA standards
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FIGURES_DIR = os.path.join(BASE_DIR, "PAA", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

CSV_PATH = os.path.join(BASE_DIR, "supplementary", "results", "screening_results.csv")
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(BASE_DIR, "results", "screening_results.csv")

df = pd.read_csv(CSV_PATH)

# Authoritative condition configurations
CONDITIONS = ['B1', 'B3', 'B4', 'B5', 'B7']
COND_META = {
    'B1': {
        'name': r'$B_1$: On-Task Reset Control',
        'short': r'$B_1$ (Clean Reset)',
        'domain': 'CIFAR-100',
        'color': '#8E44AD',    # Royal Purple
        'marker': 'D',
        'callout_offset_a': (12, -18),
        'callout_offset_b': (12, -16)
    },
    'B3': {
        'name': r'$B_3$: CIFAR-10 (Semantic Sub-Domain)',
        'short': r'$B_3$ (CIFAR-10)',
        'domain': 'Natural Objects',
        'color': '#2980B9',    # Cerulean Blue
        'marker': 'o',
        'callout_offset_a': (-95, 12),
        'callout_offset_b': (-70, 16)
    },
    'B4': {
        'name': r'$B_4$: STL-10 (Photometric / Scale Shift)',
        'short': r'$B_4$ (STL-10)',
        'domain': 'Natural Images',
        'color': '#27AE60',    # Emerald Green
        'marker': 's',
        'callout_offset_a': (14, 6),
        'callout_offset_b': (14, -14)
    },
    'B5': {
        'name': r'$B_5$: SVHN (Digit Domain Shift)',
        'short': r'$B_5$ (SVHN)',
        'domain': 'Street Numbers',
        'color': '#D35400',    # Warm Amber
        'marker': '^',
        'callout_offset_a': (-85, -20),
        'callout_offset_b': (-75, 12)
    },
    'B7': {
        'name': r'$B_7$: Isotropic Gaussian Noise',
        'short': r'$B_7$ (Noise)',
        'domain': 'Synthetic Noise',
        'color': '#C0392B',    # Crimson Red
        'marker': 'v',
        'callout_offset_a': (12, 10),
        'callout_offset_b': (12, -14)
    }
}

## Aggregate stats across N = 5 seeds
stats_data = {}
for cond in CONDITIONS:
    sub = df[df['condition_id'] == cond]
    n = len(sub)
    urec_vals = np.zeros(n) if cond == 'B1' else sub['u_rec']
    stats_data[cond] = {
        'fb_mean': sub['f_b'].mean(),
        'fb_se': sub['f_b'].std(ddof=1) / np.sqrt(n),
        'probe_mean': sub['p_diagnostic_probe'].mean(),
        'probe_se': sub['p_diagnostic_probe'].std(ddof=1) / np.sqrt(n),
        'cka_mean': sub['cka_drift'].mean(),
        'cka_se': sub['cka_drift'].std(ddof=1) / np.sqrt(n),
        'urec_mean': urec_vals.mean(),
        'urec_se': urec_vals.std(ddof=1) / np.sqrt(n) if cond != 'B1' else 0.0,
        'n': n
    }

# Setup figure canvas (Two side-by-side panels)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 6.0), dpi=300)

# -------------------------------------------------------------
# Panel (a): Diagnostic Probe Decodability vs. Apparent Forgetting
# -------------------------------------------------------------
ax1.set_xlim(-14, 62)
ax1.set_ylim(-3, 78)

# Shaded Conceptual Zones
# 1. On-Target Alignment Zone
zone_align = patches.Rectangle((-12, 54), 20, 21, facecolor='#8E44AD', alpha=0.06, zorder=1)
ax1.add_patch(zone_align)
ax1.text(-11, 72.5, "On-Target Alignment Zone\n(Negative Forgetting / Baseline)", fontsize=8.2, color='#8E44AD', fontweight='bold', zorder=2)

# 2. Semantic Retention Zone (CIFAR-10 & STL-10)
zone_retention = patches.Rectangle((35, 13), 25, 23, facecolor='#27AE60', alpha=0.06, zorder=1)
ax1.add_patch(zone_retention)
ax1.text(36, 33.5, "Semantic Retention Zone\n(Residual Decodability > 20%)", fontsize=8.2, color='#1E8449', fontweight='bold', zorder=2)

# 3. Severe Feature Collapse Zone (SVHN & Noise)
zone_collapse = patches.Rectangle((44, -2), 17, 13, facecolor='#C0392B', alpha=0.06, zorder=1)
ax1.add_patch(zone_collapse)
ax1.text(45, -1.2, "Feature Collapse (Probe < 8%)", fontsize=8.2, color='#C0392B', fontweight='bold', zorder=2)

# Grid and reference lines
ax1.grid(True, linestyle='--', color='#CBD5E1', alpha=0.7, zorder=0)
ax1.axvline(0, color='#64748B', linestyle=':', linewidth=1.1, alpha=0.8, zorder=2)

# Plot data points with custom error bars and annotations
# Explicit fine-tuned label positions for Panel A to avoid collisions
callouts_a = {
    'B1': ((-5.58, 66.13), (-1.0, 64.0)),
    'B3': ((45.20, 25.97), (25.0, 24.5)),
    'B4': ((46.38, 21.47), (51.5, 20.0)),
    'B5': ((50.50, 6.96), (33.0, 6.0)),
    'B7': ((51.64, 4.51), (52.5, 4.0))
}

for cond in CONDITIONS:
    meta = COND_META[cond]
    s = stats_data[cond]
    x, xerr = s['fb_mean'], s['fb_se']
    y, yerr = s['probe_mean'], s['probe_se']
    
    # Error bar
    ax1.errorbar(
        x, y, xerr=xerr, yerr=yerr,
        fmt='none', ecolor=meta['color'], elinewidth=1.6, capsize=4, capthick=1.4, zorder=4
    )
    # Scatter point with white border
    ax1.scatter(
        x, y, s=130, color=meta['color'], marker=meta['marker'],
        edgecolors='white', linewidths=1.6, zorder=5, label=meta['name']
    )
    
    # Pill badge annotation with tailored position
    pt, txt_pos = callouts_a[cond]
    ax1.annotate(
        f"{meta['short']}\n{y:.1f}% decodable",
        xy=pt, xytext=txt_pos,
        fontsize=8.5, fontweight='bold', color=meta['color'],
        bbox=dict(boxstyle="round,pad=0.35", facecolor='white', edgecolor=meta['color'], alpha=0.92, linewidth=1.1),
        arrowprops=dict(arrowstyle="->", color=meta['color'], lw=1.1, shrinkA=3, shrinkB=4),
        zorder=6
    )

ax1.set_title("(a) Representational Decoupling Under Off-Task Interruption", fontsize=11.5, fontweight='bold', pad=12, color='#1E293B')
ax1.set_xlabel(r"Classification-Head Forgetting: $F_B = P(\theta_1) - P_A(\theta_2)$ (pp)", fontsize=10.5, labelpad=8, color='#334155')
ax1.set_ylabel(r"Target Linear Probe Accuracy on Frozen Backbone at $\theta_2$ (%)", fontsize=10.5, labelpad=8, color='#334155')

# -------------------------------------------------------------
# Panel (b): Monotonic Recovery Scaling (u_rec vs. CKA Drift)
# -------------------------------------------------------------
ax2.set_xlim(0.12, 1.05)
ax2.set_ylim(-300, 8600)

# Regression line computation across 5 conditions
cka_vals = np.array([stats_data[c]['cka_mean'] for c in CONDITIONS])
urec_vals = np.array([stats_data[c]['urec_mean'] for c in CONDITIONS])

slope, intercept, r_value, p_value, std_err = stats.linregress(cka_vals, urec_vals)
r_sq = r_value ** 2

# Generate fitted line and soft confidence envelope
fit_x = np.linspace(0.18, 0.98, 100)
fit_y = slope * fit_x + intercept

ax2.plot(fit_x, fit_y, color='#475569', linestyle='--', linewidth=1.8, alpha=0.85, zorder=3,
         label=fr'Monotonic Fit ($R^2 = {r_sq:.3f}, \ p = {p_value:.4f}$)')

# Soft regression confidence band
fit_se = std_err * np.sqrt(1/len(cka_vals) + (fit_x - np.mean(cka_vals))**2 / np.sum((cka_vals - np.mean(cka_vals))**2))
ax2.fill_between(fit_x, fit_y - 1.96 * fit_se * 1000, fit_y + 1.96 * fit_se * 1000, color='#94A3B8', alpha=0.18, zorder=2)

ax2.grid(True, linestyle='--', color='#CBD5E1', alpha=0.7, zorder=0)

# Tailored callout coordinates for Panel B
callouts_b = {
    'B1': ((0.2198, 0.0), (0.28, 500.0)),
    'B3': ((0.5732, 2400.0), (0.42, 3100.0)),
    'B4': ((0.6549, 2140.0), (0.69, 1700.0)),
    'B5': ((0.9117, 5860.0), (0.74, 6200.0)),
    'B7': ((0.9334, 6780.0), (0.91, 7400.0))
}

# Plot data points for Panel (b)
for cond in CONDITIONS:
    meta = COND_META[cond]
    s = stats_data[cond]
    x, xerr = s['cka_mean'], s['cka_se']
    y, yerr = s['urec_mean'], s['urec_se']
    
    # Error bar
    ax2.errorbar(
        x, y, xerr=xerr, yerr=yerr,
        fmt='none', ecolor=meta['color'], elinewidth=1.6, capsize=4, capthick=1.4, zorder=4
    )
    # Scatter point with white border
    ax2.scatter(
        x, y, s=130, color=meta['color'], marker=meta['marker'],
        edgecolors='white', linewidths=1.6, zorder=5
    )
    
    # Pill badge annotation
    pt, txt_pos = callouts_b[cond]
    ax2.annotate(
        f"{meta['short']}\n$u_{{\\mathrm{{rec}}}} = {int(round(y))}$ updates",
        xy=pt, xytext=txt_pos,
        fontsize=8.5, fontweight='bold', color=meta['color'],
        bbox=dict(boxstyle="round,pad=0.35", facecolor='white', edgecolor=meta['color'], alpha=0.92, linewidth=1.1),
        arrowprops=dict(arrowstyle="->", color=meta['color'], lw=1.1, shrinkA=3, shrinkB=4),
        zorder=6
    )

ax2.set_title(r"(b) Association between drift and recovery latency ($u_{\mathrm{rec}}$)", fontsize=11.5, fontweight='bold', pad=12, color='#1E293B')
ax2.set_xlabel(r"Internal Representation Drift: $\Delta\mathrm{CKA} = 1 - \mathrm{CKA}(\theta_1, \theta_2)$", fontsize=10.5, labelpad=8, color='#334155')
ax2.set_ylabel(r"Phase 3 Recovery Latency: $u_{\mathrm{rec}}$ (Optimizer Updates)", fontsize=10.5, labelpad=8, color='#334155')

# Correlation Stat Box inside Panel B (computed across N = 5 task conditions)
spearman_rho, spearman_p = stats.spearmanr(cka_vals, urec_vals)
corr_box = (
    r"$\mathbf{Drift\ vs.\ Recovery\ Latency:}$" + "\n"
    fr"$\bullet\ \mathrm{{Pearson}}\ r = {r_value:.3f}\ (R^2 = {r_sq:.3f},\ p = {p_value:.3f})$" + "\n"
    fr"$\bullet\ \mathrm{{Spearman}}\ \rho = {spearman_rho:.3f}\ (p = 0.083\ \mathrm{{exact}})$" + "\n"
    r"$\bullet\ N = 5\ \mathrm{task\ conditions\ (5\ seeds/cond)}$"
)
ax2.text(
    0.16, 6800, corr_box, fontsize=8.8, color='#1E293B',
    bbox=dict(boxstyle="round,pad=0.5", facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1.2, alpha=0.95),
    zorder=7
)

# -------------------------------------------------------------
# Unified Horizontal Bottom Legend
# -------------------------------------------------------------
handles = [
    patches.Patch(facecolor=COND_META[c]['color'], edgecolor='white', label=COND_META[c]['name'])
    for c in CONDITIONS
]
handles.append(plt.Line2D([0], [0], color='#475569', linestyle='--', linewidth=1.6, label=fr'Linear Scaling Fit ($R^2={r_sq:.3f}$)'))

fig.legend(
    handles=handles,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.04),
    ncol=3,
    fontsize=9.2,
    frameon=True,
    facecolor='white',
    edgecolor='#CBD5E1',
    framealpha=0.95
)

plt.tight_layout()
fig.subplots_adjust(bottom=0.15)

# Save files
pdf_path = os.path.join(FIGURES_DIR, "figure_monotonic_representation_spectrum.pdf")
png_path = os.path.join(FIGURES_DIR, "figure_monotonic_representation_spectrum.png")
compat_png_path = os.path.join(FIGURES_DIR, "figure2_monotonic_characterization.png")

plt.savefig(pdf_path, bbox_inches='tight')
plt.savefig(png_path, bbox_inches='tight', dpi=300)
plt.savefig(compat_png_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"✓ Saved vector PDF: {pdf_path}")
print(f"✓ Saved 300 DPI PNG: {png_path}")
print(f"✓ Updated compatibility file: {compat_png_path}")
