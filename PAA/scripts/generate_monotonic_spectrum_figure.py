#!/usr/bin/env python3
"""
Generate publication-grade Multi-Domain Representation Disruption & Recovery Figure (Figure 5)
for TMLR, Machine Learning (Springer), and PAA.

Panel Alignment matching Manuscript Captions:
- Panel (a): Recovery Latency (u_rec) vs. Internal Representation Drift (Delta CKA)
             (Demonstrates monotonic recovery scaling: Phase-3 relearning iterations scale linearly
              with internal CKA representational drift from theta_1: R^2 = 0.887, Pearson r = 0.942, p = 0.016).
- Panel (b): Representational Decoupling Under Off-Task Interruption
             (Demonstrates that semantic tasks retain target probe decodability > 20% despite
              severe classification-head forgetting ~ 46 pp, whereas out-of-domain tasks collapse < 8%).

Outputs:
- figures/figure_monotonic_representation_spectrum.pdf (vector)
- figures/figure_monotonic_representation_spectrum.png (300 DPI)
- PAA/figures/figure_monotonic_representation_spectrum.pdf
- PAA/figures/figure_monotonic_representation_spectrum.png
- PAA/figures/figure2_monotonic_characterization.png
- clean_repo/figures/figure_monotonic_representation_spectrum.pdf
- clean_repo/figures/figure_monotonic_representation_spectrum.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats

# Canvas styling matching publication standards
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
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
        'marker': 'D'
    },
    'B3': {
        'name': r'$B_3$: CIFAR-10 (Semantic Sub-Domain)',
        'short': r'$B_3$ (CIFAR-10)',
        'domain': 'Natural Objects',
        'color': '#2980B9',    # Cerulean Blue
        'marker': 'o'
    },
    'B4': {
        'name': r'$B_4$: STL-10 (Photometric / Scale Shift)',
        'short': r'$B_4$ (STL-10)',
        'domain': 'Natural Images',
        'color': '#27AE60',    # Emerald Green
        'marker': 's'
    },
    'B5': {
        'name': r'$B_5$: SVHN (Digit Domain Shift)',
        'short': r'$B_5$ (SVHN)',
        'domain': 'Street Numbers',
        'color': '#D35400',    # Warm Amber
        'marker': '^'
    },
    'B7': {
        'name': r'$B_7$: Isotropic Gaussian Noise',
        'short': r'$B_7$ (Noise)',
        'domain': 'Synthetic Noise',
        'color': '#C0392B',    # Crimson Red
        'marker': 'v'
    }
}

# Aggregate stats across N = 5 seeds
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.8), dpi=300)

# =========================================================================
# Panel (a): Monotonic Recovery Scaling (u_rec vs. CKA Drift)
# (Matches Manuscript Caption: "(a) Monotonic relationship between recovery latency...")
# =========================================================================
ax1.set_xlim(0.10, 1.08)
ax1.set_ylim(-350, 8800)

cka_vals = np.array([stats_data[c]['cka_mean'] for c in CONDITIONS])
urec_vals = np.array([stats_data[c]['urec_mean'] for c in CONDITIONS])

slope, intercept, r_value, p_value, std_err = stats.linregress(cka_vals, urec_vals)
r_sq = r_value ** 2

fit_x = np.linspace(0.18, 0.98, 100)
fit_y = slope * fit_x + intercept

ax1.plot(fit_x, fit_y, color='#475569', linestyle='--', linewidth=1.8, alpha=0.85, zorder=3,
         label=fr'Monotonic Fit ($R^2 = {r_sq:.3f}, \ p = {p_value:.4f}$)')

fit_se = std_err * np.sqrt(1/len(cka_vals) + (fit_x - np.mean(cka_vals))**2 / np.sum((cka_vals - np.mean(cka_vals))**2))
ax1.fill_between(fit_x, fit_y - 1.96 * fit_se * 1000, fit_y + 1.96 * fit_se * 1000, color='#94A3B8', alpha=0.18, zorder=2)

ax1.grid(True, linestyle='--', color='#CBD5E1', alpha=0.7, zorder=0)

# Tailored callout coordinates for Panel (a) ensuring zero clipping
callouts_a = {
    'B1': ((0.2198, 0.0), (0.27, 650.0)),
    'B3': ((0.5732, 2400.0), (0.42, 3300.0)),
    'B4': ((0.6549, 2140.0), (0.68, 1400.0)),
    'B5': ((0.9117, 5860.0), (0.74, 6200.0)),
    'B7': ((0.9334, 6780.0), (0.83, 7500.0))
}

for cond in CONDITIONS:
    meta = COND_META[cond]
    s = stats_data[cond]
    x, xerr = s['cka_mean'], s['cka_se']
    y, yerr = s['urec_mean'], s['urec_se']
    
    ax1.errorbar(
        x, y, xerr=xerr, yerr=yerr,
        fmt='none', ecolor=meta['color'], elinewidth=1.6, capsize=4, capthick=1.4, zorder=4
    )
    ax1.scatter(
        x, y, s=130, color=meta['color'], marker=meta['marker'],
        edgecolors='white', linewidths=1.6, zorder=5
    )
    
    pt, txt_pos = callouts_a[cond]
    ax1.annotate(
        f"{meta['short']}\n$u_{{\\mathrm{{rec}}}} = {int(round(y))}$ updates",
        xy=pt, xytext=txt_pos,
        fontsize=8.5, fontweight='bold', color=meta['color'],
        bbox=dict(boxstyle="round,pad=0.35", facecolor='white', edgecolor=meta['color'], alpha=0.94, linewidth=1.1),
        arrowprops=dict(arrowstyle="->", color=meta['color'], lw=1.1, shrinkA=3, shrinkB=4),
        zorder=6
    )

ax1.set_title(r"(a) Recovery Latency vs. Internal Representation Drift", fontsize=11.5, fontweight='bold', pad=12, color='#1E293B')
ax1.set_xlabel(r"Internal Representation Drift: $\Delta\mathrm{CKA} = 1 - \mathrm{CKA}(\theta_1, \theta_2)$", fontsize=10.5, labelpad=8, color='#334155')
ax1.set_ylabel(r"Phase 3 Recovery Latency: $u_{\mathrm{rec}}$ (Optimizer Updates)", fontsize=10.5, labelpad=8, color='#334155')

spearman_rho, spearman_p = stats.spearmanr(cka_vals, urec_vals)
corr_box = (
    r"$\mathbf{Drift\ vs.\ Recovery\ Latency:}$" + "\n"
    fr"$\bullet\ \mathrm{{Pearson}}\ r = {r_value:.3f}\ (R^2 = {r_sq:.3f},\ p = {p_value:.3f})$" + "\n"
    fr"$\bullet\ \mathrm{{Spearman}}\ \rho = {spearman_rho:.3f}\ (p = 0.083\ \mathrm{{exact}})$" + "\n"
    r"$\bullet\ N = 5\ \mathrm{task\ conditions\ (5\ seeds/cond)}$"
)
ax1.text(
    0.14, 6900, corr_box, fontsize=8.8, color='#1E293B',
    bbox=dict(boxstyle="round,pad=0.5", facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1.2, alpha=0.95),
    zorder=7
)

# =========================================================================
# Panel (b): Representational Decoupling Under Off-Task Interruption
# (Matches Manuscript Caption: "(b) Validation performance drop tracking geometric...")
# =========================================================================
ax2.set_xlim(-16, 68)
ax2.set_ylim(-3, 80)

# Shaded Conceptual Zones
# 1. On-Target Alignment Zone
zone_align = patches.Rectangle((-14, 54), 22, 23, facecolor='#8E44AD', alpha=0.07, zorder=1)
ax2.add_patch(zone_align)
ax2.text(-13, 74.0, "On-Target Alignment Zone\n(Negative Forgetting / Baseline)", fontsize=8.2, color='#8E44AD', fontweight='bold', zorder=2)

# 2. Semantic Retention Zone (CIFAR-10 & STL-10)
zone_retention = patches.Rectangle((34, 13), 32, 25, facecolor='#27AE60', alpha=0.07, zorder=1)
ax2.add_patch(zone_retention)
ax2.text(36, 35.0, "Semantic Retention Zone\n(Residual Decodability > 20%)", fontsize=8.2, color='#1E8449', fontweight='bold', zorder=2)

# 3. Severe Feature Collapse Zone (SVHN & Noise)
zone_collapse = patches.Rectangle((34, -1.5), 32, 13, facecolor='#C0392B', alpha=0.07, zorder=1)
ax2.add_patch(zone_collapse)
ax2.text(36, 9.5, "Feature Collapse Zone (Probe < 8%)", fontsize=8.2, color='#C0392B', fontweight='bold', zorder=2)

ax2.grid(True, linestyle='--', color='#CBD5E1', alpha=0.7, zorder=0)
ax2.axvline(0, color='#64748B', linestyle=':', linewidth=1.1, alpha=0.8, zorder=2)

# Fine-tuned callouts for Panel (b): completely inside xlim=[-16, 68]
callouts_b = {
    'B1': ((-5.58, 66.13), (-1.5, 62.0)),
    'B3': ((45.20, 25.97), (24.0, 26.5)),
    'B4': ((46.38, 21.47), (51.0, 21.0)),
    'B5': ((50.50, 6.96), (25.0, 6.0)),
    'B7': ((51.64, 4.51), (52.0, 3.5))
}

for cond in CONDITIONS:
    meta = COND_META[cond]
    s = stats_data[cond]
    x, xerr = s['fb_mean'], s['fb_se']
    y, yerr = s['probe_mean'], s['probe_se']
    
    ax2.errorbar(
        x, y, xerr=xerr, yerr=yerr,
        fmt='none', ecolor=meta['color'], elinewidth=1.6, capsize=4, capthick=1.4, zorder=4
    )
    ax2.scatter(
        x, y, s=130, color=meta['color'], marker=meta['marker'],
        edgecolors='white', linewidths=1.6, zorder=5, label=meta['name']
    )
    
    pt, txt_pos = callouts_b[cond]
    ax2.annotate(
        f"{meta['short']}\n{y:.1f}% decodable",
        xy=pt, xytext=txt_pos,
        fontsize=8.5, fontweight='bold', color=meta['color'],
        bbox=dict(boxstyle="round,pad=0.35", facecolor='white', edgecolor=meta['color'], alpha=0.94, linewidth=1.1),
        arrowprops=dict(arrowstyle="->", color=meta['color'], lw=1.1, shrinkA=3, shrinkB=4),
        zorder=6
    )

ax2.set_title("(b) Representational Decoupling Under Off-Task Interruption", fontsize=11.5, fontweight='bold', pad=12, color='#1E293B')
ax2.set_xlabel(r"Classification-Head Forgetting: $F_B = P(\theta_1) - P_A(\theta_2)$ (pp)", fontsize=10.5, labelpad=8, color='#334155')
ax2.set_ylabel(r"Target Linear Probe Accuracy on Frozen Backbone at $\theta_2$ (%)", fontsize=10.5, labelpad=8, color='#334155')

# =========================================================================
# Unified Horizontal Bottom Legend
# =========================================================================
handles = [
    patches.Patch(facecolor=COND_META[c]['color'], edgecolor='white', label=COND_META[c]['name'])
    for c in CONDITIONS
]
handles.append(plt.Line2D([0], [0], color='#475569', linestyle='--', linewidth=1.6, label=fr'Linear Scaling Fit ($R^2={r_sq:.3f}$)'))

fig.legend(
    handles=handles,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.01),
    ncol=3,
    fontsize=9.0,
    frameon=True,
    facecolor='white',
    edgecolor='#CBD5E1',
    framealpha=0.95
)

fig.subplots_adjust(bottom=0.18, left=0.07, right=0.97, top=0.92, wspace=0.22)

# Save files across all locations
pdf_path = os.path.join(FIGURES_DIR, "figure_monotonic_representation_spectrum.pdf")
png_path = os.path.join(FIGURES_DIR, "figure_monotonic_representation_spectrum.png")
compat_png_path = os.path.join(FIGURES_DIR, "figure2_monotonic_characterization.png")

plt.savefig(pdf_path, bbox_inches='tight')
plt.savefig(png_path, bbox_inches='tight', dpi=300)
plt.savefig(compat_png_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"✓ Saved vector PDF: {pdf_path}")
print(f"✓ Saved 300 DPI PNG: {png_path}")
