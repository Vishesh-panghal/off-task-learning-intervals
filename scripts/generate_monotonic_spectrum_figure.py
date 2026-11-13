#!/usr/bin/env python3
"""
Generate publication-grade Figure 5: Multi-Domain Recovery and Decoupling Spectrum.
Designed at native 6.5-inch width:
- Panel (a): Monotonic Recovery Latency (u_rec) vs Linear CKA Drift.
- Panel (b): Classification Forgetting vs Frozen Target Linear Probe Decodability.

Typography:
- Panel titles: 9.5 pt bold
- Axis labels: 8.8 pt bold
- Tick labels: 8.0 pt bold
- Point tags & annotations: 7.5 pt
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
CLEAN_FIGURES_DIR = os.path.join(BASE_DIR, "clean_repo", "figures")
for d in [FIGURES_DIR, CLEAN_FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

CSV_PATH = os.path.join(BASE_DIR, "supplementary", "results", "screening_results.csv")
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(BASE_DIR, "results", "screening_results.csv")

df = pd.read_csv(CSV_PATH)

CONDITIONS = ['B1', 'B3', 'B4', 'B5', 'B7']
COND_META = {
    'B1': {'short': r'$B_1$ (Reset)', 'color': '#8E44AD', 'marker': 'D'},
    'B3': {'short': r'$B_3$ (CIFAR-10)', 'color': '#2980B9', 'marker': 'o'},
    'B4': {'short': r'$B_4$ (STL-10)', 'color': '#27AE60', 'marker': 's'},
    'B5': {'short': r'$B_5$ (SVHN)', 'color': '#D35400', 'marker': '^'},
    'B7': {'short': r'$B_7$ (Noise)', 'color': '#C0392B', 'marker': 'v'}
}

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
    }

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 1.40), dpi=300,
                               gridspec_kw={'wspace': 0.32})

# =========================================================================
# Panel (a): Recovery Latency vs. CKA Drift
# =========================================================================
cka_vals = np.array([stats_data[c]['cka_mean'] for c in CONDITIONS])
urec_vals = np.array([stats_data[c]['urec_mean'] for c in CONDITIONS])

slope, intercept, r_val, p_val, std_err = stats.linregress(cka_vals, urec_vals)
r_sq = r_val ** 2

fit_x = np.linspace(0.18, 0.98, 100)
fit_y = slope * fit_x + intercept
ax1.plot(fit_x, fit_y, color='#475569', linestyle='--', lw=1.3, zorder=2,
         label=rf"Fit ($R^2 = {r_sq:.3f}, p = {p_val:.3f}$)")

callouts_a = {
    'B1': (0.24, 600.0, 'left'),
    'B3': (0.57, 3100.0, 'center'),
    'B4': (0.68, 1400.0, 'left'),
    'B5': (0.86, 5200.0, 'right'),
    'B7': (0.91, 7400.0, 'right')
}

for cond in CONDITIONS:
    meta = COND_META[cond]
    s = stats_data[cond]
    x, y = s['cka_mean'], s['urec_mean']
    ax1.errorbar(x, y, xerr=s['cka_se'], yerr=s['urec_se'],
                 fmt='none', ecolor=meta['color'], elinewidth=1.2, capsize=2.5, capthick=1.0, zorder=3)
    ax1.scatter(x, y, s=35, color=meta['color'], marker=meta['marker'], edgecolors='white', lw=0.8, zorder=4)
    
    tx, ty, ha_align = callouts_a[cond]
    ax1.text(tx, ty, f"{meta['short']}\n{int(round(y))} upd", fontsize=6.8, fontweight='bold',
             color=meta['color'], ha=ha_align, va='center', zorder=5)

ax1.set_xlim(0.12, 1.05)
ax1.set_ylim(-400, 8500)
ax1.set_xlabel(r"Phase 2 Linear CKA Drift ($D_{\mathrm{CKA}}$)", fontsize=8.5, fontweight='bold')
ax1.set_ylabel(r"Target Recovery Latency ($u_{\mathrm{rec}}$, updates)", fontsize=8.5, fontweight='bold')
ax1.set_title("(a) Recovery Latency vs. CKA Drift", fontsize=9.2, fontweight='bold', color='#0F172A', pad=6)
ax1.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')
ax1.legend(loc='upper left', frameon=True, framealpha=0.92, fontsize=7.2, edgecolor='#CBD5E1')

# =========================================================================
# Panel (b): Forgetting vs. Decodability
# =========================================================================
callouts_b = {
    'B1': (63.0, 3.0, 'right'),
    'B3': (28.0, 41.0, 'left'),
    'B4': (23.5, 52.0, 'left'),
    'B5': (9.5, 47.0, 'left'),
    'B7': (7.0, 56.0, 'left')
}

for cond in CONDITIONS:
    meta = COND_META[cond]
    s = stats_data[cond]
    x, y = s['probe_mean'], s['fb_mean']
    ax2.errorbar(x, y, xerr=s['probe_se'], yerr=s['fb_se'],
                 fmt='none', ecolor=meta['color'], elinewidth=1.2, capsize=2.5, capthick=1.0, zorder=3)
    ax2.scatter(x, y, s=35, color=meta['color'], marker=meta['marker'], edgecolors='white', lw=0.8, zorder=4)
    
    tx, ty, ha_align = callouts_b[cond]
    ax2.text(tx, ty, f"{meta['short']}\n{x:.1f}% probe", fontsize=6.8, fontweight='bold',
             color=meta['color'], ha=ha_align, va='center', zorder=5)

ax2.set_xlim(0, 75)
ax2.set_ylim(-12, 60)
ax2.set_xlabel(r"Target Linear Probe Accuracy at $\boldsymbol{\theta}_2$ (%)", fontsize=8.5, fontweight='bold')
ax2.set_ylabel(r"Classification Forgetting $F_B$ (pp)", fontsize=8.5, fontweight='bold')
ax2.set_title("(b) Representational Decoupling", fontsize=9.2, fontweight='bold', color='#0F172A', pad=6)
ax2.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')

plt.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.22)

pdf_out = os.path.join(FIGURES_DIR, "figure_monotonic_representation_spectrum.pdf")
png_out = os.path.join(FIGURES_DIR, "figure_monotonic_representation_spectrum.png")
fig.savefig(pdf_out, bbox_inches='tight', pad_inches=0.02)
fig.savefig(png_out, bbox_inches='tight', pad_inches=0.02, dpi=300)

for d in [CLEAN_FIGURES_DIR]:
    fig.savefig(os.path.join(d, "figure_monotonic_representation_spectrum.pdf"), bbox_inches='tight', pad_inches=0.02)
    fig.savefig(os.path.join(d, "figure_monotonic_representation_spectrum.png"), bbox_inches='tight', pad_inches=0.02, dpi=300)

plt.close(fig)
print(f"Figure 5 successfully generated: {pdf_out}")
