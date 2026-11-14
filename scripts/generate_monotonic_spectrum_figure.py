#!/usr/bin/env python3
"""
Generate publication-grade Figure 5: Multi-Domain Recovery and Decoupling Spectrum.
Redesigned with authentic Seaborn statistical aesthetics:
- Panel (a): Recovery Latency (u_rec) vs Linear CKA Drift.
- Panel (b): Representational Decoupling: Classification Forgetting vs Probe Decodability.

Features:
- Seaborn 'ticks' theme with clean L-frames (sns.despine)
- Seaborn Deep palette for condition markers and labels
- Clean non-bold data point callouts with alternating vertical placement
- No artificial boxes or clutter; clean whitespace preservation
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks", font="sans-serif")
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['mathtext.fontset'] = 'cm'

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..")) if os.path.basename(BASE_DIR) == "clean_repo" else BASE_DIR

FIGURE_DIRS = [
    os.path.join(BASE_DIR, "figures"),
    os.path.join(ROOT_DIR, "figures"),
    os.path.join(ROOT_DIR, "clean_repo", "figures")
]
FIGURE_DIRS = list(dict.fromkeys([d for d in FIGURE_DIRS if os.path.exists(os.path.dirname(d))]))
for d in FIGURE_DIRS:
    os.makedirs(d, exist_ok=True)

CSV_CANDIDATES = [
    os.path.join(BASE_DIR, "supplementary", "results", "screening_results.csv"),
    os.path.join(ROOT_DIR, "supplementary", "results", "screening_results.csv"),
    os.path.join(BASE_DIR, "results", "screening_results.csv"),
    os.path.join(ROOT_DIR, "results", "screening_results.csv"),
    os.path.join(ROOT_DIR, "clean_repo", "results", "screening_results.csv")
]

CSV_PATH = None
for p in CSV_CANDIDATES:
    if os.path.exists(p):
        CSV_PATH = p
        break

if CSV_PATH is None:
    raise FileNotFoundError("Could not find screening_results.csv in any candidate location.")

df = pd.read_csv(CSV_PATH)

CONDITIONS = ['B1', 'B3', 'B4', 'B5', 'B7']
COND_META = {
    'B1': {'short': r'$B_1$ (Reset)', 'color': '#8172b3', 'marker': 'D'},
    'B3': {'short': r'$B_3$ (CIFAR-10)', 'color': '#4c72b0', 'marker': 'o'},
    'B4': {'short': r'$B_4$ (STL-10)', 'color': '#55a868', 'marker': 's'},
    'B5': {'short': r'$B_5$ (SVHN)', 'color': '#dd8452', 'marker': '^'},
    'B7': {'short': r'$B_7$ (Noise)', 'color': '#c44e52', 'marker': 'v'}
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.50), dpi=300,
                               gridspec_kw={'wspace': 0.34})

C_DARK = '#1e293b'

# =========================================================================
# Panel (a): Recovery Latency vs. CKA Drift
# =========================================================================
cka_vals = np.array([stats_data[c]['cka_mean'] for c in CONDITIONS])
urec_vals = np.array([stats_data[c]['urec_mean'] for c in CONDITIONS])

slope, intercept, r_val, p_val, std_err = stats.linregress(cka_vals, urec_vals)
r_sq = r_val ** 2

fit_x = np.linspace(0.18, 0.98, 100)
fit_y = slope * fit_x + intercept
ax1.plot(fit_x, fit_y, color='#64748b', linestyle='--', lw=1.4, zorder=2)

# Two-line compact fit label in open upper-left quadrant
ax1.text(0.18, 7900.0, f"Linear Fit\n$R^2 = {r_sq:.3f},\\; p = {p_val:.3f}$",
         fontsize=7.2, color='#475569', fontweight='medium', va='top', linespacing=1.2)

callouts_a = {
    'B1': (0.22, 600.0, 'left', 'bottom'),
    'B3': (0.47, 3400.0, 'center', 'bottom'),
    'B4': (0.68, 1100.0, 'center', 'top'),
    'B5': (0.97, 4900.0, 'left', 'center'),
    'B7': (0.97, 6800.0, 'left', 'center')
}

for cond in CONDITIONS:
    meta = COND_META[cond]
    s = stats_data[cond]
    x, y = s['cka_mean'], s['urec_mean']
    ax1.errorbar(x, y, xerr=s['cka_se'], yerr=s['urec_se'],
                 fmt='none', ecolor=meta['color'], elinewidth=1.2, capsize=3.0, capthick=1.0, zorder=3)
    ax1.scatter(x, y, s=40, color=meta['color'], marker=meta['marker'], edgecolors='white', lw=0.9, zorder=4)
    
    tx, ty, ha_align, va_align = callouts_a[cond]
    ax1.text(tx, ty, f"{meta['short']}\n{int(round(y))} upd", fontsize=7.2,
             color=meta['color'], ha=ha_align, va=va_align, zorder=5, fontweight='medium')

ax1.set_xlim(0.12, 1.18)
ax1.set_ylim(-400, 8500)
ax1.set_yticks([0, 2000, 4000, 6000, 8000])
ax1.set_yticklabels(["0", "2k", "4k", "6k", "8k"], fontsize=7.8)
ax1.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax1.set_xticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7.8)
ax1.set_xlabel(r"Phase 2 Linear CKA Drift ($D_{\mathrm{CKA}}$)", fontsize=8.6, color=C_DARK)
ax1.set_ylabel(r"Target Recovery Latency ($u_{\mathrm{rec}}$, updates)", fontsize=8.6, color=C_DARK)
ax1.set_title("(a) Recovery Latency vs. CKA Drift", fontsize=9.2, fontweight='semibold', color=C_DARK, pad=10)
ax1.grid(axis='y', linestyle='--', alpha=0.35, color='#cbd5e1')
sns.despine(ax=ax1, trim=False)

# =========================================================================
# Panel (b): Forgetting vs. Decodability
# =========================================================================
callouts_b = {
    'B1': (66.1, 5.0, 'center', 'bottom'),
    'B3': (28.5, 45.2, 'left', 'center'),
    'B4': (21.0, 56.5, 'left', 'bottom'),
    'B5': (5.5, 36.0, 'left', 'top'),
    'B7': (2.5, 56.5, 'left', 'bottom')
}

for cond in CONDITIONS:
    meta = COND_META[cond]
    s = stats_data[cond]
    x, y = s['probe_mean'], s['fb_mean']
    ax2.errorbar(x, y, xerr=s['probe_se'], yerr=s['fb_se'],
                 fmt='none', ecolor=meta['color'], elinewidth=1.2, capsize=3.0, capthick=1.0, zorder=3)
    ax2.scatter(x, y, s=40, color=meta['color'], marker=meta['marker'], edgecolors='white', lw=0.9, zorder=4)
    
    tx, ty, ha_align, va_align = callouts_b[cond]
    ax2.text(tx, ty, f"{meta['short']}\n{x:.1f}% probe", fontsize=7.2,
             color=meta['color'], ha=ha_align, va=va_align, zorder=5, fontweight='medium')

ax2.set_xlim(0, 75)
ax2.set_ylim(-12, 68)
ax2.set_yticks([0, 15, 30, 45, 60])
ax2.set_yticklabels(["0", "15", "30", "45", "60"], fontsize=7.8)
ax2.set_xticks([0, 20, 40, 60])
ax2.set_xticklabels(["0%", "20%", "40%", "60%"], fontsize=7.8)
ax2.set_xlabel(r"Target Linear Probe Accuracy at $\boldsymbol{\theta}_2$", fontsize=8.6, color=C_DARK)
ax2.set_ylabel(r"Classification Forgetting $F_B$ (pp)", fontsize=8.6, color=C_DARK)
ax2.set_title("(b) Representational Decoupling", fontsize=9.2, fontweight='semibold', color=C_DARK, pad=10)
ax2.grid(axis='y', linestyle='--', alpha=0.35, color='#cbd5e1')
sns.despine(ax=ax2, trim=False)

plt.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.18)

for d in FIGURE_DIRS:
    pdf_p = os.path.join(d, "figure_monotonic_representation_spectrum.pdf")
    png_p = os.path.join(d, "figure_monotonic_representation_spectrum.png")
    fig.savefig(pdf_p, bbox_inches='tight', pad_inches=0.04)
    fig.savefig(png_p, bbox_inches='tight', pad_inches=0.04, dpi=300)
    print(f"Saved: {pdf_p}")

plt.close(fig)
print("Figure 5 generation complete.")
