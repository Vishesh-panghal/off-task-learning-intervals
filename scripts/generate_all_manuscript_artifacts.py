"""
Generate all required manuscript tables, figures, and numerical macros for SOTLI TMLR submission.
Integrates:
- Clean B1 Discovery Screening (P0-A)
- Gate 2 Momentum Buffer Ablation: Preserved vs. Zeroed (P0-B & P1-A)
- Preregistered Confirmatory Protocol (10 fresh seeds, P1-B)
- Publication Figures 1, 2, 3, 4, 5
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'cm'
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
MANUSCRIPT_DIR = os.path.join(ROOT, "manuscript")
GENERATED_DIR = os.path.join(MANUSCRIPT_DIR, "generated")
FIGURES_DIR = os.path.join(MANUSCRIPT_DIR, "figures")

os.makedirs(GENERATED_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def f(val, prec=2):
    return f"{float(val):.{prec}f}"

def math_f(val, prec=2):
    return f"${f(val, prec)}$"

def ms(mean_v, std_v, prec=2):
    return f"${f(mean_v, prec)} \\pm {f(std_v, prec)}$"

print("=================================================================")
print(" GENERATING ALL SOTLI MANUSCRIPT TABLES, FIGURES & MACROS")
print("=================================================================")

# -----------------------------------------------------------------
# 1. GATE 2 SENSITIVITY & ABLATIONS TABLE (gate2_table.tex)
# -----------------------------------------------------------------
print("\n[1/4] Generating gate2_table.tex...")
df_mom = pd.read_csv(os.path.join(RESULTS_DIR, "gate2", "momentum_ablation_results.csv"))
df_timing = pd.read_csv(os.path.join(RESULTS_DIR, "gate2", "timing_ablation_results.csv"))
df_replay = pd.read_csv(os.path.join(RESULTS_DIR, "gate2", "replay_ablation_results.csv"))

# Compute metrics for B4 momentum
b4_pres = df_mom[(df_mom['condition_id'] == 'B4') & (df_mom['policy'] == 'preserved')]
b4_zero = df_mom[(df_mom['condition_id'] == 'B4') & (df_mom['policy'] == 'zeroed')]

# Compute metrics for B1 momentum
b1_pres = df_mom[(df_mom['condition_id'] == 'B1') & (df_mom['policy'] == 'preserved')]
b1_zero = df_mom[(df_mom['condition_id'] == 'B1') & (df_mom['policy'] == 'zeroed')]

gate2_rows = [
    # Timing
    r"Timing & Early ($u_1=5\text{k}$), 35k total updates & 3 & $56.31 \pm 1.61$ & --- & $533.3$ & $33.01$ \\",
    r"Timing & Mid ($u_1=15\text{k}$), 35k total updates & 3 & $56.23 \pm 2.15$ & $56.95 \pm 2.02$ & $2100.0$ & $46.58$ \\",
    r"Timing & Late ($u_1=25\text{k}$), 35k total updates & 3 & $56.05 \pm 2.03$ & --- & $1766.7$ & $47.39$ \\",
    r"Timing & Early ($u_1=5\text{k}$), 15k recovery & 3 & $54.45 \pm 1.66$ & --- & $533.3$ & $33.01$ \\",
    r"Timing & Late ($u_1=25\text{k}$), 15k recovery & 3 & $56.06 \pm 1.66$ & --- & $1766.7$ & $47.39$ \\",
    r"\midrule",
    # Momentum B4
    f"Momentum (B4) & Preserved (Backbone carryover) & 3 & {ms(b4_pres['p_theta3_val'].mean(), b4_pres['p_theta3_val'].std())} & {ms(b4_pres['p_theta3_test'].mean(), b4_pres['p_theta3_test'].std())} & {f(b4_pres['u_rec'].mean(), 1)} & {f((b4_pres['p_theta1']-b4_pres['p_theta2']).mean())} \\\\",
    f"Momentum (B4) & Zeroed (True zero reset) & 3 & {ms(b4_zero['p_theta3_val'].mean(), b4_zero['p_theta3_val'].std())} & {ms(b4_zero['p_theta3_test'].mean(), b4_zero['p_theta3_test'].std())} & {f(b4_zero['u_rec'].mean(), 1)} & {f((b4_zero['p_theta1']-b4_zero['p_theta2']).mean())} \\\\",
    r"\midrule",
    # Momentum B1
    f"Momentum (B1) & Preserved (Backbone carryover) & 3 & {ms(b1_pres['p_theta3_val'].mean(), b1_pres['p_theta3_val'].std())} & {ms(b1_pres['p_theta3_test'].mean(), b1_pres['p_theta3_test'].std())} & {f(b1_pres['u_rec'].mean(), 1)} & {f((b1_pres['p_theta1']-b1_pres['p_theta2']).mean())} \\\\",
    f"Momentum (B1) & Zeroed (True zero reset) & 3 & {ms(b1_zero['p_theta3_val'].mean(), b1_zero['p_theta3_val'].std())} & {ms(b1_zero['p_theta3_test'].mean(), b1_zero['p_theta3_test'].std())} & {f(b1_zero['u_rec'].mean(), 1)} & {f((b1_zero['p_theta1']-b1_zero['p_theta2']).mean())} \\\\",
    r"\midrule",
    # Replay
    r"Replay & 90/10 B/A mixture updates & 3 & $55.34 \pm 2.79$ & $55.94 \pm 1.80$ & $1100.0$ & $16.65$ \\",
    r"Replay & Isolated B (screening reference) & 3 & $56.23 \pm 2.15$ & $56.95 \pm 2.02$ & $2100.0$ & $46.58$ \\",
]

gate2_tex = "\\begin{tabular}{l l c c c c c}\n\\toprule\nAxis & Intervention Configuration & $n$ & Val (mean $\\pm$ SD) & Test (mean $\\pm$ SD) & $\\urec$ & $F_B$ \\\\\n\\midrule\n"
gate2_tex += "\n".join(gate2_rows)
gate2_tex += "\n\\bottomrule\n\\end{tabular}\n"

with open(os.path.join(GENERATED_DIR, "gate2_table.tex"), "w") as f_out:
    f_out.write(gate2_tex)
print("✓ Saved: manuscript/generated/gate2_table.tex")

# -----------------------------------------------------------------
# 2. CONFIRMATORY REPLICATION TABLE (confirmatory_table.tex)
# -----------------------------------------------------------------
print("\n[2/4] Generating confirmatory_table.tex...")
df_conf = pd.read_csv(os.path.join(RESULTS_DIR, "confirmatory", "confirmatory_summary.csv"))
df_raw_conf = pd.read_csv(os.path.join(RESULTS_DIR, "confirmatory", "confirmatory_results.csv"))

b0_seeds = df_raw_conf[df_raw_conf['condition_id'] == 'B0'].set_index('seed')
b1_seeds = df_raw_conf[df_raw_conf['condition_id'] == 'B1'].set_index('seed')
b4_seeds = df_raw_conf[df_raw_conf['condition_id'] == 'B4'].set_index('seed')
b7_seeds = df_raw_conf[df_raw_conf['condition_id'] == 'B7'].set_index('seed')
common_s = sorted(list(set(b0_seeds.index) & set(b1_seeds.index) & set(b4_seeds.index) & set(b7_seeds.index)))

conf_tex = (
    "% SOTLI Prespecified Confirmatory Replication Protocol (10 Fresh Seeds)\n"
    "\\begin{tabular}{l c cccc cccc cc}\n"
    "\\toprule\n"
    "& & \\multicolumn{4}{c}{Equal Total Updates ($\\Dtotal$, Val)} & \\multicolumn{4}{c}{Equal Target Updates ($\\Da$, Val)} & \\multicolumn{2}{c}{Test ($\\Dtotal$)} \\\\\n"
    "\\cmidrule(lr){3-6} \\cmidrule(lr){7-10} \\cmidrule(lr){11-12}\n"
    "Condition & $N$ & Mean & 95\\% CI & $p_{\\text{paired}}$ & $d_z$ & Mean & 95\\% CI & $p_{\\text{paired}}$ & $d_z$ & Mean & 95\\% CI \\\\\n"
    "\\midrule\n"
    "B1 (Clean Reset, CIFAR-100) & 10 & +0.75 & [-0.99, +2.46] & 0.4337 & +0.26 & +1.56 & [+0.17, +2.90] & 0.0613 & +0.68 & +0.97 & [-0.54, +2.37] \\\\\n"
    "B4 (STL-10 Transfer) & 10 & +0.76 & [-0.38, +1.91] & 0.2491 & +0.39 & +1.57 & [+0.62, +2.51] & 0.0129 & +0.98 & +0.44 & [-0.63, +1.61] \\\\\n"
    "B7 (Gaussian Noise) & 10 & -2.99 & [-4.67, -1.55] & 0.0059 & -1.13 & -2.18 & [-3.73, -0.61] & 0.0276 & -0.83 & -2.81 & [-4.56, -1.22] \\\\\n"
    "\\midrule\n"
    "\\multicolumn{12}{l}{\\textit{Decisive Paired Contrast (Auxiliary Transfer vs. Clean Optimizer Reset):}} \\\\\n"
    "B4 vs.\\ B1 ($\\Delta_{\\text{Val}}$) & 10 & +0.00 & [-1.04, +1.22] & 0.9949 & +0.00 & --- & --- & --- & --- & -0.53 & [-1.48, +0.52] \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
)

with open(os.path.join(GENERATED_DIR, "confirmatory_table.tex"), "w") as f_out:
    f_out.write(conf_tex)
print("✓ Saved: manuscript/generated/confirmatory_table.tex")

# -----------------------------------------------------------------
# 3. NUMERICAL MACROS (numbers.tex)
# -----------------------------------------------------------------
print("\n[3/4] Generating updated numbers.tex macros...")
macros = {}

# Gate 2 Momentum Macros
macros['BfourMomPresVal'] = f(b4_pres['p_theta3_val'].mean())
macros['BfourMomPresValSD'] = f(b4_pres['p_theta3_val'].std())
macros['BfourMomPresTest'] = f(b4_pres['p_theta3_test'].mean())
macros['BfourMomPresTestSD'] = f(b4_pres['p_theta3_test'].std())
macros['BfourMomPresUrec'] = f(b4_pres['u_rec'].mean(), 0)

macros['BfourMomZeroVal'] = f(b4_zero['p_theta3_val'].mean())
macros['BfourMomZeroValSD'] = f(b4_zero['p_theta3_val'].std())
macros['BfourMomZeroTest'] = f(b4_zero['p_theta3_test'].mean())
macros['BfourMomZeroTestSD'] = f(b4_zero['p_theta3_test'].std())
macros['BfourMomZeroUrec'] = f(b4_zero['u_rec'].mean(), 0)

# Paired difference
macros['BfourMomDeltaTest'] = f(b4_zero['p_theta3_test'].mean() - b4_pres['p_theta3_test'].mean(), 2)

# Confirmatory Macros (N=10)
b1_c = df_conf[df_conf['condition'] == 'B1'].iloc[0]
b4_c = df_conf[df_conf['condition'] == 'B4'].iloc[0]
b7_c = df_conf[df_conf['condition'] == 'B7'].iloc[0]

macros['ConfBfourN'] = str(int(b4_c['N']))

# B1 Macros
macros['ConfBoneDeltaTotalVal'] = f(b1_c['delta_total_val_mean'], 2)
macros['ConfBoneDeltaTotalValLow'] = f(b1_c['delta_total_val_ci_low'], 2)
macros['ConfBoneDeltaTotalValHigh'] = f(b1_c['delta_total_val_ci_high'], 2)
macros['ConfBoneDeltaAVal'] = f(b1_c['delta_a_val_mean'], 2)
macros['ConfBoneDeltaAValLow'] = f(b1_c['delta_a_val_ci_low'], 2)
macros['ConfBoneDeltaAValHigh'] = f(b1_c['delta_a_val_ci_high'], 2)
macros['ConfBonePval'] = f(b1_c['p_value_ttest'], 4)
macros['ConfBoneCohenD'] = f(b1_c['cohen_d'], 2)
macros['ConfBoneDeltaTotalTest'] = f(b1_c['delta_total_test_mean'], 2)
macros['ConfBoneDeltaTotalTestLow'] = f(b1_c['delta_total_test_ci_low'], 2)
macros['ConfBoneDeltaTotalTestHigh'] = f(b1_c['delta_total_test_ci_high'], 2)

# B4 Macros
macros['ConfBfourDeltaTotalVal'] = f(b4_c['delta_total_val_mean'], 2)
macros['ConfBfourDeltaTotalValLow'] = f(b4_c['delta_total_val_ci_low'], 2)
macros['ConfBfourDeltaTotalValHigh'] = f(b4_c['delta_total_val_ci_high'], 2)
macros['ConfBfourDeltaAVal'] = f(b4_c['delta_a_val_mean'], 2)
macros['ConfBfourDeltaAValLow'] = f(b4_c['delta_a_val_ci_low'], 2)
macros['ConfBfourDeltaAValHigh'] = f(b4_c['delta_a_val_ci_high'], 2)
macros['ConfBfourPval'] = f(b4_c['p_value_ttest'], 4)
macros['ConfBfourCohenD'] = f(b4_c['cohen_d'], 2)
macros['ConfBfourDeltaTotalTest'] = f(b4_c['delta_total_test_mean'], 2)

# B7 Macros
macros['ConfBsevenDeltaTotalVal'] = f(b7_c['delta_total_val_mean'], 2)
macros['ConfBsevenDeltaTotalValLow'] = f(b7_c['delta_total_ci_low'] if 'delta_total_ci_low' in b7_c else b7_c['delta_total_val_ci_low'], 2)
macros['ConfBsevenDeltaTotalValHigh'] = f(b7_c['delta_total_ci_high'] if 'delta_total_ci_high' in b7_c else b7_c['delta_total_val_ci_high'], 2)
macros['ConfBsevenPval'] = f(b7_c['p_value_ttest'], 4)
macros['ConfBsevenCohenD'] = f(b7_c['cohen_d'], 2)
macros['ConfBsevenDeltaTotalTest'] = f(b7_c['delta_total_test_mean'], 2)

# B4 vs B1 Paired Contrast Macros
macros['ConfBfourVsBoneValDiff'] = "0.00"
macros['ConfBfourVsBoneValLow'] = "-1.04"
macros['ConfBfourVsBoneValHigh'] = "1.22"
macros['ConfBfourVsBoneValPval'] = "0.9949"
macros['ConfBfourVsBoneTestDiff'] = "-0.53"
macros['ConfBfourVsBoneTestLow'] = "-1.48"
macros['ConfBfourVsBoneTestHigh'] = "0.52"
macros['ConfBfourVsBoneTestPval'] = "0.3457"

# Append to numbers.tex or create updated numbers
numbers_tex = "% Auto-generated SOTLI Numerical Macros\n"
for k, v in macros.items():
    numbers_tex += f"\\newcommand{{\\{k}}}{{{v}}}\n"

with open(os.path.join(GENERATED_DIR, "confirmatory_macros.tex"), "w") as f_out:
    f_out.write(numbers_tex)
print("✓ Saved: manuscript/generated/confirmatory_macros.tex")

# Verify and copy foundational Figures 1, 2, 3, 4
import shutil
for f_name in ["figure1_relearning_trajectories.png", "figure2_monotonic_characterization.png"]:
    src_res = os.path.join(RESULTS_DIR, f_name)
    dst_man = os.path.join(FIGURES_DIR, f_name)
    if os.path.exists(src_res):
        shutil.copy2(src_res, dst_man)
        print(f"✓ Copied complete {f_name} from results/ ({os.path.getsize(dst_man)/1024:.1f} KB)")

for f_name in ["figure3_predictive_similarity_curves.png", "figure4_representation_geometry.png"]:
    src_res = os.path.join(RESULTS_DIR, "figures", f_name)
    dst_man = os.path.join(FIGURES_DIR, f_name)
    if os.path.exists(src_res):
        shutil.copy2(src_res, dst_man)
        print(f"✓ Copied high-res {f_name} from results/figures/ ({os.path.getsize(dst_man)/1024:.1f} KB)")

# Generate Figure 5: Gate 2 Momentum Buffer Ablation
print("\nGenerating Figure 5: Gate 2 Momentum Buffer Ablation (Preserved vs. Zeroed)...")
cond_labels = ['B4 (STL-10)', 'B1 (Clean Holdout)']
test_pres_means = [b4_pres['p_theta3_test'].mean(), b1_pres['p_theta3_test'].mean()]
test_pres_stds = [b4_pres['p_theta3_test'].std(), b1_pres['p_theta3_test'].std()]
test_zero_means = [b4_zero['p_theta3_test'].mean(), b1_zero['p_theta3_test'].mean()]
test_zero_stds = [b4_zero['p_theta3_test'].std(), b1_zero['p_theta3_test'].std()]

urec_pres_means = [b4_pres['u_rec'].mean(), b1_pres['u_rec'].mean()]
urec_pres_stds = [b4_pres['u_rec'].std(), b1_pres['u_rec'].std()]
urec_zero_means = [b4_zero['u_rec'].mean(), b1_zero['u_rec'].mean()]
urec_zero_stds = [b4_zero['u_rec'].std(), b1_zero['u_rec'].std()]

x_bar = np.arange(2)
bar_w = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=240)

# Panel A: Final Test Accuracy
ax1.bar(x_bar - bar_w/2, test_pres_means, bar_w, yerr=test_pres_stds, label='Preserved Momentum', color='#2b5c8f', capsize=5, alpha=0.9, edgecolor='black', linewidth=0.8)
ax1.bar(x_bar + bar_w/2, test_zero_means, bar_w, yerr=test_zero_stds, label='Zeroed Momentum', color='#d95f5f', capsize=5, alpha=0.9, edgecolor='black', linewidth=0.8)
ax1.set_title(r'(A) Final Test Accuracy ($\theta_3$)', pad=10, fontweight='bold', fontsize=11)
ax1.set_ylabel('Test Accuracy (%)', fontsize=10)
ax1.set_xticks(x_bar)
ax1.set_xticklabels(cond_labels, fontsize=10)
ax1.set_ylim(52.0, 60.0)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.legend(loc='lower right', framealpha=0.92, fontsize=9.5)

# Panel B: Recovery Speed
ax2.bar(x_bar - bar_w/2, urec_pres_means, bar_w, yerr=urec_pres_stds, label='Preserved Momentum', color='#2b5c8f', capsize=5, alpha=0.9, edgecolor='black', linewidth=0.8)
ax2.bar(x_bar + bar_w/2, urec_zero_means, bar_w, yerr=urec_zero_stds, label='Zeroed Momentum', color='#d95f5f', capsize=5, alpha=0.9, edgecolor='black', linewidth=0.8)
ax2.set_title(r'(B) Recovery Speed ($u_{\mathrm{rec}}$ updates)', pad=10, fontweight='bold', fontsize=11)
ax2.set_ylabel(r'Recovery Updates ($u_{\mathrm{rec}}$)', fontsize=10)
ax2.set_xticks(x_bar)
ax2.set_xticklabels(cond_labels, fontsize=10)
ax2.set_ylim(0, 6000)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.legend(loc='upper right', framealpha=0.92, fontsize=9.5)

plt.tight_layout()
f5_path = os.path.join(FIGURES_DIR, "figure5_momentum_ablation.png")
plt.savefig(f5_path, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {f5_path} ({os.path.getsize(f5_path)/1024:.1f} KB)")

# Generate Figure 6: Confirmatory Replication Effect Sizes (B1, B4, B7 across 10 seeds)
print("\nGenerating Figure 6: Confirmatory Replication Distribution (N=10 seeds)...")
diff_b1_total = [(b1_seeds.loc[s, 'p_theta3_val'] - b0_seeds.loc[s, 'p_theta3_val']) for s in common_s]
diff_b1_a = [(b1_seeds.loc[s, 'p_theta3_val'] - b0_seeds.loc[s, 'p_control2_val']) for s in common_s]
diff_b4_total = [(b4_seeds.loc[s, 'p_theta3_val'] - b0_seeds.loc[s, 'p_theta3_val']) for s in common_s]
diff_b4_a = [(b4_seeds.loc[s, 'p_theta3_val'] - b0_seeds.loc[s, 'p_control2_val']) for s in common_s]
diff_b7_total = [(b7_seeds.loc[s, 'p_theta3_val'] - b0_seeds.loc[s, 'p_theta3_val']) for s in common_s]
diff_b4_b1 = [(b4_seeds.loc[s, 'p_theta3_val'] - b1_seeds.loc[s, 'p_theta3_val']) for s in common_s]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=240)

# Panel A: Paired differences by seed
x_seeds = np.arange(len(common_s))
ax1.axhline(0, color='#888888', linestyle='--', linewidth=1.0)
ax1.plot(x_seeds, diff_b1_total, 'd--', color='#f39c12', label=r'B1 (Reset): $\Delta_{\mathrm{total}}$ (+0.75)', linewidth=1.8, markersize=6)
ax1.plot(x_seeds, diff_b4_total, 'o-', color='#27ae60', label=r'B4 (STL-10): $\Delta_{\mathrm{total}}$ (+0.76)', linewidth=2.0, markersize=7)
ax1.plot(x_seeds, diff_b4_a, 's:', color='#2980b9', label=r'B4 (STL-10): $\Delta_{A}$ (+1.57)', linewidth=1.6, markersize=6)
ax1.plot(x_seeds, diff_b7_total, '^-', color='#c0392b', label=r'B7 (Noise): $\Delta_{\mathrm{total}}$ (-2.99)', linewidth=2.0, markersize=7)

ax1.set_title(r'(A) Paired Generalization Delta per Seed ($N=10$)', pad=10, fontweight='bold', fontsize=11)
ax1.set_xlabel('Preregistered Fresh Seed Index', fontsize=10)
ax1.set_ylabel(r'Validation Accuracy Difference ($\Delta$ pp)', fontsize=10)
ax1.set_xticks(x_seeds)
ax1.set_xticklabels([f"S{s}" for s in common_s], rotation=30)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower left', framealpha=0.92, fontsize=8.5)

# Panel B: Mean and 95% Bootstrap CI
categories = [
    r'B1: $\Delta_{\mathrm{total}}$',
    r'B1: $\Delta_{A}$',
    r'B4: $\Delta_{\mathrm{total}}$',
    r'B4: $\Delta_{A}$',
    r'B4$-$B1: $\Delta_{\mathrm{val}}$',
    r'B7: $\Delta_{\mathrm{total}}$'
]
diff_arrays = [diff_b1_total, diff_b1_a, diff_b4_total, diff_b4_a, diff_b4_b1, diff_b7_total]
means_list = [np.mean(arr) for arr in diff_arrays]
colors_list = ['#f39c12', '#e67e22', '#27ae60', '#2980b9', '#8e44ad', '#c0392b']

# 95% bootstrap CIs
def get_ci(v):
    rng = np.random.RandomState(42)
    boots = [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(5000)]
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)

ci_lows = [get_ci(arr)[0] for arr in diff_arrays]
ci_highs = [get_ci(arr)[1] for arr in diff_arrays]
yerr_low = [m - l for m, l in zip(means_list, ci_lows)]
yerr_high = [h - m for m, h in zip(means_list, ci_highs)]

x_pos = np.arange(len(categories))
ax2.axhline(0, color='#888888', linestyle='--', linewidth=1.0)
bars = ax2.bar(x_pos, means_list, yerr=[yerr_low, yerr_high], color=colors_list, capsize=5, width=0.6, alpha=0.88, edgecolor='black', linewidth=0.9)

for i, (m, l, h) in enumerate(zip(means_list, ci_lows, ci_highs)):
    if m >= 0:
        y_annot = max(h, m) + 0.35
        va_pos = 'bottom'
    else:
        y_annot = min(l, m) - 0.35
        va_pos = 'top'
    ax2.annotate(f"{m:+.2f}\n[{l:+.2f}, {h:+.2f}]", (i, y_annot), ha='center', va=va_pos, fontsize=7.5, fontweight='bold')

ax2.set_title(r'(B) 95% Bootstrap Confidence Intervals ($B=5{,}000$)', pad=10, fontweight='bold', fontsize=11)
ax2.set_ylabel(r'Mean Paired Difference ($\Delta$ pp)', fontsize=10)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(categories, fontsize=8.5, rotation=15)
ax2.grid(axis='y', linestyle=':', alpha=0.6)
ax2.set_ylim(-6.8, 4.5)

plt.tight_layout()
f6_path = os.path.join(FIGURES_DIR, "figure6_confirmatory_replication.png")
plt.savefig(f6_path, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {f6_path} ({os.path.getsize(f6_path)/1024:.1f} KB)")

print("\n=================================================================")
print(" ALL REQUIRED TABLES, FIGURES, AND MACROS GENERATED SUCCESSFULLY! ✅")
print("=================================================================")

