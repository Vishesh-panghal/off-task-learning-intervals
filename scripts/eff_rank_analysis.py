"""
Representation Rank & Geometry Analysis Engine for SOTLI (Gate 3).
Visualizes:
1. Effective Representation Rank (EffRank) evolution across transitions: theta_1 -> theta_2 -> theta_3.
2. Diagnostic linear probe accuracy vs. apparent head drop F_B at theta_2.
Generates Publication Figure 4: figure4_representation_geometry.png.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCREENING_CSV = os.path.join(PROJECT_ROOT, "results", "screening_results.csv")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")


def main():
    print("=================================================================")
    print(" SOTLI GATE 3: REPRESENTATION RANK & GEOMETRY ANALYSIS")
    print(" Target Venue: TMLR")
    print("=================================================================\n")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = pd.read_csv(SCREENING_CSV)

    conditions = ['B0', 'B1', 'B3', 'B4', 'B5', 'B7']
    labels = {
        'B0': 'B0 (Continuous)',
        'B1': 'B1 (Head-Switch)',
        'B3': 'B3 (CIFAR-10)',
        'B4': 'B4 (STL-10)',
        'B5': 'B5 (SVHN)',
        'B7': 'B7 (Noise)'
    }
    colors = {
        'B0': '#7f8c8d', 'B1': '#2980b9', 'B3': '#27ae60',
        'B4': '#d35400', 'B5': '#8e44ad', 'B7': '#c0392b'
    }

    # Aggregate metrics
    summary = df.groupby('condition_id').agg({
        'eff_rank_theta1': ['mean', 'std'],
        'eff_rank_theta2': ['mean', 'std'],
        'eff_rank_theta3': ['mean', 'std'],
        'p_diagnostic_probe': ['mean', 'std'],
        'p_theta2': ['mean', 'std'],
        'f_b': ['mean', 'std']
    })

    print("--- Effective Rank Geometry Summary ---")
    for c in conditions:
        r1 = summary.loc[c, ('eff_rank_theta1', 'mean')]
        r2 = summary.loc[c, ('eff_rank_theta2', 'mean')]
        r3 = summary.loc[c, ('eff_rank_theta3', 'mean')]
        probe = summary.loc[c, ('p_diagnostic_probe', 'mean')]
        head = summary.loc[c, ('p_theta2', 'mean')]
        print(f"Condition {c:2s} | θ1 Rank: {r1:.1f} -> θ2 Rank: {r2:.1f} -> θ3 Rank: {r3:.1f} | θ2 Frozen Head: {head:.1f}% vs Probe: {probe:.1f}%")

    # -------------------------------------------------------------
    # Generate Publication Figure 4
    # -------------------------------------------------------------
    plt.rcParams.update({'font.size': 11, 'mathtext.fontset': 'cm', 'font.family': 'sans-serif'})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel A: Effective Rank Progression across Transitions
    x_checkpoints = [1, 2, 3]
    checkpoint_names = [r'$\theta_1$' + '\n(Pre-Interval)', r'$\theta_2$' + '\n(Post-Interval)', r'$\theta_3$' + '\n(Terminal)']

    for c in conditions:
        means = [
            summary.loc[c, ('eff_rank_theta1', 'mean')],
            summary.loc[c, ('eff_rank_theta2', 'mean')],
            summary.loc[c, ('eff_rank_theta3', 'mean')]
        ]
        stds = [
            summary.loc[c, ('eff_rank_theta1', 'std')],
            summary.loc[c, ('eff_rank_theta2', 'std')],
            summary.loc[c, ('eff_rank_theta3', 'std')]
        ]
        ax1.errorbar(
            x_checkpoints, means, yerr=stds, label=labels[c],
            color=colors[c], marker='o', linewidth=2.0, markersize=7, capsize=4
        )

    ax1.set_xticks(x_checkpoints)
    ax1.set_xticklabels(checkpoint_names, fontsize=11, fontweight='bold')
    ax1.set_ylabel(r'Effective Representation Rank $\mathrm{EffRank}(Z)$', fontsize=12)
    ax1.set_title('(A) Effective Rank Trajectory Across Transitions', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(frameon=True, facecolor='white', loc='lower left', fontsize=10)

    # Panel B: Diagnostic Probe vs Frozen Head Accuracy at theta_2
    x_pos = np.arange(len(conditions))
    width = 0.35

    probe_means = [summary.loc[c, ('p_diagnostic_probe', 'mean')] for c in conditions]
    probe_stds = [summary.loc[c, ('p_diagnostic_probe', 'std')] for c in conditions]
    head_means = [summary.loc[c, ('p_theta2', 'mean')] for c in conditions]
    head_stds = [summary.loc[c, ('p_theta2', 'std')] for c in conditions]

    rects1 = ax2.bar(x_pos - width/2, head_means, width, yerr=head_stds, label='Frozen Classifier Head A', color='#95a5a6', capsize=3)
    rects2 = ax2.bar(x_pos + width/2, probe_means, width, yerr=probe_stds, label=r'Linear Diagnostic Probe on Backbone $w_2$', color='#2c3e50', capsize=3)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(conditions, fontsize=11, fontweight='bold')
    ax2.set_ylabel('Accuracy on Task A Validation (%)', fontsize=12)
    ax2.set_title(r'(B) Representation Retention vs. Head Misalignment at $\theta_2$', fontsize=12, fontweight='bold', pad=10)
    ax2.grid(True, linestyle='--', alpha=0.4, axis='y')
    ax2.legend(frameon=True, facecolor='white', loc='upper right', fontsize=10)

    fig.suptitle('SOTLI Gate 3: Representation Geometry & Backbone Diagnostic Analysis', fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig_path = os.path.join(FIGURES_DIR, "figure4_representation_geometry.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"\n✓ Generated Figure 4 -> {fig_path}")

    print("\n=================================================================")
    print(" REPRESENTATION RANK ANALYSIS COMPLETED SUCCESSFULLY! 🎉")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
