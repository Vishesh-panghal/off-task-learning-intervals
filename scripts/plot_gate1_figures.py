"""
Plotting Script for SOTLI Stage 1 (Gate 1).
Generates publication-quality figures for TMLR:
1. Figure 1: Relearning Dynamics Curves (Dense tracking of Phase 3 across 5 seeds).
2. Figure 2: The Monotonic Representation Continuum (CKA Drift vs Recovery Velocity & Diagnostic Probe Retention).
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Styling for TMLR standard
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2.0,
    'lines.markersize': 7,
    'grid.alpha': 0.3,
})

RESULTS_DIR = "./results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

CSV_PATH = os.path.join(RESULTS_DIR, "screening_results.csv")
TRAJ_DIR = os.path.join(RESULTS_DIR, "trajectories")

# Colors & labels for conditions
PALETTE = {
    'B0': '#2c3e50',   # Dark Slate Navy
    'B1': '#8e44ad',   # Royal Purple
    'B3': '#2980b9',   # Blue (CIFAR-10)
    'B4': '#27ae60',   # Emerald Green (STL-10)
    'B5': '#e67e22',   # Amber Orange (SVHN)
    'B7': '#c0392b',   # Crimson Red (Gaussian Noise)
}

LABELS = {
    'B0': 'B0: Continuous Baseline (Task A)',
    'B1': 'B1: Head-Switch Control (Task A + temp head)',
    'B3': 'B3: Related Domain (CIFAR-10)',
    'B4': 'B4: Distinct Domain (STL-10)',
    'B5': 'B5: Distant Domain (SVHN)',
    'B7': 'B7: Synthetic Control (Gaussian Noise)',
}


def plot_relearning_trajectories():
    """Generates Figure 1: Mean validation trajectories during Phase 3 with std error bands."""
    print("Generating Figure 1: Phase 3 Relearning Trajectories...")
    conditions = ['B1', 'B3', 'B4', 'B5', 'B7']
    seeds = [42, 123, 456, 789, 2026]

    # Load baseline B0 endpoints for reference line
    df = pd.read_csv(CSV_PATH)
    b0_mean = df[df['condition_id'] == 'B0']['p_theta3_val'].mean()
    p_before_mean = df['p_theta1'].mean()

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    # Reference lines
    ax.axhline(b0_mean, color='#2c3e50', linestyle='--', linewidth=1.8, label=f'B0 Continuous Target (Mean: {b0_mean:.2f}%)')
    ax.axhline(p_before_mean, color='#7f8c8d', linestyle=':', linewidth=1.5, label=f'θ₁ Pre-Interval Target (Mean: {p_before_mean:.2f}%)')

    for cond in conditions:
        traj_matrix = []
        for s in seeds:
            fpath = os.path.join(TRAJ_DIR, f"trajectory_{cond}_seed{s}.csv")
            if os.path.exists(fpath):
                tdf = pd.read_csv(fpath)
                traj_matrix.append(tdf['val_acc'].values)

        if len(traj_matrix) == 0:
            continue

        traj_matrix = np.array(traj_matrix)
        steps = tdf['rel_update'].values
        mean_curve = np.mean(traj_matrix, axis=0)
        std_err = np.std(traj_matrix, axis=0) / np.sqrt(len(traj_matrix))

        ax.plot(steps, mean_curve, color=PALETTE[cond], label=LABELS[cond], linewidth=2.2)
        ax.fill_between(steps, mean_curve - std_err, mean_curve + std_err, color=PALETTE[cond], alpha=0.15)

    ax.set_title("Phase 3 Relearning Dynamics Across 5 Seeds (N=30 Runs)", pad=12, fontweight='bold')
    ax.set_xlabel("Relearning Updates on Task A ($u_2$ with batch size 128)")
    ax.set_ylabel("Target Task Validation Accuracy (%)")
    ax.set_xlim(0, 15000)
    ax.set_ylim(25, 62)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', framealpha=0.95, edgecolor='#bdc3c7')

    plt.tight_layout()
    out_file = os.path.join(FIGURES_DIR, "figure1_relearning_trajectories.png")
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_file}")


def plot_monotonic_characterization():
    """Generates Figure 2: Representation Drift vs Recovery Velocity and Diagnostic Probe Retention."""
    print("Generating Figure 2: The Monotonic Representation Continuum...")
    df = pd.read_csv(CSV_PATH)

    conditions = ['B1', 'B3', 'B4', 'B5', 'B7']
    cond_names = ['B1 (Head-Switch)', 'B3 (CIFAR-10)', 'B4 (STL-10)', 'B5 (SVHN)', 'B7 (Noise)']

    cka_means, cka_errs = [], []
    urec_means, urec_errs = [], []
    probe_means, probe_errs = [], []
    fb_means, fb_errs = [], []

    for cond in conditions:
        sub = df[df['condition_id'] == cond]
        cka_means.append(sub['cka_drift'].mean())
        cka_errs.append(sub['cka_drift'].std() / np.sqrt(len(sub)))
        urec_means.append(sub['u_rec'].mean())
        urec_errs.append(sub['u_rec'].std() / np.sqrt(len(sub)))
        probe_means.append(sub['p_diagnostic_probe'].mean())
        probe_errs.append(sub['p_diagnostic_probe'].std() / np.sqrt(len(sub)))
        fb_means.append(sub['f_b'].mean())
        fb_errs.append(sub['f_b'].std() / np.sqrt(len(sub)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)

    # Panel A: Diagnostic Probe vs Forgetting (Decoupling representation from head misalignment)
    for i, cond in enumerate(conditions):
        ax1.errorbar(
            fb_means[i], probe_means[i],
            xerr=fb_errs[i], yerr=probe_errs[i],
            fmt='o', color=PALETTE[cond], label=cond_names[i],
            markersize=9, capsize=4, elinewidth=1.5
        )
        ax1.annotate(
            cond, (fb_means[i], probe_means[i]),
            textcoords="offset points", xytext=(8, -4),
            fontweight='bold', color=PALETTE[cond]
        )

    ax1.set_title("(A) Classifier Forgetting vs. Representation Retention", pad=10, fontweight='bold')
    ax1.set_xlabel("Apparent Forgetting on Frozen Head A: $F_B$ (%)")
    ax1.set_ylabel("Diagnostic Probe Acc on Frozen Backbone at $\\theta_2$ (%)")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', framealpha=0.9)

    # Panel B: CKA Drift vs Recovery Updates
    for i, cond in enumerate(conditions):
        ax2.errorbar(
            cka_means[i], urec_means[i],
            xerr=cka_errs[i], yerr=urec_errs[i],
            fmt='s', color=PALETTE[cond], label=cond_names[i],
            markersize=9, capsize=4, elinewidth=1.5
        )
        ax2.annotate(
            cond, (cka_means[i], urec_means[i]),
            textcoords="offset points", xytext=(8, -4),
            fontweight='bold', color=PALETTE[cond]
        )

    ax2.set_title("(B) Linear CKA Drift vs. Recovery Horizon ($u_{\\mathrm{rec}}$)", pad=10, fontweight='bold')
    ax2.set_xlabel("Representation Drift: $D_{\\mathrm{CKA}} = 1 - \\mathrm{CKA}(\\theta_1, \\theta_2)$")
    ax2.set_ylabel("Recovery Velocity: $u_{\\mathrm{rec}}$ (Updates)")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()
    out_file = os.path.join(FIGURES_DIR, "figure2_monotonic_characterization.png")
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out_file}")


if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING SOTLI GATE 1 PUBLICATION FIGURES")
    print("=" * 60)
    plot_relearning_trajectories()
    plot_monotonic_characterization()
    print("=" * 60)
    print("ALL GATE 1 FIGURES SUCCESSFULLY GENERATED! ✅")
    print("=" * 60)
