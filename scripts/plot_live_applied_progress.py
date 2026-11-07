"""
Live Training Progress & Trajectory Dashboard for SOTLI Applied Fine-Tuning.

Generates and continuously updates a publication-quality 3-panel figure:
- Panel 1: Training & Validation Loss across Phase 1, Phase 2 (shaded), and Phase 3
- Panel 2: Validation Top-1 Accuracy (%) & Balanced Accuracy (%)
- Panel 3: Validation Macro AUC & Diagnostic Representation Effective Rank
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_applied_trajectory(
    trajectory_csv,
    output_png=None,
    condition="C0",
    seed=42,
    u1=5000,
    uB=2000,
    u2=5000
):
    if not os.path.exists(trajectory_csv) or os.path.getsize(trajectory_csv) == 0:
        return None

    try:
        df = pd.read_csv(trajectory_csv)
    except Exception as e:
        print(f"[Plotter] Could not read {trajectory_csv}: {e}")
        return None

    if len(df) == 0 or "step" not in df.columns:
        return None

    # Ensure numeric conversion
    for col in ["step", "train_loss", "val_loss", "val_top1", "val_bal_acc", "val_macro_auc", "eff_rank"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if output_png is None:
        output_png = os.path.splitext(trajectory_csv)[0] + "_dashboard.png"

    total_updates = u1 + uB + u2
    b1 = u1
    b2 = u1 + uB

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, dpi=180)
    fig.suptitle(f"SOTLI Applied Fine-Tuning — Live Trajectory [{condition} | Seed {seed}]",
                 fontsize=14, fontweight='bold', y=0.98)

    # Styling helper for phase shading
    for ax in axes:
        ax.axvspan(b1, b2, color='#fdebd0', alpha=0.55, label='Phase 2 (Interval)' if ax == axes[0] else "")
        ax.axvline(b1, color='#b03a2e', linestyle='--', linewidth=1.2, alpha=0.8)
        ax.axvline(b2, color='#2874a6', linestyle='--', linewidth=1.2, alpha=0.8)
        ax.grid(True, linestyle=':', alpha=0.55)
        ax.set_xlim(0, max(total_updates, df["step"].max() + 50))

    # ----------------------------------------------------
    # Panel 1: Loss
    # ----------------------------------------------------
    ax1 = axes[0]
    ax1.plot(df["step"], df["train_loss"], label="Train Loss (Step)", color="#7f8c8d", alpha=0.6, linewidth=1.0)
    # Moving average
    if len(df) >= 5:
        ma = df["train_loss"].rolling(window=min(10, len(df)), min_periods=1).mean()
        ax1.plot(df["step"], ma, label="Train Loss (Smoothed)", color="#c0392b", linewidth=2.0)
    if "val_loss" in df.columns and df["val_loss"].notnull().any():
        val_df = df.dropna(subset=["val_loss"])
        ax1.plot(val_df["step"], val_df["val_loss"], 'o-', label="Val Loss", color="#1f618d", linewidth=1.8, markersize=4)

    ax1.set_ylabel("Cross-Entropy Loss", fontweight='semibold')
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.text(b1 / 2, ax1.get_ylim()[1] * 0.9 if ax1.get_ylim()[1] > 0 else 1.0,
             "Phase 1 (Target A)", ha='center', fontsize=9, fontweight='bold', color='#555')
    ax1.text((b1 + b2) / 2, ax1.get_ylim()[1] * 0.9 if ax1.get_ylim()[1] > 0 else 1.0,
             f"Phase 2 ({condition})", ha='center', fontsize=9, fontweight='bold', color='#935116')
    ax1.text((b2 + total_updates) / 2, ax1.get_ylim()[1] * 0.9 if ax1.get_ylim()[1] > 0 else 1.0,
             "Phase 3 (Relearning)", ha='center', fontsize=9, fontweight='bold', color='#555')

    # ----------------------------------------------------
    # Panel 2: Accuracies (Top-1 & Balanced)
    # ----------------------------------------------------
    ax2 = axes[1]
    if "val_top1" in df.columns and df["val_top1"].notnull().any():
        vdf = df.dropna(subset=["val_top1"])
        ax2.plot(vdf["step"], vdf["val_top1"], 's-', color="#2980b9", linewidth=2.0, markersize=5, label="Top-1 Accuracy (%)")
    if "val_bal_acc" in df.columns and df["val_bal_acc"].notnull().any():
        bdf = df.dropna(subset=["val_bal_acc"])
        ax2.plot(bdf["step"], bdf["val_bal_acc"], '^-', color="#d35400", linewidth=2.0, markersize=5, label="Balanced Accuracy (%)")

    ax2.set_ylabel("Accuracy (%)", fontweight='semibold')
    ax2.legend(loc="lower right", framealpha=0.9)

    # ----------------------------------------------------
    # Panel 3: Macro AUC & Diagnostics
    # ----------------------------------------------------
    ax3 = axes[2]
    color_auc = '#27ae60'
    color_rank = '#8e44ad'

    ax3.set_xlabel("Optimization Updates (Budget)", fontweight='semibold')
    ax3.set_ylabel("Macro AUC (OVR)", color=color_auc, fontweight='semibold')
    if "val_macro_auc" in df.columns and df["val_macro_auc"].notnull().any():
        adf = df.dropna(subset=["val_macro_auc"])
        line1 = ax3.plot(adf["step"], adf["val_macro_auc"], 'd-', color=color_auc, linewidth=2.0, markersize=6, label="Macro AUC")
        ax3.tick_params(axis='y', labelcolor=color_auc)
        ax3.set_ylim(min(0.5, adf["val_macro_auc"].min() - 0.05), 1.0)

    # Twin axis for Effective Rank
    if "eff_rank" in df.columns and df["eff_rank"].notnull().any():
        rdf = df.dropna(subset=["eff_rank"])
        if len(rdf) > 0:
            ax3_twin = ax3.twinx()
            ax3_twin.set_ylabel("Effective Rank (2k Probe)", color=color_rank, fontweight='semibold')
            line2 = ax3_twin.plot(rdf["step"], rdf["eff_rank"], 'P--', color=color_rank, linewidth=2.0, markersize=7, label="EffRank")
            ax3_twin.tick_params(axis='y', labelcolor=color_rank)
            ax3_twin.grid(False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, bbox_inches='tight')
    plt.close(fig)

    return output_png


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot live applied training trajectory.")
    parser.add_argument("--csv", type=str, required=True, help="Path to training_trajectory.csv")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path")
    parser.add_argument("--condition", type=str, default="C0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--u1", type=int, default=5000)
    parser.add_argument("--uB", type=int, default=2000)
    parser.add_argument("--u2", type=int, default=5000)
    args = parser.parse_args()

    out = plot_applied_trajectory(args.csv, args.out, args.condition, args.seed, args.u1, args.uB, args.u2)
    if out:
        print(f"[Plotter] Saved live dashboard plot to {out}")
