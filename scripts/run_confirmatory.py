"""
SOTLI Preregistered Confirmatory Protocol (P1-B).
Executes a well-powered confirmatory replication on fresh seeds:
- 10 fresh seeds: [1001, 1002, ..., 1010]
- Core conditions: B0 (Continuous Baseline), B4 (STL-10 auxiliary), B7 (Gaussian Noise control)
- Dual-budget evaluation:
    Delta_total = Acc(B4) - Acc(B0, Control 1: Equal Total Updates = 35k)
    Delta_A     = Acc(B4) - Acc(B0, Control 2: Equal Target Updates = 30k)
- Statistical analysis: paired t-test, Wilcoxon signed-rank, 95% paired bootstrap CI
- Fully reproducible: fixed worker seeds, isolated eval partitions, B1 leakage fix applied.
"""

import os
import sys
import argparse
import random
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import stats

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.resnet_gn import ResNet18GroupNorm
from src.data.datasets import (
    get_cifar100_loaders, get_auxiliary_loader, get_fixed_probe_set
)
from src.utils.metrics import (
    evaluate_accuracy, extract_probe_features, compute_effective_rank,
    compute_linear_cka
)
from src.training.evaluator import RelearningDynamicsTracker, run_diagnostic_probe_at_theta2
from src.training.harness import (
    infinite_loader, stash_head_optimizer_state, restore_head_optimizer_state
)
from scripts.run_screening import (
    train_phase1_if_needed, run_condition_b0, run_condition_sotli, save_result_to_csv
)

CONFIRMATORY_SEEDS = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def bootstrap_ci(diffs, num_bootstrap=5000, alpha=0.05, seed=42):
    """Computes paired bootstrap confidence interval (percentile) for the mean."""
    rng = np.random.RandomState(seed)
    n = len(diffs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    boot_means = [float(np.mean(rng.choice(diffs, size=n, replace=True))) for _ in range(num_bootstrap)]
    mean_val = float(np.mean(diffs))
    std_val = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    ci_lower = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    return mean_val, std_val, ci_lower, ci_upper


def run_confirmatory_analysis(results_csv, output_dir):
    """
    Computes preregistered confirmatory statistics from results_csv.
    Generates summary report and LaTeX table.
    """
    import pandas as pd
    if not os.path.exists(results_csv):
        print(f"No results file found at {results_csv}")
        return

    df = pd.read_csv(results_csv)
    print("\n" + "=" * 80)
    print(" SOTLI PREREGISTERED CONFIRMATORY HYPOTHESIS TESTING REPORT")
    print(f" Source: {results_csv}")
    print(f" Seeds Evaluated ({len(df['seed'].unique())}): {sorted(df['seed'].unique().tolist())}")
    print("=" * 80)

    if 'B0' not in df['condition_id'].values:
        print("Error: B0 baseline not present in results.")
        return

    b0_df = df[df['condition_id'] == 'B0'].set_index('seed')
    summary_rows = []

    for cond in [c for c in df['condition_id'].unique() if c != 'B0']:
        cond_df = df[df['condition_id'] == cond].set_index('seed')
        common_seeds = b0_df.index.intersection(cond_df.index)
        n = len(common_seeds)
        if n < 2:
            continue

        b0_paired = b0_df.loc[common_seeds]
        cond_paired = cond_df.loc[common_seeds]

        # Delta Total (Equal Total Updates = 35k)
        diff_total_val = (cond_paired['p_theta3_val'] - b0_paired['p_theta3_val']).values
        diff_total_test = (cond_paired['p_theta3_test'] - b0_paired['p_theta3_test']).values

        # Delta A (Equal Task A Exposure = 30k)
        diff_a_val = (cond_paired['p_theta3_val'] - b0_paired['p_control2_val']).values

        m_tot_val, s_tot_val, lo_tot_val, hi_tot_val = bootstrap_ci(diff_total_val)
        m_tot_test, s_tot_test, lo_tot_test, hi_tot_test = bootstrap_ci(diff_total_test)
        m_a_val, s_a_val, lo_a_val, hi_a_val = bootstrap_ci(diff_a_val)

        # Statistical tests
        t_stat_tot, p_val_tot = stats.ttest_rel(cond_paired['p_theta3_val'], b0_paired['p_theta3_val'])
        w_stat_tot, p_wilc_tot = stats.wilcoxon(diff_total_val) if np.any(diff_total_val != 0) else (0, 1.0)
        cohen_d_tot = m_tot_val / s_tot_val if s_tot_val > 0 else 0.0

        print(f"\nCondition: {cond} (N={n} seeds)")
        print(f"  Δ_total (Val)  : {m_tot_val:+.2f} pp [95% CI: {lo_tot_val:+.2f}, {hi_tot_val:+.2f}] (p_ttest={p_val_tot:.4f}, p_wilcoxon={p_wilc_tot:.4f}, d={cohen_d_tot:.2f})")
        print(f"  Δ_total (Test) : {m_tot_test:+.2f} pp [95% CI: {lo_tot_test:+.2f}, {hi_tot_test:+.2f}]")
        print(f"  Δ_A     (Val)  : {m_a_val:+.2f} pp [95% CI: {lo_a_val:+.2f}, {hi_a_val:+.2f}]")

        summary_rows.append({
            'condition': cond,
            'N': n,
            'delta_total_val_mean': m_tot_val,
            'delta_total_val_ci_low': lo_tot_val,
            'delta_total_val_ci_high': hi_tot_val,
            'p_value_ttest': p_val_tot,
            'p_value_wilcoxon': p_wilc_tot,
            'cohen_d': cohen_d_tot,
            'delta_total_test_mean': m_tot_test,
            'delta_total_test_ci_low': lo_tot_test,
            'delta_total_test_ci_high': hi_tot_test,
            'delta_a_val_mean': m_a_val,
            'delta_a_val_ci_low': lo_a_val,
            'delta_a_val_ci_high': hi_a_val,
        })

    # Direct Paired Test: STL-10 (B4) vs Clean Head-Switch & Momentum Reset (B1)
    if 'B4' in df['condition_id'].values and 'B1' in df['condition_id'].values:
        b4_df = df[df['condition_id'] == 'B4'].set_index('seed')
        b1_df = df[df['condition_id'] == 'B1'].set_index('seed')
        common_b4_b1 = b4_df.index.intersection(b1_df.index)
        if len(common_b4_b1) >= 2:
            diff_b4_b1_val = (b4_df.loc[common_b4_b1, 'p_theta3_val'] - b1_df.loc[common_b4_b1, 'p_theta3_val']).values
            diff_b4_b1_test = (b4_df.loc[common_b4_b1, 'p_theta3_test'] - b1_df.loc[common_b4_b1, 'p_theta3_test']).values
            m_v, s_v, lo_v, hi_v = bootstrap_ci(diff_b4_b1_val)
            m_t, s_t, lo_t, hi_t = bootstrap_ci(diff_b4_b1_test)
            t_stat, p_val = stats.ttest_rel(b4_df.loc[common_b4_b1, 'p_theta3_val'], b1_df.loc[common_b4_b1, 'p_theta3_val'])
            w_stat, p_wilc = stats.wilcoxon(diff_b4_b1_val) if np.any(diff_b4_b1_val != 0) else (0, 1.0)
            d_val = m_v / s_v if s_v > 0 else 0.0

            print("\n" + "=" * 80)
            print(f" DECISIVE PAIRED TEST: STL-10 (B4) vs CLEAN HEAD-SWITCH & RESET (B1) [N={len(common_b4_b1)} seeds]")
            print(f"  Δ(B4 - B1) Val : {m_v:+.2f} pp [95% CI: {lo_v:+.2f}, {hi_v:+.2f}] (p_ttest={p_val:.4f}, p_wilcoxon={p_wilc:.4f}, d={d_val:.2f})")
            print(f"  Δ(B4 - B1) Test: {m_t:+.2f} pp [95% CI: {lo_t:+.2f}, {hi_t:+.2f}]")
            print("=" * 80 + "\n")

    # Export summary CSV and LaTeX table
    os.makedirs(output_dir, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, "confirmatory_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n✓ Saved confirmatory summary to {summary_csv}")

    # Generate LaTeX table
    tex_path = os.path.join(output_dir, "confirmatory_table.tex")
    with open(tex_path, "w") as f:
        f.write("% Auto-generated SOTLI Confirmatory Results Table\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\hline\n")
        f.write("Condition & $N$ & $\\Delta_{\\text{total}}$ (Val) & 95\\% CI & $p_{\\text{paired}}$ & Cohen's $d$ & $\\Delta_{\\text{total}}$ (Test) \\\\\n")
        f.write("\\hline\n")
        for r in summary_rows:
            f.write(f"{r['condition']} & {r['N']} & {r['delta_total_val_mean']:+.2f} & [{r['delta_total_val_ci_low']:+.2f}, {r['delta_total_val_ci_high']:+.2f}] & {r['p_value_ttest']:.4f} & {r['cohen_d']:.2f} & {r['delta_total_test_mean']:+.2f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
    print(f"✓ Saved LaTeX table to {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="SOTLI Preregistered Confirmatory Protocol")
    parser.add_argument("--conditions", nargs="+", default=["B0", "B4", "B7"],
                        help="Preregistered conditions to evaluate (default: B0 B4 B7)")
    parser.add_argument("--seeds", nargs="+", type=int, default=CONFIRMATORY_SEEDS,
                        help="Preregistered random seeds (default: 10 fresh seeds 1001-1010)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="./data_cache")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/confirmatory")
    parser.add_argument("--results_dir", type=str, default="./results/confirmatory")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run high-speed smoke test to verify confirmatory pipeline")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_default_device()
    print("=================================================================")
    print(" SOTLI PREREGISTERED CONFIRMATORY PROTOCOL (P1-B)")
    print(f" Target Venue: TMLR")
    print(f" Execution Device: {device}")
    print(f" Conditions: {args.conditions}")
    print(f" Fresh Seeds ({len(args.seeds)}): {args.seeds}")
    print(f" Smoke Test Mode: {args.smoke_test}")
    print("=================================================================")

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.smoke_test:
        u1, u_b, u2 = 40, 20, 40
        results_csv = os.path.join(args.results_dir, "confirmatory_results_smoke.csv")
    else:
        u1, u_b, u2 = 15000, 5000, 15000
        results_csv = os.path.join(args.results_dir, "confirmatory_results.csv")

    lr, momentum, weight_decay = 0.1, 0.9, 5e-4
    total_runs = len(args.conditions) * len(args.seeds)
    current_run = 0

    def get_completed_runs():
        completed = set()
        if os.path.exists(results_csv):
            with open(results_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'condition_id' in row and 'seed' in row:
                        try:
                            completed.add((row['condition_id'], int(row['seed'])))
                        except ValueError:
                            pass
        return completed

    for seed in args.seeds:
        completed_runs = get_completed_runs()
        # Check if all conditions for this seed are already done before loading/training Phase 1
        seed_conditions_pending = [c for c in args.conditions if (c, seed) not in completed_runs]
        if not seed_conditions_pending:
            print(f"\n[Seed {seed}] All conditions ({args.conditions}) already completed in {results_csv}. Skipping.")
            continue

        # Phase 1: Shared checkpoint theta_1 for this seed
        theta1_path, opt1_path, p_before, eff_rank_theta1, p_probe_theta1 = train_phase1_if_needed(
            seed=seed, data_dir=args.data_dir, checkpoint_dir=args.checkpoint_dir,
            u1=u1, lr=lr, momentum=momentum, weight_decay=weight_decay,
            device=device, smoke_test=args.smoke_test
        )

        for cond in args.conditions:
            current_run += 1
            completed_runs = get_completed_runs()
            if (cond, seed) in completed_runs:
                print(f"\n[Seed {seed} | {cond}] Already completed in {results_csv}. Skipping.")
                continue

            print(f"\n>>> Confirmatory [{current_run}/{total_runs}] Condition: {cond} | Seed: {seed} <<<")

            if cond == 'B0':
                result = run_condition_b0(
                    seed=seed, theta1_path=theta1_path, opt1_path=opt1_path,
                    p_before=p_before, eff_rank_theta1=eff_rank_theta1,
                    data_dir=args.data_dir, checkpoint_dir=args.checkpoint_dir,
                    u_b=u_b, u2=u2, lr=lr, momentum=momentum, weight_decay=weight_decay,
                    device=device, smoke_test=args.smoke_test,
                    p_probe_theta1=p_probe_theta1
                )
            else:
                result = run_condition_sotli(
                    condition_id=cond, seed=seed, theta1_path=theta1_path, opt1_path=opt1_path,
                    p_before=p_before, eff_rank_theta1=eff_rank_theta1,
                    data_dir=args.data_dir, checkpoint_dir=args.checkpoint_dir,
                    results_dir=args.results_dir,
                    u_b=u_b, u2=u2, lr=lr, momentum=momentum, weight_decay=weight_decay,
                    device=device, smoke_test=args.smoke_test,
                    p_probe_theta1=p_probe_theta1
                )

            save_result_to_csv(result, results_csv)
            completed_runs.add((cond, seed))
            print(f"✓ Saved confirmatory result for {cond} (Seed {seed}) -> {results_csv}")

    print("\n=================================================================")
    print(" ALL CONFIRMATORY RUNS COMPLETED! COMPUTING HYPOTHESIS TESTS...")
    print("=================================================================\n")

    run_confirmatory_analysis(results_csv, args.results_dir)


if __name__ == "__main__":
    main()
