"""
Audit & Reconciliation Script for SOTLI Stage 1 (Gate 1).
Reads results/screening_results.csv, computes paired differences against Control 1 (B0 total)
and Control 2 (B0 task-A matched), computes 95% bootstrap confidence intervals (B=2,000),
and formally evaluates the Gate 1 Decision Criteria for TMLR.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np


def bootstrap_ci(diffs, num_bootstrap=2000, alpha=0.05, seed=42):
    """Computes paired bootstrap confidence interval for the mean"""
    rng = np.random.RandomState(seed)
    n = len(diffs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    boot_means = []
    for _ in range(num_bootstrap):
        resample = rng.choice(diffs, size=n, replace=True)
        boot_means.append(np.mean(resample))
    
    mean_val = float(np.mean(diffs))
    std_val = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    ci_lower = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    return mean_val, std_val, ci_lower, ci_upper


def audit_screening_results(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: Results file not found at {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print("=" * 80)
    print(" SOTLI GATE 1: DISCOVERY SCREENING DATA AUDIT & STATISTICAL RECONCILIATION")
    print(f" Source CSV: {csv_path}")
    print(f" Total Logged Runs: {len(df)}")
    print("=" * 80)

    # Check conditions and seeds present
    conditions = df['condition_id'].unique().tolist()
    seeds = df['seed'].unique().tolist()
    print(f"Conditions present: {conditions}")
    print(f"Seeds present ({len(seeds)}): {seeds}")

    if 'B0' not in conditions:
        print("Error: Baseline B0 (Continuous) missing from results. Cannot compute paired deltas.")
        sys.exit(1)

    # Pivot table to pair conditions by seed
    b0_rows = df[df['condition_id'] == 'B0'].set_index('seed')

    metrics_summary = []
    
    print("\n" + "-" * 80)
    print(f"{'Condition':<12} | {'Δ_total (%)':<16} | {'Δ_A (%)':<16} | {'F_B (Drop %)':<12} | {'u_rec (Updates)':<15}")
    print("-" * 80)

    gate1_signals = []
    head_switch_perf = None

    for cond in conditions:
        cond_df = df[df['condition_id'] == cond].set_index('seed')
        common_seeds = sorted(list(set(b0_rows.index) & set(cond_df.index)))

        if len(common_seeds) == 0:
            continue

        delta_total_list = []
        delta_a_list = []
        fb_list = []
        ra_list = []
        urec_list = []
        aulc_list = []
        cka_list = []
        eff_rank_change_list = []

        for s in common_seeds:
            b0_total_acc = b0_rows.loc[s, 'p_theta3_val']
            b0_a_acc = b0_rows.loc[s, 'p_control2_val']
            # If intermediate control2 wasn't tracked or is N/A, fall back to b0_total_acc
            try:
                b0_a_acc = float(b0_a_acc)
            except (ValueError, TypeError):
                b0_a_acc = b0_total_acc

            cond_acc = cond_df.loc[s, 'p_theta3_val']
            cond_fb = cond_df.loc[s, 'f_b']
            cond_ra = cond_df.loc[s, 'r_a']
            cond_urec = cond_df.loc[s, 'u_rec']
            cond_aulc = cond_df.loc[s, 'aulc']
            cond_cka = cond_df.loc[s, 'cka_drift']
            cond_er1 = cond_df.loc[s, 'eff_rank_theta1']
            cond_er2 = cond_df.loc[s, 'eff_rank_theta2']

            delta_total_list.append(cond_acc - b0_total_acc)
            delta_a_list.append(cond_acc - b0_a_acc)
            fb_list.append(cond_fb)
            ra_list.append(cond_ra)
            if cond_urec >= 0:
                urec_list.append(cond_urec)
            aulc_list.append(cond_aulc)
            cka_list.append(cond_cka)
            eff_rank_change_list.append(cond_er2 - cond_er1)

        d_tot_m, d_tot_s, d_tot_l, d_tot_u = bootstrap_ci(delta_total_list)
        d_a_m, d_a_s, d_a_l, d_a_u = bootstrap_ci(delta_a_list)
        fb_m = float(np.mean(fb_list))
        urec_m = float(np.mean(urec_list)) if len(urec_list) > 0 else -1.0

        if cond == 'B1':
            head_switch_perf = d_tot_m

        if cond != 'B0':
            gate1_signals.append({
                'cond': cond,
                'delta_total_mean': d_tot_m,
                'ci_lower': d_tot_l,
                'ci_upper': d_tot_u,
                'fb_mean': fb_m,
                'urec_mean': urec_m,
                'aulc_mean': float(np.mean(aulc_list)),
                'cka_mean': float(np.mean(cka_list)),
            })

        print(f"{cond:<12} | {d_tot_m:+.2f} [{d_tot_l:+.2f}, {d_tot_u:+.2f}] | {d_a_m:+.2f} [{d_a_l:+.2f}, {d_a_u:+.2f}] | {fb_m:6.2f}%      | {urec_m:8.1f}")

    print("-" * 80)
    print("\n" + "=" * 80)
    print(" FORMAL GATE 1 DECISION EVALUATION (TMLR PROTOCOL)")
    print("=" * 80)

    # Criterion 1: Signal Confirmed: Mean(Δ_total) >= 1.0% with 95% CI not spanning zero
    confirmed_conds = [s for s in gate1_signals if s['delta_total_mean'] >= 1.0 and s['ci_lower'] > 0.0]

    # Criterion 2: Parity with Altered Dynamics (Δ_total ~ 0 but recovery accelerated)
    parity_conds = [s for s in gate1_signals if abs(s['delta_total_mean']) < 1.0 and s['urec_mean'] > 0 and s['urec_mean'] < 2000]

    # Criterion 3: Head-Switch Equivalence: structured tasks don't beat B1
    beating_b1 = [s for s in gate1_signals if head_switch_perf is not None and s['cond'] not in ['B1', 'B7'] and s['delta_total_mean'] > head_switch_perf + 0.5]

    if confirmed_conds:
        print("\n✅ DECISION: [SIGNAL CONFIRMED - PROCEED TO GATE 2]")
        for c in confirmed_conds:
            print(f"   ★ {c['cond']}: Δ_total = {c['delta_total_mean']:+.2f}% (95% CI: [{c['ci_lower']:+.2f}, {c['ci_upper']:+.2f}]), F_B = {c['fb_mean']:.2f}%")
        print("\nRecommended Next Step: Launch Gate 2 (Confound Elimination: schedule timing ablation and optimizer momentum).")

    elif beating_b1:
        print("\n⏳ DECISION: [STRUCTURED SIGNAL PRESENT OVER HEAD-SWITCH]")
        for c in beating_b1:
            print(f"   ★ {c['cond']} outperforms Head-Switch Control B1 by {c['delta_total_mean'] - head_switch_perf:+.2f}%")
        print("\nRecommended Next Step: Expand seed count to N=10 to tighten confidence intervals for Gate 2.")

    elif parity_conds:
        print("\n📊 DECISION: [PARITY WITH ALTERED DYNAMICS - PROCEED TO DYNAMICS CHARACTERIZATION]")
        for c in parity_conds:
            print(f"   ★ {c['cond']}: Δ_total = {c['delta_total_mean']:+.2f}%, u_rec = {c['urec_mean']:.1f} updates")
        print("\nRecommended Next Step: Focus manuscript framing on rapid representation recovery dynamics rather than terminal accuracy boosts.")

    else:
        print("\n⚠️ DECISION: [NULL SIGNAL / HEAD-SWITCH EQUIVALENCE / INCONCLUSIVE]")
        print("   Structured auxiliary tasks did not demonstrate a statistically significant gain over baselines.")
        print("   Check individual conditions and seed variance before scaling compute.")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit SOTLI Screening Results")
    parser.add_argument("--csv", type=str, default="./results/screening_results.csv",
                        help="Path to results CSV file")
    args = parser.parse_args()
    audit_screening_results(args.csv)
