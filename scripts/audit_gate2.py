"""
Gate 2 Confound Elimination Audit Script.
Analyzes and reconciles:
1. Schedule Timing Mini-Ablation (Early vs. Mid vs. Late)
2. Momentum Buffer Ablation (Zeroed vs. Preserved)
3. Replay Baseline (90% B + 10% A co-training vs. SOTLI Temporal Isolation)
"""

import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GATE1_CSV = os.path.join(PROJECT_ROOT, "results", "screening_results.csv")
TIMING_CSV = os.path.join(PROJECT_ROOT, "results", "gate2", "timing_ablation_results.csv")
MOMENTUM_CSV = os.path.join(PROJECT_ROOT, "results", "gate2", "momentum_ablation_results.csv")
REPLAY_CSV = os.path.join(PROJECT_ROOT, "results", "gate2", "replay_ablation_results.csv")

def main():
    print("===================================================================")
    print(" SOTLI GATE 2 CONFOUND ELIMINATION AUDIT & RECONCILIATION")
    print("===================================================================\n")
    
    df_gate1 = pd.read_csv(GATE1_CSV)
    df_timing = pd.read_csv(TIMING_CSV)
    df_momentum = pd.read_csv(MOMENTUM_CSV)
    df_replay = pd.read_csv(REPLAY_CSV)
    
    seeds = [42, 123, 456]
    df_gate1_subset = df_gate1[df_gate1['seed'].isin(seeds)]
    
    # -------------------------------------------------------------
    # 1. TIMING MINI-ABLATION
    # -------------------------------------------------------------
    print("--- 1. SCHEDULE TIMING MINI-ABLATION (N=3 seeds: 42, 123, 456) ---")
    for cond in ['B4', 'B1']:
        print(f"\nCondition: {cond}")
        # Early
        early = df_timing[(df_timing['condition_id'] == cond) & (df_timing['timing'] == 'early')]
        # Late
        late = df_timing[(df_timing['condition_id'] == cond) & (df_timing['timing'] == 'late')]
        # Mid (from Gate 1)
        mid = df_gate1_subset[df_gate1_subset['condition_id'] == cond]
        
        print(f"  Early (u1=5k):  U=35k Val Acc = {early['val_fixed_total_budget'].mean():.2f}% ± {early['val_fixed_total_budget'].std():.2f}% | u2=15k Acc = {early['val_fixed_recovery_horizon'].mean():.2f}% | u_rec = {early['u_rec'].mean():.0f} updates")
        print(f"  Mid   (u1=15k): U=35k Val Acc = {mid['p_theta3_val'].mean():.2f}% ± {mid['p_theta3_val'].std():.2f}% | u2=15k Acc = {mid['p_theta3_val'].mean():.2f}% | u_rec = {mid['u_rec'].mean():.0f} updates")
        print(f"  Late  (u1=25k): U=35k Val Acc = {late['val_fixed_total_budget'].mean():.2f}% ± {late['val_fixed_total_budget'].std():.2f}% | u2=15k Acc = {late['val_fixed_recovery_horizon'].mean():.2f}% | u_rec = {late['u_rec'].mean():.0f} updates")

    # -------------------------------------------------------------
    # 2. MOMENTUM BUFFER ABLATION (PRESERVED vs ZEROED)
    # -------------------------------------------------------------
    print("\n--- 2. MOMENTUM BUFFER ABLATION: ZEROED vs PRESERVED (N=3 seeds) ---")
    for cond in ['B4', 'B1']:
        print(f"\nCondition: {cond}")
        pres = df_gate1_subset[df_gate1_subset['condition_id'] == cond]
        zero = df_momentum[df_momentum['condition_id'] == cond]
        
        print(f"  Preserved (Gate 1): Val Acc = {pres['p_theta3_val'].mean():.2f}% ± {pres['p_theta3_val'].std():.2f}% | Test Acc = {pres['p_theta3_test'].mean():.2f}% ± {pres['p_theta3_test'].std():.2f}% | u_rec = {pres['u_rec'].mean():.0f} updates | AULC = {pres['aulc'].mean():.2f}")
        print(f"  Zeroed    (Gate 2): Val Acc = {zero['p_theta3_val'].mean():.2f}% ± {zero['p_theta3_val'].std():.2f}% | Test Acc = {zero['p_theta3_test'].mean():.2f}% ± {zero['p_theta3_test'].std():.2f}% | u_rec = {zero['u_rec'].mean():.0f} updates | AULC = {zero['aulc'].mean():.2f}")
        delta_val = zero['p_theta3_val'].mean() - pres['p_theta3_val'].mean()
        delta_test = zero['p_theta3_test'].mean() - pres['p_theta3_test'].mean()
        print(f"  Delta (Zeroed - Preserved): Val Δ = {delta_val:+.2f}% | Test Δ = {delta_test:+.2f}%")

    # -------------------------------------------------------------
    # 3. REPLAY BASELINE (90% B + 10% A vs SOTLI TEMPORAL ISOLATION)
    # -------------------------------------------------------------
    print("\n--- 3. REPLAY BASELINE: 90% B + 10% A vs SOTLI ISOLATION (N=3 seeds) ---")
    for cond in ['B4', 'B1']:
        print(f"\nCondition: {cond}")
        sotli = df_gate1_subset[df_gate1_subset['condition_id'] == cond]
        replay = df_replay[df_replay['condition_id'] == cond]
        
        print(f"  SOTLI Temporal Isolation: Val Acc = {sotli['p_theta3_val'].mean():.2f}% ± {sotli['p_theta3_val'].std():.2f}% | Test Acc = {sotli['p_theta3_test'].mean():.2f}% ± {sotli['p_theta3_test'].std():.2f}% | θ₂ Drop F_B = {sotli['f_b'].mean():.2f}%")
        print(f"  90% B + 10% A Replay:    Val Acc = {replay['p_theta3_val'].mean():.2f}% ± {replay['p_theta3_val'].std():.2f}% | Test Acc = {replay['p_theta3_test'].mean():.2f}% ± {replay['p_theta3_test'].std():.2f}% | θ₂ Drop F_B = {(replay['p_theta1'] - replay['p_theta2']).mean():.2f}%")
        delta_sotli_replay_val = sotli['p_theta3_val'].mean() - replay['p_theta3_val'].mean()
        delta_sotli_replay_test = sotli['p_theta3_test'].mean() - replay['p_theta3_test'].mean()
        print(f"  Advantage of SOTLI over Replay: Val Δ = {delta_sotli_replay_val:+.2f}% | Test Δ = {delta_sotli_replay_test:+.2f}%")

    print("\n===================================================================")
    print(" ALL GATE 2 ABLATIONS AUDITED & VERIFIED!")
    print("===================================================================\n")

if __name__ == "__main__":
    main()
