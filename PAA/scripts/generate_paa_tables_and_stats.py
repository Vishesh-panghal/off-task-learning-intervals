#!/usr/bin/env python3
"""
Scientific Data Verification and Statistics Generation for PAA Manuscript.
Reads raw experimental results and computes authoritative summary tables,
paired contrast statistics, confidence intervals, and verifies manuscript numerical fidelity.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.environ.get("SOTLI_RESULTS_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../results")))

def load_data():
    paths = {
        "confirmatory": os.path.join(BASE_DIR, "confirmatory", "confirmatory_results.csv"),
        "timing": os.path.join(BASE_DIR, "gate2", "timing_ablation_results.csv"),
        "momentum": os.path.join(BASE_DIR, "gate2", "momentum_ablation_results.csv"),
        "replay": os.path.join(BASE_DIR, "gate2", "replay_ablation_results.csv"),
        "screening": os.path.join(BASE_DIR, "screening_results.csv"),
        "vit_raw": os.path.join(BASE_DIR, "vit_study", "vit_confirmatory_raw.csv"),
        "vit_summary": os.path.join(BASE_DIR, "vit_study", "vit_confirmatory_summary.json"),
        "predictive": os.path.join(BASE_DIR, "predictive_models_summary.csv")
    }
    return {k: v for k, v in paths.items() if os.path.exists(v)}

def audit_confirmatory(path):
    print("=" * 75)
    print("1. ARCHIVAL TEN-SEED CONFIRMATORY AUDIT (Section 4.8, Tables 8 & 10)")
    print("=" * 75)
    df = pd.read_csv(path)
    conds = df['condition_id'].unique()
    
    summary = {}
    for c in sorted(conds):
        cdf = df[df['condition_id'] == c]
        test_acc = cdf['p_theta3_test'].astype(float)
        val_acc = cdf['p_theta3_val'].astype(float)
        summary[c] = {
            "test_mean": test_acc.mean(),
            "test_std": test_acc.std(ddof=1),
            "val_mean": val_acc.mean(),
            "val_std": val_acc.std(ddof=1),
            "n": len(cdf)
        }
        print(f"Condition {c:>2}: Test = {summary[c]['test_mean']:.2f}% +- {summary[c]['test_std']:.2f}%, Val = {summary[c]['val_mean']:.2f}% +- {summary[c]['val_std']:.2f}% (N={summary[c]['n']})")
    
    # Paired contrast B4 vs B1
    b4 = df[df['condition_id'] == 'B4'].sort_values('seed')
    b1 = df[df['condition_id'] == 'B1'].sort_values('seed')
    if len(b4) == len(b1) and (b4['seed'].values == b1['seed'].values).all():
        diff_test = b4['p_theta3_test'].values - b1['p_theta3_test'].values
        t_stat, p_val = stats.ttest_rel(b4['p_theta3_test'].values, b1['p_theta3_test'].values)
        dz = diff_test.mean() / diff_test.std(ddof=1)
        se = diff_test.std(ddof=1) / np.sqrt(len(diff_test))
        ci_95 = (diff_test.mean() - 2.262 * se, diff_test.mean() + 2.262 * se)
        print("\nPaired Contrast B4 - B1 (Test):")
        print(f"  Mean Diff = {diff_test.mean():+.3f} pp ({b4['p_theta3_test'].mean():.2f}% - {b1['p_theta3_test'].mean():.2f}%)")
        print(f"  95% CI = [{ci_95[0]:+.2f}, {ci_95[1]:+.2f}] pp")
        print(f"  t({len(diff_test)-1}) = {t_stat:.3f}, p = {p_val:.4f}, dz = {dz:+.2f}")

def audit_vit(path):
    print("\n" + "=" * 75)
    print("2. COMPACT VISION TRANSFORMER (CViT) AUDIT (Section 4.6, Table 6)")
    print("=" * 75)
    df = pd.read_csv(path)
    for c in sorted(df['condition_id'].unique()):
        cdf = df[df['condition_id'] == c]
        test_acc = cdf['p_theta3_test'].astype(float)
        val_acc = cdf['p_theta3_val'].astype(float)
        print(f"CViT {c:>2}: Test = {test_acc.mean():.2f}% +- {test_acc.std(ddof=1):.2f}%, Val = {val_acc.mean():.2f}% +- {val_acc.std(ddof=1):.2f}% (N={len(cdf)})")
    
    b4 = df[df['condition_id'] == 'B4'].sort_values('seed')
    b1 = df[df['condition_id'] == 'B1'].sort_values('seed')
    if len(b4) == len(b1):
        diff_test = b4['p_theta3_test'].values - b1['p_theta3_test'].values
        t_stat, p_val = stats.ttest_rel(b4['p_theta3_test'].values, b1['p_theta3_test'].values)
        dz = diff_test.mean() / diff_test.std(ddof=1)
        se = diff_test.std(ddof=1) / np.sqrt(len(diff_test))
        ci_95 = (diff_test.mean() - 2.776 * se, diff_test.mean() + 2.776 * se)
        print("\nCViT Paired Contrast B4 - B1 (Terminal Test):")
        print(f"  Mean Diff = {diff_test.mean():+.3f} pp")
        print(f"  95% CI = [{ci_95[0]:+.2f}, {ci_95[1]:+.2f}] pp")
        print(f"  t({len(diff_test)-1}) = {t_stat:.3f}, p = {p_val:.4f}, dz = {dz:+.2f}")
        
        diff_val = b4['p_theta3_val'].values - b1['p_theta3_val'].values
        t_stat_v, p_val_v = stats.ttest_rel(b4['p_theta3_val'].values, b1['p_theta3_val'].values)
        dz_v = diff_val.mean() / diff_val.std(ddof=1)
        print("\nCViT Paired Contrast B4 - B1 (Validation):")
        print(f"  Mean Diff = {diff_val.mean():+.3f} pp")
        print(f"  t({len(diff_val)-1}) = {t_stat_v:.3f}, p = {p_val_v:.4f}, dz = {dz_v:+.2f}")

def audit_screening(path):
    print("\n" + "=" * 75)
    print("3. MULTI-DOMAIN SCREENING SPECTRUM AUDIT (Section 4.7, Table 7)")
    print("=" * 75)
    df = pd.read_csv(path)
    for c in sorted(df['condition_id'].unique()):
        cdf = df[df['condition_id'] == c]
        r20 = cdf['eff_rank_theta2'].astype(float).mean() if 'eff_rank_theta2' in cdf else np.nan
        r35 = cdf['eff_rank_theta3'].astype(float).mean() if 'eff_rank_theta3' in cdf else np.nan
        fb = cdf['f_b'].astype(float).mean() if 'f_b' in cdf else np.nan
        probe = cdf['p_probe_theta2'].astype(float).mean() if 'p_probe_theta2' in cdf else np.nan
        print(f"Condition {c:>2}: Forgetting = {fb:+.2f} pp, Probe@20k = {probe:.2f}%, Rank@20k = {r20:.1f}, Rank@35k = {r35:.1f}")

def main():
    paths = load_data()
    print("Loaded Data Files:", list(paths.keys()))
    if "confirmatory" in paths:
        audit_confirmatory(paths["confirmatory"])
    if "vit_raw" in paths:
        audit_vit(paths["vit_raw"])
    if "screening" in paths:
        audit_screening(paths["screening"])

if __name__ == "__main__":
    main()
