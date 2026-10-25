#!/usr/bin/env python3
"""
PAA Experimental Verification Auditor for SOTLI.
Computes means, standard deviations, paired differences, Student's t-tests,
confidence intervals, exact sign-flip permutation tests, and 2x2 factorial effects.
"""

import argparse
import json
import itertools
import numpy as np
import pandas as pd
from scipy import stats

def compute_primary_stats(primary_csv):
    df = pd.read_csv(primary_csv)
    n = len(df)
    c0_30k = df['c0_30k'].values
    c0_35k = df['c0_35k'].values
    c3_35k = df['c3_35k'].values
    s1_35k = df['s1_35k'].values

    # Primary contrast: S1(35k) - C0(35k)
    diff_s1_c0 = s1_35k - c0_35k
    mean_diff_s1_c0 = float(np.mean(diff_s1_c0))
    sd_diff_s1_c0 = float(np.std(diff_s1_c0, ddof=1))
    se_s1_c0 = sd_diff_s1_c0 / np.sqrt(n)
    t_stat_s1_c0, p_val_s1_c0 = stats.ttest_rel(s1_35k, c0_35k)
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_s1_c0 = [mean_diff_s1_c0 - t_crit * se_s1_c0, mean_diff_s1_c0 + t_crit * se_s1_c0]

    # Exact sign-flip permutation test (2^n permutations)
    total_permutations = 1 << n
    indices = np.arange(total_permutations, dtype=np.uint32)
    weights = np.abs(diff_s1_c0)
    obs_abs = np.abs(np.mean(diff_s1_c0))
    bit_sums = np.zeros(total_permutations, dtype=np.float64)
    for i in range(n):
        mask = (indices >> i) & 1
        bit_sums += mask * weights[i]
    perm_means = (2 * bit_sums - np.sum(weights)) / n
    extreme_count = int(np.sum(np.abs(perm_means) >= obs_abs - 1e-9))
    p_exact_sign_flip = float(extreme_count / total_permutations)

    # Secondary contrast: S1(35k) - C3(35k)
    diff_s1_c3 = s1_35k - c3_35k
    mean_diff_s1_c3 = float(np.mean(diff_s1_c3))
    t_stat_s1_c3, p_val_s1_c3 = stats.ttest_rel(s1_35k, c3_35k)

    # Secondary contrast: C3(35k) - C0(35k)
    diff_c3_c0 = c3_35k - c0_35k
    mean_diff_c3_c0 = float(np.mean(diff_c3_c0))
    t_stat_c3_c0, p_val_c3_c0 = stats.ttest_rel(c3_35k, c0_35k)

    # Secondary contrast: S1(35k) - C0(30k) [Matched pre-interruption target exposure]
    diff_s1_c0_30k = s1_35k - c0_30k
    mean_diff_s1_c0_30k = float(np.mean(diff_s1_c0_30k))
    t_stat_s1_c0_30k, p_val_s1_c0_30k = stats.ttest_rel(s1_35k, c0_30k)

    return {
        "n_seeds": n,
        "c0_35k_mean": float(np.mean(c0_35k)),
        "c0_35k_sd": float(np.std(c0_35k, ddof=1)),
        "s1_35k_mean": float(np.mean(s1_35k)),
        "s1_35k_sd": float(np.std(s1_35k, ddof=1)),
        "c3_35k_mean": float(np.mean(c3_35k)),
        "c3_35k_sd": float(np.std(c3_35k, ddof=1)),
        "c0_30k_mean": float(np.mean(c0_30k)),
        "c0_30k_sd": float(np.std(c0_30k, ddof=1)),
        "primary_s1_minus_c0": {
            "mean_diff_pp": mean_diff_s1_c0,
            "sd_diff_pp": sd_diff_s1_c0,
            "t_stat": float(t_stat_s1_c0),
            "p_value_parametric": float(p_val_s1_c0),
            "ci95_parametric": ci_s1_c0,
            "exact_sign_flip_p": p_exact_sign_flip,
            "total_permutations": total_permutations,
            "extreme_tail_count": extreme_count
        },
        "secondary_s1_minus_c3": {
            "mean_diff_pp": mean_diff_s1_c3,
            "t_stat": float(t_stat_s1_c3),
            "p_value": float(p_val_s1_c3)
        },
        "secondary_c3_minus_c0": {
            "mean_diff_pp": mean_diff_c3_c0,
            "t_stat": float(t_stat_c3_c0),
            "p_value": float(p_val_c3_c0)
        },
        "secondary_s1_minus_c0_30k": {
            "mean_diff_pp": mean_diff_s1_c0_30k,
            "t_stat": float(t_stat_s1_c0_30k),
            "p_value": float(p_val_s1_c0_30k)
        }
    }

def compute_factorial_stats(factorial_csv):
    df = pd.read_csv(factorial_csv)
    n = len(df)
    f00 = df['f00'].values
    f01 = df['f01'].values
    f10 = df['f10'].values
    f11 = df['f11'].values

    # Head main effect: ((f10 - f00) + (f11 - f01)) / 2
    head_effect = ((f10 - f00) + (f11 - f01)) / 2.0
    mean_head = float(np.mean(head_effect))
    sd_head = float(np.std(head_effect, ddof=1))
    se_head = sd_head / np.sqrt(n)
    t_stat_head, p_val_head = stats.ttest_1samp(head_effect, 0.0)
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_head = [mean_head - t_crit * se_head, mean_head + t_crit * se_head]

    # Momentum main effect: ((f01 - f00) + (f11 - f10)) / 2
    mom_effect = ((f01 - f00) + (f11 - f10)) / 2.0
    mean_mom = float(np.mean(mom_effect))
    sd_mom = float(np.std(mom_effect, ddof=1))
    se_mom = sd_mom / np.sqrt(n)
    t_stat_mom, p_val_mom = stats.ttest_1samp(mom_effect, 0.0)
    ci_mom = [mean_mom - t_crit * se_mom, mean_mom + t_crit * se_mom]

    # Head x momentum interaction: (f11 - f10) - (f01 - f00)
    inter_effect = (f11 - f10) - (f01 - f00)
    mean_inter = float(np.mean(inter_effect))
    sd_inter = float(np.std(inter_effect, ddof=1))
    se_inter = sd_inter / np.sqrt(n)
    t_stat_inter, p_val_inter = stats.ttest_1samp(inter_effect, 0.0)
    ci_inter = [mean_inter - t_crit * se_inter, mean_inter + t_crit * se_inter]

    return {
        "n_seeds": n,
        "cell_means": {
            "f00": float(np.mean(f00)),
            "f01": float(np.mean(f01)),
            "f10": float(np.mean(f10)),
            "f11": float(np.mean(f11))
        },
        "cell_sds": {
            "f00": float(np.std(f00, ddof=1)),
            "f01": float(np.std(f01, ddof=1)),
            "f10": float(np.std(f10, ddof=1)),
            "f11": float(np.std(f11, ddof=1))
        },
        "head_main_effect": {
            "mean_pp": mean_head,
            "sd_pp": sd_head,
            "t_stat": float(t_stat_head),
            "p_value": float(p_val_head),
            "ci95": ci_head
        },
        "momentum_main_effect": {
            "mean_pp": mean_mom,
            "sd_pp": sd_mom,
            "t_stat": float(t_stat_mom),
            "p_value": float(p_val_mom),
            "ci95": ci_mom
        },
        "interaction_effect": {
            "mean_pp": mean_inter,
            "sd_pp": sd_inter,
            "t_stat": float(t_stat_inter),
            "p_value": float(p_val_inter),
            "ci95": ci_inter
        }
    }

def main():
    parser = argparse.ArgumentParser(description="PAA Experimental Verification Statistical Auditor")
    parser.add_argument("--primary", required=True, help="Path to primary.csv")
    parser.add_argument("--factorial", required=True, help="Path to factorial.csv")
    parser.add_argument("--output", default="verification_results.json", help="Path to output JSON")
    args = parser.parse_args()

    primary_res = compute_primary_stats(args.primary)
    factorial_res = compute_factorial_stats(args.factorial)

    results = {
        "audit_target": "Pattern Analysis and Applications (PAA) SOTLI Manuscript",
        "primary_analysis": primary_res,
        "factorial_analysis": factorial_res
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 70)
    print("PAA EXPERIMENTAL VERIFICATION AUDIT RESULTS")
    print("=" * 70)
    p = primary_res["primary_s1_minus_c0"]
    print(f"Primary S1 - C0 Contrast (N = {primary_res['n_seeds']}):")
    print(f"  Mean Difference:      {p['mean_diff_pp']:+.4f} pp")
    print(f"  Paired t-test:        t({primary_res['n_seeds']-1}) = {p['t_stat']:.4f}, p = {p['p_value_parametric']:.4f}")
    print(f"  95% Confidence Int:   [{p['ci95_parametric'][0]:+.4f}, {p['ci95_parametric'][1]:+.4f}] pp")
    print(f"  Exact Sign-Flip Test: p = {p['exact_sign_flip_p']:.4f} ({p['extreme_tail_count']:,} / {p['total_permutations']:,} permutations)")

    f = factorial_res
    print("-" * 70)
    print(f"Factorial 2x2 Decomposition (N = {f['n_seeds']}):")
    h = f["head_main_effect"]
    print(f"  Head Main Effect:     {h['mean_pp']:+.4f} pp, t = {h['t_stat']:.4f}, p = {h['p_value']:.4f}")
    m = f["momentum_main_effect"]
    print(f"  Momentum Main Effect: {m['mean_pp']:+.4f} pp, t = {m['t_stat']:.4f}, p = {m['p_value']:.4f}")
    i = f["interaction_effect"]
    print(f"  Head x Momentum:      {i['mean_pp']:+.4f} pp, t = {i['t_stat']:.4f}, p = {i['p_value']:.4f}")
    print("=" * 70)
    print(f"Results successfully written to: {args.output}")

if __name__ == "__main__":
    main()
