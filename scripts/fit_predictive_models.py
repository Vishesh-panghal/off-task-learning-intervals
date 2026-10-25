"""
Empirical Model Comparison Engine for SOTLI (Gate 3).
Fits and evaluates three competing scientific hypotheses:
1. Null Model (constant baseline)
2. Monotonic Linear Model
3. Inverted-U Non-Monotonic Model

Evaluates across three similarity axes:
- Shared-Backbone Gradient Cosine S_g
- Fixed-Probe Representation Drift D_CKA
- Task2Vec Fisher Distance d_task
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCREENING_CSV = os.path.join(PROJECT_ROOT, "results", "screening_results.csv")
TASK2VEC_CSV = os.path.join(PROJECT_ROOT, "results", "task2vec_distances.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")


def compute_metrics(y_true, y_pred, num_params):
    n = len(y_true)
    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - num_params, 1)
    mse = ss_res / n
    # AIC and BIC (assuming Gaussian errors)
    if ss_res > 0:
        aic = n * np.log(ss_res / n) + 2 * num_params
        bic = n * np.log(ss_res / n) + num_params * np.log(n)
    else:
        aic, bic = -np.inf, -np.inf
    return {
        'R2': round(float(r2), 4),
        'Adj_R2': round(float(adj_r2), 4),
        'MSE': round(float(mse), 4),
        'AIC': round(float(aic), 2),
        'BIC': round(float(bic), 2)
    }


def main():
    print("=================================================================")
    print(" SOTLI GATE 3: EMPIRICAL PREDICTIVE SIMILARITY MODEL COMPARISON")
    print(" Target Venue: TMLR")
    print("=================================================================\n")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    df_screen = pd.read_csv(SCREENING_CSV)
    df_task2vec = pd.read_csv(TASK2VEC_CSV)

    # Compute baseline reference values per seed for Delta_total and Delta_A
    b0_vals = df_screen[df_screen['condition_id'] == 'B0'].set_index('seed')['p_theta3_val']
    b0_intermediates = df_screen[df_screen['condition_id'] == 'B0'].set_index('seed')['p_control2_val']

    df_screen['delta_total'] = df_screen.apply(lambda r: r['p_theta3_val'] - b0_vals[r['seed']], axis=1)
    df_screen['delta_a'] = df_screen.apply(lambda r: r['p_theta3_val'] - b0_intermediates[r['seed']], axis=1)

    # Merge task2vec distance
    t2v_map = df_task2vec.set_index('condition_id')['d_task'].to_dict()
    df_screen['d_task'] = df_screen['condition_id'].map(t2v_map)

    # Aggregate by condition
    cond_stats = df_screen.groupby('condition_id').agg({
        'delta_total': ['mean', 'std', 'count'],
        'delta_a': ['mean', 'std'],
        's_g': 'mean',
        'cka_drift': 'mean',
        'd_task': 'mean'
    })
    
    # Flatten multi-index
    cond_df = pd.DataFrame({
        'condition_id': cond_stats.index,
        'delta_total': cond_stats[('delta_total', 'mean')],
        'delta_total_std': cond_stats[('delta_total', 'std')],
        'delta_a': cond_stats[('delta_a', 'mean')],
        'delta_a_std': cond_stats[('delta_a', 'std')],
        's_g': cond_stats[('s_g', 'mean')],
        'cka_drift': cond_stats[('cka_drift', 'mean')],
        'd_task': cond_stats[('d_task', 'mean')],
    }).reset_index(drop=True)

    print("--- Condition Summary Statistics ---")
    print(cond_df[['condition_id', 'delta_total', 'delta_a', 's_g', 'cka_drift', 'd_task']].to_string(index=False))
    print("\n")

    # Fit models for both Delta_total and Delta_A across each metric axis
    axes = ['s_g', 'cka_drift', 'd_task']
    targets = ['delta_total', 'delta_a']
    
    model_results = []

    for target in targets:
        y = cond_df[target].values
        # Null model
        y_null_pred = np.full_like(y, np.mean(y))
        null_metrics = compute_metrics(y, y_null_pred, num_params=1)
        model_results.append({
            'Target': target,
            'Metric_Axis': 'All',
            'Model_Type': 'Null (Constant)',
            **null_metrics
        })

        for axis in axes:
            x = cond_df[axis].values
            
            # Linear Monotonic: y = b0 + b1 * x
            p_lin = np.polyfit(x, y, deg=1)
            y_lin_pred = np.polyval(p_lin, x)
            lin_metrics = compute_metrics(y, y_lin_pred, num_params=2)
            model_results.append({
                'Target': target,
                'Metric_Axis': axis,
                'Model_Type': 'Monotonic Linear',
                **lin_metrics
            })

            # Inverted-U (Quadratic): y = b0 + b1 * x + b2 * x^2
            p_quad = np.polyfit(x, y, deg=2)
            y_quad_pred = np.polyval(p_quad, x)
            quad_metrics = compute_metrics(y, y_quad_pred, num_params=3)
            model_results.append({
                'Target': target,
                'Metric_Axis': axis,
                'Model_Type': 'Inverted-U (Quadratic)',
                **quad_metrics
            })

    df_models = pd.DataFrame(model_results)
    summary_csv = os.path.join(OUTPUT_DIR, "predictive_models_summary.csv")
    df_models.to_csv(summary_csv, index=False)
    print(f"✓ Model comparison summary saved -> {summary_csv}\n")
    print("--- Predictive Model Comparison Results ---")
    print(df_models.to_string(index=False))

    # -------------------------------------------------------------
    # Generate Publication Figure 3: Predictive Similarity Curves
    # -------------------------------------------------------------
    plt.rcParams.update({'font.size': 11, 'mathtext.fontset': 'cm', 'font.family': 'sans-serif'})
    fig, axes_plot = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    
    axis_configs = [
        ('d_task', 'Task2Vec Fisher Distance $d_{\\text{task}}$', axes_plot[0]),
        ('cka_drift', 'Fixed-Probe CKA Drift $D_{\\text{CKA}}$', axes_plot[1]),
        ('s_g', 'Shared-Backbone Gradient Cosine $S_g$', axes_plot[2])
    ]

    colors = {
        'B0': '#7f8c8d', 'B1': '#2980b9', 'B3': '#27ae60',
        'B4': '#d35400', 'B5': '#8e44ad', 'B7': '#c0392b'
    }

    for metric_col, label_name, ax in axis_configs:
        x_pts = cond_df[metric_col].values
        y_pts = cond_df['delta_a'].values
        y_err = cond_df['delta_a_std'].values
        
        # Plot points with error bars
        for _, row in cond_df.iterrows():
            c_id = row['condition_id']
            ax.errorbar(
                row[metric_col], row['delta_a'], yerr=row['delta_a_std'],
                fmt='o', color=colors[c_id], ecolor=colors[c_id],
                elinewidth=1.5, capsize=4, markersize=8, zorder=5
            )
            # Label point
            ax.annotate(
                f"{c_id}", (row[metric_col], row['delta_a']),
                textcoords="offset points", xytext=(6, 6),
                fontsize=10, fontweight='bold', color=colors[c_id]
            )

        # Plot Quadratic fit line
        x_dense = np.linspace(min(x_pts) - 0.05, max(x_pts) + 0.05, 200)
        p_quad = np.polyfit(x_pts, y_pts, deg=2)
        y_quad_dense = np.polyval(p_quad, x_dense)
        p_lin = np.polyfit(x_pts, y_pts, deg=1)
        y_lin_dense = np.polyval(p_lin, x_dense)

        # Get R2 scores
        r2_quad = df_models[(df_models['Target'] == 'delta_a') & (df_models['Metric_Axis'] == metric_col) & (df_models['Model_Type'] == 'Inverted-U (Quadratic)')]['R2'].values[0]
        r2_lin = df_models[(df_models['Target'] == 'delta_a') & (df_models['Metric_Axis'] == metric_col) & (df_models['Model_Type'] == 'Monotonic Linear')]['R2'].values[0]

        ax.plot(x_dense, y_quad_dense, '--', color='#e67e22', linewidth=2.0, label=f'Inverted-U ($R^2={r2_quad:.2f}$)')
        ax.plot(x_dense, y_lin_dense, ':', color='#34495e', linewidth=1.5, label=f'Linear ($R^2={r2_lin:.2f}$)')
        
        # Zero baseline line
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.7)
        ax.set_xlabel(label_name, fontsize=12, fontweight='bold')
        ax.set_ylabel(r'Compute-Matched Gain $\Delta_A$ (%)', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(frameon=True, facecolor='white', loc='lower left' if metric_col == 's_g' else 'upper right')
        ax.set_title(f"Gain vs. {label_name.split()[0]}", fontsize=13, fontweight='bold', pad=10)

    fig.suptitle("SOTLI Gate 3: Empirical Predictive Similarity Models Across Metric Axes", fontsize=15, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    fig_path = os.path.join(FIGURES_DIR, "figure3_predictive_similarity_curves.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"\n✓ Generated Figure 3 -> {fig_path}")

    print("\n=================================================================")
    print(" PREDICTIVE MODELING COMPLETED SUCCESSFULLY! 🎉")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
