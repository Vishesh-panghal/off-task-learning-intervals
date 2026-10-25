"""
SOTLI 5-Seed Vision Transformer Confirmatory Study (P-ViT-5).
Evaluates 5 fresh confirmatory seeds: [3001, 3002, 3003, 3004, 3005]
Across the Core Triad:
  - B0: Continuous Baseline (Control 1 at 35k, Control 2 at 30k)
  - B1: Clean On-Task Reset (Head swap + momentum clear, 5k Task A)
  - B4: Auxiliary Transfer (STL-10 natural images, 5k Task B)

Strictly enforces:
1. Shared Phase 1 baseline (15k updates) per seed (Checkpoint theta_1).
2. Paired within-seed branch architecture.
3. FP16 Automatic Mixed Precision (AMP) for optimal GPU throughput.
4. Resume capability: skips seeds/conditions already completed in CSV.
5. Automated statistical hypothesis evaluation: Paired t-test, Wilcoxon signed-rank,
   95% bootstrap CIs, Cohen's dz, and TOST equivalence testing.
"""

import os
import sys
import copy
import time
import json
import csv
import random
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.vit_compact import CompactViT
from src.data.datasets import (
    get_cifar100_loaders, get_auxiliary_loader, get_fixed_probe_set
)
from src.utils.metrics import (
    extract_probe_features, compute_effective_rank
)
from src.training.harness import (
    infinite_loader, stash_head_optimizer_state, restore_head_optimizer_state
)

CONFIRMATORY_VIT_SEEDS = [3001, 3002, 3003, 3004, 3005]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_accuracy_amp(model, loader, device, head='A'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(images, head=head)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return (correct / total) * 100.0 if total > 0 else 0.0


def run_linear_probe_amp(model, train_loader, val_loader, device, num_classes=100, epochs=3, lr=1e-2):
    model.eval()
    probe = nn.Linear(model.feat_dim, num_classes).to(device)
    optimizer = optim.SGD(probe.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        probe.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                feats = model.extract_features(images)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = probe(feats)
                loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

    probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            feats = model.extract_features(images)
            with torch.amp.autocast('cuda'):
                out = probe(feats)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return (correct / total) * 100.0


def bootstrap_ci(deltas, n_resamples=5000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    means = [np.mean(rng.choice(deltas, size=len(deltas), replace=True)) for _ in range(n_resamples)]
    return (
        float(np.mean(deltas)),
        float(np.std(deltas, ddof=1)),
        float(np.percentile(means, 100 * (alpha / 2))),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def compute_tost_pvalue(deltas, eq_margin=1.25):
    """Computes Two One-Sided Test (TOST) equivalence p-value"""
    d_bar = np.mean(deltas)
    s_d = np.std(deltas, ddof=1)
    n = len(deltas)
    se = s_d / np.sqrt(n)
    df = n - 1

    t1 = (d_bar - (-eq_margin)) / se
    p1 = 1.0 - stats.t.cdf(t1, df=df)

    t2 = (d_bar - eq_margin) / se
    p2 = stats.t.cdf(t2, df=df)

    p_tost = max(p1, p2)
    return float(t1), float(p1), float(t2), float(p2), float(p_tost)


def save_row_to_csv(row, filepath):
    file_exists = os.path.exists(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="SOTLI 5-Seed ViT Confirmatory Protocol")
    parser.add_argument("--seeds", nargs="+", type=int, default=CONFIRMATORY_VIT_SEEDS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_dir", type=str, default="./data_cache")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/vit_study")
    parser.add_argument("--results_dir", type=str, default="./results/vit_study")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    results_csv = os.path.join(args.results_dir, "vit_confirmatory_raw.csv")

    print("=" * 80)
    print(" SOTLI 5-SEED VISION TRANSFORMER CONFIRMATORY STUDY (P-ViT-5)")
    print(f" Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Precision: FP16 Automatic Mixed Precision (AMP)")
    print(f" Confirmatory Seeds ({len(args.seeds)}): {args.seeds}")
    print(f" Core Conditions: B0 (Continuous), B1 (Clean Reset), B4 (STL-10 Transfer)")
    print(f" Cumulative Updates per Run: 35,000 (Phase 1: 15k, Phase 2: 5k, Phase 3: 15k)")
    print(f" Results Destination: {results_csv}")
    print(f" Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Check already completed runs
    completed_runs = set()
    if os.path.exists(results_csv):
        with open(results_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'condition_id' in row and 'seed' in row:
                    try:
                        completed_runs.add((row['condition_id'], int(row['seed'])))
                    except ValueError:
                        pass
        print(f"Found {len(completed_runs)} already completed runs in {results_csv}.")

    probe_loader = get_fixed_probe_set(data_dir=args.data_dir, num_samples=2000, seed=42)
    criterion = nn.CrossEntropyLoss()

    total_study_start = time.time()

    for seed_idx, seed in enumerate(args.seeds, 1):
        print("\n" + "#" * 80)
        print(f" SEED [{seed_idx}/{len(args.seeds)}]: Seed {seed}")
        print("#" * 80)

        seed_conditions = ['B0', 'B1', 'B4']
        pending_conds = [c for c in seed_conditions if (c, seed) not in completed_runs]
        if not pending_conds:
            print(f"All conditions for Seed {seed} already completed. Skipping to next seed.")
            continue

        set_seed(seed)
        train_loader_a, val_loader_a, test_loader_a = get_cifar100_loaders(
            data_dir=args.data_dir, batch_size=128, seed=seed
        )

        theta1_path = os.path.join(args.checkpoint_dir, f"vit_seed{seed}_theta1.pt")
        opt1_path = os.path.join(args.checkpoint_dir, f"vit_seed{seed}_opt1.pt")

        # -------------------------------------------------------------
        # PHASE 1: Train or Load Shared Baseline Checkpoint theta_1
        # -------------------------------------------------------------
        if os.path.exists(theta1_path) and os.path.exists(opt1_path):
            print(f"\n[Seed {seed}] Reusing existing θ₁ checkpoint: {theta1_path}")
            ckpt = torch.load(theta1_path, map_location=device, weights_only=False)
            p_theta1_val = ckpt['val_acc']
            p_theta1_test = ckpt['test_acc']
            p_probe_theta1 = ckpt['probe_acc']
            eff_rank_theta1 = ckpt['eff_rank']
        else:
            print(f"\n[Seed {seed}] Phase 1: Training shared ViT baseline (15,000 updates)...")
            t_p1_start = time.time()
            model = CompactViT(img_size=32, patch_size=4, num_classes_a=100, embed_dim=192, depth=6, num_heads=4).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scaler = torch.amp.GradScaler('cuda')
            loader_a_iter = infinite_loader(train_loader_a)

            for step in range(1, 15001):
                model.train()
                images, labels = next(loader_a_iter)
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    outputs = model(images, head='A')
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if step % 2500 == 0 or step == 15000:
                    val_acc = evaluate_accuracy_amp(model, val_loader_a, device, head='A')
                    elapsed = time.time() - t_p1_start
                    print(f"[Seed {seed} | Phase 1] Step {step}/15000 | Loss: {loss.item():.4f} | Val Acc: {val_acc:.2f}% | Speed: {step/elapsed:.1f} steps/s")

            p_theta1_val = evaluate_accuracy_amp(model, val_loader_a, device, head='A')
            p_theta1_test = evaluate_accuracy_amp(model, test_loader_a, device, head='A')
            p_probe_theta1 = run_linear_probe_amp(model, train_loader_a, val_loader_a, device)
            probe_feats_theta1 = extract_probe_features(model, probe_loader, device)
            eff_rank_theta1 = compute_effective_rank(probe_feats_theta1)

            torch.save({
                'model_state': model.state_dict(),
                'val_acc': p_theta1_val,
                'test_acc': p_theta1_test,
                'probe_acc': p_probe_theta1,
                'eff_rank': eff_rank_theta1,
            }, theta1_path)
            torch.save({
                'optimizer_state': optimizer.state_dict(),
                'scaler_state': scaler.state_dict(),
            }, opt1_path)
            print(f"✓ Locked θ₁ checkpoint for Seed {seed}: Val = {p_theta1_val:.2f}%, Test = {p_theta1_test:.2f}%")

        # -------------------------------------------------------------
        # BRANCHES: B0, B1, B4
        # -------------------------------------------------------------
        # If B0 is needed or was already completed, get Control 2 val
        p_control2_val_stored = None

        for cond in ['B0', 'B1', 'B4']:
            if (cond, seed) in completed_runs:
                print(f"[Seed {seed} | {cond}] Already completed. Skipping.")
                continue

            print("\n" + "-" * 60)
            print(f" Executing Condition {cond} for Seed {seed}...")
            print("-" * 60)
            t_cond_start = time.time()

            branch_model = CompactViT(img_size=32, patch_size=4, num_classes_a=100, embed_dim=192, depth=6, num_heads=4).to(device)
            ckpt = torch.load(theta1_path, map_location=device, weights_only=False)
            branch_model.load_state_dict(ckpt['model_state'])

            opt_data = torch.load(opt1_path, map_location=device, weights_only=False)

            if cond == 'B0':
                branch_opt = optim.AdamW(branch_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                branch_opt.load_state_dict(opt_data['optimizer_state'])
                branch_scaler = torch.amp.GradScaler('cuda')
                branch_scaler.load_state_dict(opt_data['scaler_state'])
                loader_iter = infinite_loader(train_loader_a)

                p_ctrl2 = None
                for step in range(1, 20001):
                    branch_model.train()
                    images, labels = next(loader_iter)
                    images, labels = images.to(device), labels.to(device)

                    branch_opt.zero_grad()
                    with torch.amp.autocast('cuda'):
                        outputs = branch_model(images, head='A')
                        loss = criterion(outputs, labels)

                    branch_scaler.scale(loss).backward()
                    branch_scaler.step(branch_opt)
                    branch_scaler.update()

                    if step == 15000:
                        p_ctrl2 = evaluate_accuracy_amp(branch_model, val_loader_a, device, head='A')
                        p_control2_val_stored = p_ctrl2

                    if step % 2500 == 0 or step == 20000:
                        val_acc = evaluate_accuracy_amp(branch_model, val_loader_a, device, head='A')
                        print(f"[Seed {seed} | B0] Step {step}/20000 | Loss: {loss.item():.4f} | Val Acc: {val_acc:.2f}% | Elapsed: {(time.time()-t_cond_start)/60:.1f}m")

                val_final = evaluate_accuracy_amp(branch_model, val_loader_a, device, head='A')
                test_final = evaluate_accuracy_amp(branch_model, test_loader_a, device, head='A')
                probe_final = run_linear_probe_amp(branch_model, train_loader_a, val_loader_a, device)
                probe_feats_final = extract_probe_features(branch_model, probe_loader, device)
                eff_rank_final = compute_effective_rank(probe_feats_final)

                row = {
                    'seed': seed,
                    'condition_id': 'B0',
                    'p_theta1_val': p_theta1_val,
                    'p_theta1_test': p_theta1_test,
                    'p_probe_theta1': p_probe_theta1,
                    'eff_rank_theta1': eff_rank_theta1,
                    'p_theta2_val': p_theta1_val,
                    'p_probe_theta2': p_probe_theta1,
                    'eff_rank_theta2': eff_rank_theta1,
                    'p_control2_val': p_ctrl2,
                    'p_theta3_val': val_final,
                    'p_theta3_test': test_final,
                    'p_probe_theta3': probe_final,
                    'eff_rank_theta3': eff_rank_final,
                    'runtime_minutes': (time.time() - t_cond_start) / 60.0,
                }

            else:
                # SOTLI Schedule (B1 or B4)
                opt_a = optim.AdamW(branch_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                opt_a.load_state_dict(opt_data['optimizer_state'])

                head_a_param_ids = {id(p) for p in branch_model.head_a.parameters()}
                stashed_opt_state = stash_head_optimizer_state(opt_a, head_a_param_ids)
                branch_model.freeze_head_a()

                if cond == 'B1':
                    aux_loader_b, num_classes_b = train_loader_a, 100
                elif cond == 'B4':
                    aux_loader_b, num_classes_b = get_auxiliary_loader('B4', data_dir=args.data_dir, batch_size=128, seed=seed)

                branch_model.init_auxiliary_head(num_classes_b, device=device)
                opt_b = optim.AdamW([p for p in branch_model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
                scaler_b = torch.amp.GradScaler('cuda')
                loader_b_iter = infinite_loader(aux_loader_b)

                for step in range(1, 5001):
                    branch_model.train()
                    images, labels = next(loader_b_iter)
                    images, labels = images.to(device), labels.to(device)

                    opt_b.zero_grad()
                    with torch.amp.autocast('cuda'):
                        outputs = branch_model(images, head='B')
                        loss = criterion(outputs, labels)

                    scaler_b.scale(loss).backward()
                    scaler_b.step(opt_b)
                    scaler_b.update()

                p_theta2_val = evaluate_accuracy_amp(branch_model, val_loader_a, device, head='A')
                p_probe_theta2 = run_linear_probe_amp(branch_model, train_loader_a, val_loader_a, device)
                probe_feats_theta2 = extract_probe_features(branch_model, probe_loader, device)
                eff_rank_theta2 = compute_effective_rank(probe_feats_theta2)

                branch_model.discard_auxiliary_head()
                branch_model.unfreeze_head_a()
                opt_p3 = optim.AdamW(branch_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                restore_head_optimizer_state(opt_p3, head_a_param_ids, stashed_opt_state)
                scaler_p3 = torch.amp.GradScaler('cuda')
                loader_a_iter = infinite_loader(train_loader_a)

                for step in range(1, 15001):
                    branch_model.train()
                    images, labels = next(loader_a_iter)
                    images, labels = images.to(device), labels.to(device)

                    opt_p3.zero_grad()
                    with torch.amp.autocast('cuda'):
                        outputs = branch_model(images, head='A')
                        loss = criterion(outputs, labels)

                    scaler_p3.scale(loss).backward()
                    scaler_p3.step(opt_p3)
                    scaler_p3.update()

                    if step % 2500 == 0 or step == 15000:
                        val_acc = evaluate_accuracy_amp(branch_model, val_loader_a, device, head='A')
                        print(f"[Seed {seed} | {cond} Phase 3] Step {step}/15000 | Loss: {loss.item():.4f} | Val Acc: {val_acc:.2f}% | Elapsed: {(time.time()-t_cond_start)/60:.1f}m")

                val_final = evaluate_accuracy_amp(branch_model, val_loader_a, device, head='A')
                test_final = evaluate_accuracy_amp(branch_model, test_loader_a, device, head='A')
                probe_final = run_linear_probe_amp(branch_model, train_loader_a, val_loader_a, device)
                probe_feats_final = extract_probe_features(branch_model, probe_loader, device)
                eff_rank_final = compute_effective_rank(probe_feats_final)

                row = {
                    'seed': seed,
                    'condition_id': cond,
                    'p_theta1_val': p_theta1_val,
                    'p_theta1_test': p_theta1_test,
                    'p_probe_theta1': p_probe_theta1,
                    'eff_rank_theta1': eff_rank_theta1,
                    'p_theta2_val': p_theta2_val,
                    'p_probe_theta2': p_probe_theta2,
                    'eff_rank_theta2': eff_rank_theta2,
                    'p_control2_val': 'N/A',
                    'p_theta3_val': val_final,
                    'p_theta3_test': test_final,
                    'p_probe_theta3': probe_final,
                    'eff_rank_theta3': eff_rank_final,
                    'runtime_minutes': (time.time() - t_cond_start) / 60.0,
                }

            save_row_to_csv(row, results_csv)
            completed_runs.add((cond, seed))
            print(f"✓ Saved result for Seed {seed} | {cond} -> {results_csv}")

    # -------------------------------------------------------------------------
    # AUTOMATED STATISTICAL HYPOTHESIS EVALUATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" ALL 5 SEEDS COMPLETED! COMPUTING HYPOTHESIS TESTS ON ViT...")
    print("=" * 80)

    df = pd.read_csv(results_csv)
    b0_df = df[df['condition_id'] == 'B0'].set_index('seed')
    b1_df = df[df['condition_id'] == 'B1'].set_index('seed')
    b4_df = df[df['condition_id'] == 'B4'].set_index('seed')

    common_seeds = sorted(list(set(b0_df.index) & set(b1_df.index) & set(b4_df.index)))
    n_seeds = len(common_seeds)
    print(f"Evaluating across {n_seeds} paired seeds: {common_seeds}")

    # 1. Decisive Paired Contrast B4 - B1
    diff_val_b4_b1 = (b4_df.loc[common_seeds, 'p_theta3_val'] - b1_df.loc[common_seeds, 'p_theta3_val']).values
    diff_test_b4_b1 = (b4_df.loc[common_seeds, 'p_theta3_test'] - b1_df.loc[common_seeds, 'p_theta3_test']).values

    m_v, s_v, lo_v, hi_v = bootstrap_ci(diff_val_b4_b1)
    m_t, s_t, lo_t, hi_t = bootstrap_ci(diff_test_b4_b1)

    t_val, p_val = stats.ttest_rel(b4_df.loc[common_seeds, 'p_theta3_val'], b1_df.loc[common_seeds, 'p_theta3_val'])
    w_val, p_wilc_val = stats.wilcoxon(diff_val_b4_b1) if np.any(diff_val_b4_b1 != 0) else (0, 1.0)
    d_z_val = m_v / s_v if s_v > 0 else 0.0

    t1, p1, t2, p2, p_tost = compute_tost_pvalue(diff_val_b4_b1, eq_margin=1.25)

    # 2. Equal Target Updates Δ_A (vs Control 2 @30k)
    ctrl2_vals = b0_df.loc[common_seeds, 'p_control2_val'].astype(float).values
    diff_a_b4 = b4_df.loc[common_seeds, 'p_theta3_val'].values - ctrl2_vals
    diff_a_b1 = b1_df.loc[common_seeds, 'p_theta3_val'].values - ctrl2_vals

    m_a_b4, s_a_b4, lo_a_b4, hi_a_b4 = bootstrap_ci(diff_a_b4)
    m_a_b1, s_a_b1, lo_a_b1, hi_a_b1 = bootstrap_ci(diff_a_b1)

    # 3. Equal Total Updates Δ_total (vs Control 1 @35k)
    b0_final_val = b0_df.loc[common_seeds, 'p_theta3_val'].values
    diff_tot_b4 = b4_df.loc[common_seeds, 'p_theta3_val'].values - b0_final_val
    diff_tot_b1 = b1_df.loc[common_seeds, 'p_theta3_val'].values - b0_final_val

    m_tot_b4, s_tot_b4, lo_tot_b4, hi_tot_b4 = bootstrap_ci(diff_tot_b4)
    m_tot_b1, s_tot_b1, lo_tot_b1, hi_tot_b1 = bootstrap_ci(diff_tot_b1)

    print("\n" + "-" * 80)
    print(" SOTLI VISION TRANSFORMER CONFIRMATORY REPORT (N=5 SEEDS)")
    print("-" * 80)
    print(f"1. Decisive Contrast Δ(B4 - B1) Validation:")
    print(f"   Mean = {m_v:+.2f} pp [95% CI: {lo_v:+.2f}, {hi_v:+.2f}], SD = {s_v:.2f} pp")
    print(f"   t({n_seeds-1}) = {t_val:.3f}, p_ttest = {p_val:.4f}, p_wilcoxon = {p_wilc_val:.4f}, dz = {d_z_val:.2f}")
    print(f"   TOST Equivalence (margin ±1.25 pp): t1={t1:+.3f} (p={p1:.4f}), t2={t2:+.3f} (p={p2:.4f}) -> p_TOST = {p_tost:.4f}")
    print(f"\n2. Decisive Contrast Δ(B4 - B1) Holdout Test:")
    print(f"   Mean = {m_t:+.2f} pp [95% CI: {lo_t:+.2f}, {hi_t:+.2f}], SD = {s_t:.2f} pp")
    print(f"\n3. Equal Target Updates Δ_A (vs 30k update baseline):")
    print(f"   B4 (STL-10): {m_a_b4:+.2f} pp [95% CI: {lo_a_b4:+.2f}, {hi_a_b4:+.2f}]")
    print(f"   B1 (Reset) : {m_a_b1:+.2f} pp [95% CI: {lo_a_b1:+.2f}, {hi_a_b1:+.2f}]")
    print(f"\n4. Equal Total Compute Δ_total (vs 35k update baseline):")
    print(f"   B4 (STL-10): {m_tot_b4:+.2f} pp [95% CI: {lo_tot_b4:+.2f}, {hi_tot_b4:+.2f}]")
    print(f"   B1 (Reset) : {m_tot_b1:+.2f} pp [95% CI: {lo_tot_b1:+.2f}, {hi_tot_b1:+.2f}]")
    print("-" * 80 + "\n")

    summary_json = os.path.join(args.results_dir, "vit_confirmatory_summary.json")
    with open(summary_json, 'w') as f:
        json.dump({
            'n_seeds': n_seeds,
            'seeds': common_seeds,
            'total_runtime_hours': (time.time() - total_study_start) / 3600.0,
            'b4_vs_b1_val': {
                'mean': m_v, 'sd': s_v, 'ci_low': lo_v, 'ci_high': hi_v,
                't_stat': float(t_val), 'p_ttest': float(p_val),
                'p_wilcoxon': float(p_wilc_val), 'cohen_dz': float(d_z_val),
                'p_tost_125': float(p_tost),
            },
            'b4_vs_b1_test': {
                'mean': m_t, 'sd': s_t, 'ci_low': lo_t, 'ci_high': hi_t,
            },
            'delta_a': {
                'b4_mean': m_a_b4, 'b4_ci_low': lo_a_b4, 'b4_ci_high': hi_a_b4,
                'b1_mean': m_a_b1, 'b1_ci_low': lo_a_b1, 'b1_ci_high': hi_a_b1,
            },
            'delta_total': {
                'b4_mean': m_tot_b4, 'b4_ci_low': lo_tot_b4, 'b4_ci_high': hi_tot_b4,
                'b1_mean': m_tot_b1, 'b1_ci_low': lo_tot_b1, 'b1_ci_high': hi_tot_b1,
            }
        }, f, indent=2)
    print(f"✓ Saved full statistical summary to {summary_json}")


if __name__ == "__main__":
    main()
