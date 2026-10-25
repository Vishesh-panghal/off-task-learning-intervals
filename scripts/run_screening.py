"""
Screening Runner for SOTLI Stage 1 (Gate 1).
Executes the 30-run Discovery Screening Matrix (6 conditions x 5 seeds):
  - B0: Continuous Baseline (Control 1: 35k updates, Control 2: 30k updates)
  - B1: Head-Switch Control (CIFAR-100 with temporary head for 5k updates)
  - B3: Related Natural Images (CIFAR-10 for 5k updates)
  - B4: Distinct Natural Images (STL-10 for 5k updates)
  - B5: Distant Real Domain (SVHN for 5k updates)
  - B7: Synthetic Random Noise (Fixed Gaussian Noise for 5k updates)

Strictly enforces:
1. Paired branch architecture: all conditions branch from the exact same theta_1 checkpoint per seed.
2. Independent RNG streams for Task A vs Task B loaders.
3. Checkpoint exports at theta_1 (15k), theta_2 (20k), theta_3 (35k), and theta_C_A (30k).
4. Dense validation tracking during Phase 3 (every 100 updates).
5. Comprehensive metrics logging to results/screening_results.csv.
"""

import os
import sys
import copy
import argparse
import random
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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
    compute_linear_cka, compute_gradient_cosine_backbone
)
from src.training.evaluator import RelearningDynamicsTracker, run_diagnostic_probe_at_theta2
from src.training.harness import (
    infinite_loader, stash_head_optimizer_state, restore_head_optimizer_state
)


def set_seed(seed):
    """Sets deterministic seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_default_device():
    """Detects available acceleration hardware: MPS -> CUDA -> CPU"""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train_phase1_if_needed(seed, data_dir, checkpoint_dir, u1, lr, momentum, weight_decay, device, smoke_test=False):
    """
    Executes or loads Phase 1 training (u1 updates on Task A).
    Returns (theta1_path, opt1_path, p_before, eff_rank_theta1).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    suffix = "_smoke" if smoke_test else ""
    theta1_path = os.path.join(checkpoint_dir, f"seed_{seed}_theta1{suffix}.pt")
    opt1_path = os.path.join(checkpoint_dir, f"seed_{seed}_opt_a_theta1{suffix}.pt")
    meta_path = os.path.join(checkpoint_dir, f"seed_{seed}_meta_theta1{suffix}.pt")

    if os.path.exists(theta1_path) and os.path.exists(opt1_path) and os.path.exists(meta_path):
        print(f"[Seed {seed}] Found cached Checkpoint θ₁: {theta1_path}")
        meta = torch.load(meta_path, map_location='cpu')
        p_probe_theta1 = meta.get('p_probe_theta1', meta.get('p_before', 0.0))
        return theta1_path, opt1_path, meta['p_before'], meta['eff_rank_theta1'], p_probe_theta1

    print(f"\n========================================================")
    print(f"[Seed {seed}] Training Phase 1 on Task A for {u1} updates...")
    print(f"========================================================")

    set_seed(seed)
    train_loader_a, val_loader_a, _ = get_cifar100_loaders(data_dir=data_dir, batch_size=128, seed=seed)
    probe_loader = get_fixed_probe_set(data_dir=data_dir, num_samples=2000, seed=42)

    model = ResNet18GroupNorm(num_classes_a=100, num_groups=32).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    loader_iter = infinite_loader(train_loader_a)
    log_freq = 10 if smoke_test else 2500

    for step in range(1, u1 + 1):
        model.train()
        images, labels = next(loader_iter)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, head='A')
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if step % log_freq == 0 or step == u1:
            val_acc = evaluate_accuracy(model, val_loader_a, device, head='A')
            print(f"[Seed {seed} | Phase 1] Step {step}/{u1} - Loss: {loss.item():.4f} - Val Acc: {val_acc:.2f}%")

    p_before = evaluate_accuracy(model, val_loader_a, device, head='A')
    probe_feats_theta1 = extract_probe_features(model, probe_loader, device)
    eff_rank_theta1 = compute_effective_rank(probe_feats_theta1)

    probe_epochs = 1 if smoke_test else 5
    p_probe_theta1 = run_diagnostic_probe_at_theta2(
        model, train_loader_a, val_loader_a, device, num_classes_a=100, probe_epochs=probe_epochs
    )

    torch.save(model.state_dict(), theta1_path)
    torch.save(optimizer.state_dict(), opt1_path)
    torch.save({
        'p_before': p_before,
        'eff_rank_theta1': eff_rank_theta1,
        'probe_feats_theta1': probe_feats_theta1,
        'p_probe_theta1': p_probe_theta1
    }, meta_path)

    print(f"[Seed {seed}] Checkpoint θ₁ serialized. P_before: {p_before:.2f}%, Probe: {p_probe_theta1:.2f}%, EffRank: {eff_rank_theta1:.2f}\n")
    return theta1_path, opt1_path, p_before, eff_rank_theta1, p_probe_theta1


def run_condition_b0(seed, theta1_path, opt1_path, p_before, eff_rank_theta1, data_dir, checkpoint_dir,
                     u_b, u2, lr, momentum, weight_decay, device, smoke_test=False, p_probe_theta1=None):
    """
    Executes Condition B0 (Continuous Baseline on Task A).
    Total continuation updates = u_b + u2 = 20,000 updates.
    Exports:
      - theta_C_A at update u2 (cumulative 30k updates, Control 2: Equal Task A Exposure)
      - theta_C_total at update u_b + u2 (cumulative 35k updates, Control 1: Equal Total Updates)
    """
    print(f"\n--- Running Condition B0 (Continuous Baseline) [Seed {seed}] ---")
    set_seed(seed)
    train_loader_a, val_loader_a, test_loader_a = get_cifar100_loaders(data_dir=data_dir, batch_size=128, seed=seed)
    probe_loader = get_fixed_probe_set(data_dir=data_dir, num_samples=2000, seed=42)

    model = ResNet18GroupNorm(num_classes_a=100, num_groups=32).to(device)
    model.load_state_dict(torch.load(theta1_path, map_location=device))

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer.load_state_dict(torch.load(opt1_path, map_location=device))
    criterion = nn.CrossEntropyLoss()

    total_cont_updates = u_b + u2
    control2_target_step = u2  # +15k steps from theta1 => 30k total updates
    log_freq = 10 if smoke_test else 2500

    loader_iter = infinite_loader(train_loader_a)
    p_control2_val = None
    suffix = "_smoke" if smoke_test else ""

    theta_c_a_path = os.path.join(checkpoint_dir, f"seed_{seed}_B0_theta_C_A_30k{suffix}.pt")
    theta_c_total_path = os.path.join(checkpoint_dir, f"seed_{seed}_B0_theta_C_total_35k{suffix}.pt")

    for step in range(1, total_cont_updates + 1):
        model.train()
        images, labels = next(loader_iter)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, head='A')
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Intermediate checkpoint for Control 2 (30k updates)
        if step == control2_target_step:
            p_control2_val = evaluate_accuracy(model, val_loader_a, device, head='A')
            torch.save(model.state_dict(), theta_c_a_path)
            print(f"[Seed {seed} | B0] Step {step} (+15k): Saved Control 2 Checkpoint (Val Acc: {p_control2_val:.2f}%)")

        if step % log_freq == 0 or step == total_cont_updates:
            val_acc = evaluate_accuracy(model, val_loader_a, device, head='A')
            print(f"[Seed {seed} | B0] Continuation Step {step}/{total_cont_updates} - Val Acc: {val_acc:.2f}%")

    # Final Checkpoint for Control 1 (35k updates)
    p_control1_val = evaluate_accuracy(model, val_loader_a, device, head='A')
    p_control1_test = evaluate_accuracy(model, test_loader_a, device, head='A')
    torch.save(model.state_dict(), theta_c_total_path)

    probe_epochs = 1 if smoke_test else 5
    p_probe_theta3 = run_diagnostic_probe_at_theta2(
        model, train_loader_a, val_loader_a, device, num_classes_a=100, probe_epochs=probe_epochs
    )

    probe_feats_theta3 = extract_probe_features(model, probe_loader, device)
    eff_rank_theta3 = compute_effective_rank(probe_feats_theta3)

    return {
        'run_id': f"B0_seed{seed}{suffix}",
        'condition_id': 'B0',
        'seed': seed,
        'checkpoint_theta1': theta1_path,
        'checkpoint_theta2': 'N/A',
        'checkpoint_theta3': theta_c_total_path,
        'p_theta1': p_before,
        'p_theta2': p_before,  # No interval interruption
        'f_b': 0.0,
        'p_diagnostic_probe': p_before,
        'p_probe_theta1': p_probe_theta1 if p_probe_theta1 is not None else p_before,
        'p_probe_theta2': p_before,
        'p_probe_theta3': p_probe_theta3,
        'p_theta3_val': p_control1_val,
        'p_theta3_test': p_control1_test,
        'p_control2_val': p_control2_val,
        'r_a': p_control1_val - p_before,
        's_g': 1.0,
        'cka_drift': 0.0,
        'eff_rank_theta1': eff_rank_theta1,
        'eff_rank_theta2': eff_rank_theta1,
        'eff_rank_theta3': eff_rank_theta3,
        'u_rec': 0,
        'aulc': p_control1_val,
        'acc_horizon_500': p_before,
        'acc_horizon_1000': p_before,
        'acc_horizon_2000': p_before,
    }


def run_condition_sotli(condition_id, seed, theta1_path, opt1_path, p_before, eff_rank_theta1,
                        data_dir, checkpoint_dir, results_dir,
                        u_b, u2, lr, momentum, weight_decay, device, smoke_test=False,
                        p_probe_theta1=None):
    """
    Executes an auxiliary interval condition (B1, B3, B4, B5, B7) branching from theta_1.
    """
    suffix = "_smoke" if smoke_test else ""
    run_id = f"{condition_id}_seed{seed}{suffix}"
    print(f"\n--- Running Condition {condition_id} [Seed {seed}] ---")

    set_seed(seed)
    train_loader_a, val_loader_a, test_loader_a = get_cifar100_loaders(data_dir=data_dir, batch_size=128, seed=seed)
    probe_loader = get_fixed_probe_set(data_dir=data_dir, num_samples=2000, seed=42)
    meta_path = os.path.join(checkpoint_dir, f"seed_{seed}_meta_theta1{suffix}.pt")
    meta = torch.load(meta_path, map_location='cpu')
    probe_feats_theta1 = meta['probe_feats_theta1']

    # Load shared backbone and Head A from theta_1
    model = ResNet18GroupNorm(num_classes_a=100, num_groups=32).to(device)
    model.load_state_dict(torch.load(theta1_path, map_location=device))

    # Load Optimizer A to get its state
    optimizer_a = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer_a.load_state_dict(torch.load(opt1_path, map_location=device))
    criterion = nn.CrossEntropyLoss()

    # Obtain auxiliary loader
    aux_loader_b, num_classes_b = get_auxiliary_loader(
        condition_id=condition_id, data_dir=data_dir, batch_size=128, seed=seed
    )

    # -------------------------------------------------------------
    # TRANSITION A -> B
    # -------------------------------------------------------------
    head_a_param_ids = {id(p) for p in model.head_a.parameters()}
    stashed_opt_state = stash_head_optimizer_state(optimizer_a, head_a_param_ids)
    model.freeze_head_a()
    model.init_auxiliary_head(num_classes_b=num_classes_b, device=device)

    # Pre-interval Gradient Cosine S_g on shared backbone
    loader_a_iter = infinite_loader(train_loader_a)
    loader_b_iter = infinite_loader(aux_loader_b)
    batch_a_sample = next(loader_a_iter)
    batch_b_sample = next(loader_b_iter)
    s_g = compute_gradient_cosine_backbone(model, batch_a_sample, batch_b_sample, criterion, device)
    print(f"[{run_id}] Shared-Backbone Gradient Cosine S_g(A, B): {s_g:.4f}")

    # -------------------------------------------------------------
    # PHASE 2: Auxiliary Interval Training (u_b updates)
    # -------------------------------------------------------------
    print(f"[{run_id}] Phase 2: Training on {condition_id} for {u_b} updates...")
    optimizer_b = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    log_freq_p2 = 10 if smoke_test else 1000
    for step in range(1, u_b + 1):
        model.train()
        images, labels = next(loader_b_iter)
        images, labels = images.to(device), labels.to(device)

        optimizer_b.zero_grad()
        outputs = model(images, head='B')
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_b.step()

        if step % log_freq_p2 == 0 or step == u_b:
            print(f"[{run_id} | Phase 2] Step {step}/{u_b} - Aux Loss: {loss.item():.4f}")

    # Checkpoint theta_2 (End of Phase 2)
    theta2_path = os.path.join(checkpoint_dir, f"seed_{seed}_{condition_id}_theta2{suffix}.pt")
    torch.save(model.state_dict(), theta2_path)

    # Evaluations at theta_2
    p_after_b = evaluate_accuracy(model, val_loader_a, device, head='A')
    f_b = p_before - p_after_b

    probe_epochs = 1 if smoke_test else 5
    p_diagnostic_probe = run_diagnostic_probe_at_theta2(
        model, train_loader_a, val_loader_a, device, num_classes_a=100, probe_epochs=probe_epochs
    )

    probe_feats_theta2 = extract_probe_features(model, probe_loader, device)
    cka_drift = 1.0 - compute_linear_cka(probe_feats_theta1, probe_feats_theta2)
    eff_rank_theta2 = compute_effective_rank(probe_feats_theta2)

    print(f"[{run_id}] Checkpoint θ₂ locked.")
    print(f"       P_after_B (Frozen Head A): {p_after_b:.2f}% (Drop F_B: {f_b:.2f}%)")
    print(f"       Diagnostic Probe on Frozen Backbone: {p_diagnostic_probe:.2f}%")
    print(f"       Representation Drift D_CKA: {cka_drift:.4f}, EffRank: {eff_rank_theta2:.2f}")

    # -------------------------------------------------------------
    # TRANSITION B -> A
    # -------------------------------------------------------------
    model.discard_auxiliary_head()
    model.unfreeze_head_a()
    # Re-link optimizer A and restore Head A momentum buffer
    optimizer_a = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    restore_head_optimizer_state(optimizer_a, head_a_param_ids, stashed_opt_state)

    # -------------------------------------------------------------
    # PHASE 3: Target Task Relearning with Dense Tracking (u2 updates)
    # -------------------------------------------------------------
    print(f"[{run_id}] Phase 3: Relearning Task A for {u2} updates with dense logging...")
    dynamics_tracker = RelearningDynamicsTracker(p_before=p_before)
    dense_freq = 5 if smoke_test else 100
    log_freq_p3 = 10 if smoke_test else 2500

    for step in range(1, u2 + 1):
        model.train()
        images, labels = next(loader_a_iter)
        images, labels = images.to(device), labels.to(device)

        optimizer_a.zero_grad()
        outputs = model(images, head='A')
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_a.step()

        if step % dense_freq == 0 or step == u2:
            val_acc = evaluate_accuracy(model, val_loader_a, device, head='A')
            dynamics_tracker.record_step(rel_update=step, val_acc=val_acc)
            if step % log_freq_p3 == 0:
                print(f"[{run_id} | Phase 3] Step {step}/{u2} - Val Acc: {val_acc:.2f}%")

    # Checkpoint theta_3 (End of Phase 3 / Final Model)
    theta3_path = os.path.join(checkpoint_dir, f"seed_{seed}_{condition_id}_theta3{suffix}.pt")
    torch.save(model.state_dict(), theta3_path)

    p_relearned = evaluate_accuracy(model, val_loader_a, device, head='A')
    p_relearned_test = evaluate_accuracy(model, test_loader_a, device, head='A')
    r_a = p_relearned - p_after_b

    probe_epochs = 1 if smoke_test else 5
    p_probe_theta3 = run_diagnostic_probe_at_theta2(
        model, train_loader_a, val_loader_a, device, num_classes_a=100, probe_epochs=probe_epochs
    )

    probe_feats_theta3 = extract_probe_features(model, probe_loader, device)
    eff_rank_theta3 = compute_effective_rank(probe_feats_theta3)
    dynamics_summary = dynamics_tracker.get_summary()

    # Save trajectory CSV
    traj_dir = os.path.join(results_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    traj_file = os.path.join(traj_dir, f"trajectory_{run_id}.csv")
    with open(traj_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rel_update", "val_acc"])
        for step, acc in dynamics_tracker.trajectory:
            writer.writerow([step, acc])

    print(f"[{run_id}] Final Checkpoint θ₃ locked.")
    print(f"       P_relearned (Val): {p_relearned:.2f}%, Test Acc: {p_relearned_test:.2f}%")
    print(f"       Recovery Gain R_A: {r_a:.2f}%, u_rec: {dynamics_summary['u_rec']} updates, AULC: {dynamics_summary['aulc']:.2f}")

    return {
        'run_id': run_id,
        'condition_id': condition_id,
        'seed': seed,
        'checkpoint_theta1': theta1_path,
        'checkpoint_theta2': theta2_path,
        'checkpoint_theta3': theta3_path,
        'p_theta1': p_before,
        'p_theta2': p_after_b,
        'f_b': f_b,
        'p_diagnostic_probe': p_diagnostic_probe,
        'p_probe_theta1': p_probe_theta1 if p_probe_theta1 is not None else meta.get('p_probe_theta1', 'N/A'),
        'p_probe_theta2': p_diagnostic_probe,
        'p_probe_theta3': p_probe_theta3,
        'p_theta3_val': p_relearned,
        'p_theta3_test': p_relearned_test,
        'p_control2_val': 'N/A',
        'r_a': r_a,
        's_g': s_g,
        'cka_drift': cka_drift,
        'eff_rank_theta1': eff_rank_theta1,
        'eff_rank_theta2': eff_rank_theta2,
        'eff_rank_theta3': eff_rank_theta3,
        **dynamics_summary
    }


def save_result_to_csv(result_dict, csv_path):
    """Appends a single run's dictionary to the main CSV result file"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    fieldnames = [
        'run_id', 'condition_id', 'seed',
        'p_theta1', 'p_theta2', 'f_b', 'p_diagnostic_probe',
        'p_probe_theta1', 'p_probe_theta2', 'p_probe_theta3',
        'p_theta3_val', 'p_theta3_test', 'p_control2_val', 'r_a',
        's_g', 'cka_drift',
        'eff_rank_theta1', 'eff_rank_theta2', 'eff_rank_theta3',
        'u_rec', 'aulc',
        'acc_horizon_500', 'acc_horizon_1000', 'acc_horizon_2000',
        'checkpoint_theta1', 'checkpoint_theta2', 'checkpoint_theta3'
    ]

    import fcntl
    with open(csv_path, 'a', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if not file_exists:
                writer.writeheader()
            writer.writerow(result_dict)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description="SOTLI Stage 1 Discovery Screening Runner")
    parser.add_argument("--conditions", nargs="+", default=["B0", "B1", "B3", "B4", "B5", "B7"],
                        help="List of condition IDs to run (default: B0 B1 B3 B4 B5 B7)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2026],
                        help="List of random seeds (default: 42 123 456 789 2026)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use: mps, cuda, or cpu (auto-detected by default)")
    parser.add_argument("--data_dir", type=str, default="./data_cache")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run high-speed smoke test (scaled-down updates) to verify pipeline end-to-end")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_default_device()
    print("=================================================================")
    print(" SOTLI STAGE 1 (GATE 1) DISCOVERY SCREENING RUNNER")
    print(f" Target Venue: TMLR")
    print(f" Execution Device: {device}")
    print(f" Conditions: {args.conditions}")
    print(f" Seeds: {args.seeds}")
    print(f" Smoke Test Mode: {args.smoke_test}")
    print("=================================================================")

    if args.smoke_test:
        u1 = 40
        u_b = 20
        u2 = 40
        results_csv = os.path.join(args.results_dir, "screening_results_smoke.csv")
    else:
        u1 = 15000
        u_b = 5000
        u2 = 15000
        results_csv = os.path.join(args.results_dir, "screening_results.csv")

    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4

    total_runs = len(args.conditions) * len(args.seeds)
    current_run = 0

    for seed in args.seeds:
        # Phase 1: Shared checkpoint theta_1
        theta1_path, opt1_path, p_before, eff_rank_theta1, p_probe_theta1 = train_phase1_if_needed(
            seed=seed, data_dir=args.data_dir, checkpoint_dir=args.checkpoint_dir,
            u1=u1, lr=lr, momentum=momentum, weight_decay=weight_decay,
            device=device, smoke_test=args.smoke_test
        )

        for cond in args.conditions:
            current_run += 1
            print(f"\n>>> [{current_run}/{total_runs}] Condition: {cond} | Seed: {seed} <<<")

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
            print(f"✓ Saved results for {cond} (Seed {seed}) -> {results_csv}")

    print("\n=================================================================")
    print(" ALL REQUESTED SCREENING RUNS FINISHED SUCCESSFULLY! 🎉")
    print(f" Results File: {results_csv}")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
