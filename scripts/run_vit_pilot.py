"""
SOTLI Vision Transformer Pilot Protocol (P-ViT).
Evaluates 1 complete pilot seed (Seed 2026) across the Core Triad:
  - Phase 1: Shared baseline for 15,000 updates (Checkpoint theta_1)
  - Branch B0: Continuous Baseline (+20,000 updates on Task A)
  - Branch B1: Clean On-Task Reset (+5,000 updates Task A on temp head, +15,000 updates Task A relearn)
  - Branch B4: Auxiliary Transfer (+5,000 updates STL-10 on aux head, +15,000 updates Task A relearn)

Strictly enforces:
1. FP16 Automatic Mixed Precision (torch.amp.autocast) for maximal GPU efficiency.
2. Identical update allocations (u1=15k, u_b=5k, u2=15k, total=35k).
3. Matched phase transitions: stashing Head A optimizer state, zeroing backbone momentum, restoring Head A.
4. Comprehensive checkpointing: model, optimizer, grad scaler, RNG states (CPU, CUDA, NumPy, random).
5. Representation diagnostics: frozen linear probes and effective rank at theta_1, theta_2, and theta_3.
"""

import os
import sys
import copy
import time
import json
import random
import argparse
import numpy as np
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_states():
    return {
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    }


def set_rng_states(states):
    if not isinstance(states, dict):
        return
    try:
        if states.get('torch_cpu') is not None and isinstance(states['torch_cpu'], torch.Tensor):
            torch.set_rng_state(states['torch_cpu'].to(dtype=torch.uint8, device='cpu'))
    except Exception:
        pass
    try:
        if states.get('torch_cuda') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(states['torch_cuda'])
    except Exception:
        pass
    try:
        if states.get('numpy') is not None:
            np.random.set_state(states['numpy'])
    except Exception:
        pass
    try:
        if states.get('python') is not None:
            random.setstate(states['python'])
    except Exception:
        pass


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


def main():
    parser = argparse.ArgumentParser(description="SOTLI ViT Pilot Protocol")
    parser.add_argument("--seed", type=int, default=2026, help="Pilot random seed")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_dir", type=str, default="./data_cache")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/vit_pilot")
    parser.add_argument("--results_dir", type=str, default="./results/vit_pilot")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 80)
    print(" SOTLI VISION TRANSFORMER PILOT PROTOCOL (P-ViT)")
    print(f" Pilot Seed: {args.seed}")
    print(f" Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Precision: FP16 Automatic Mixed Precision (AMP)")
    print(f" Updates: Phase 1 (15k) -> Phase 2 (5k) -> Phase 3 (15k) [Cumulative: 35k]")
    print(f" Conditions to evaluate: B0 (Continuous), B1 (Clean Reset), B4 (STL-10 Transfer)")
    print(f" Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    t_pilot_start = time.time()
    set_seed(args.seed)

    # 1. Prepare Data Loaders
    print("\n[Data] Loading CIFAR-100 and Evaluation Partitions...")
    train_loader_a, val_loader_a, test_loader_a = get_cifar100_loaders(
        data_dir=args.data_dir, batch_size=128, seed=args.seed
    )
    probe_loader = get_fixed_probe_set(data_dir=args.data_dir, num_samples=2000, seed=42)
    criterion = nn.CrossEntropyLoss()

    theta1_path = os.path.join(args.checkpoint_dir, f"vit_seed{args.seed}_theta1.pt")
    opt1_path = os.path.join(args.checkpoint_dir, f"vit_seed{args.seed}_opt1.pt")

    if os.path.exists(theta1_path) and os.path.exists(opt1_path):
        print("\n" + "=" * 80)
        print(f" FOUND EXISTING CHECKPOINT θ₁: {theta1_path}")
        print(" Reusing existing Phase 1 baseline metrics and proceeding directly to branches...")
        print("=" * 80)
        ckpt = torch.load(theta1_path, map_location=device, weights_only=False)
        p_theta1_val = ckpt['val_acc']
        p_theta1_test = ckpt['test_acc']
        p_probe_theta1 = ckpt['probe_acc']
        eff_rank_theta1 = ckpt['eff_rank']
        print(f"  θ₁ Validation Acc: {p_theta1_val:.2f}% | Test Acc: {p_theta1_test:.2f}%")
        print(f"  Linear Probe     : {p_probe_theta1:.2f}% | Effective Rank: {eff_rank_theta1:.2f}")
    else:
        # -------------------------------------------------------------------------
        # PHASE 1: Train Shared Baseline Model for 15,000 Updates
        # -------------------------------------------------------------------------
        print("\n" + "=" * 80)
        print(" PHASE 1: Training Shared ViT Baseline on CIFAR-100 (15,000 updates)")
        print("=" * 80)
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
                speed = step / elapsed
                print(f"[Phase 1] Step {step}/15000 | Loss: {loss.item():.4f} | Val Acc: {val_acc:.2f}% | Speed: {speed:.1f} steps/s")

        # Evaluate and Lock Checkpoint theta_1
        p_theta1_val = evaluate_accuracy_amp(model, val_loader_a, device, head='A')
        p_theta1_test = evaluate_accuracy_amp(model, test_loader_a, device, head='A')
        p_probe_theta1 = run_linear_probe_amp(model, train_loader_a, val_loader_a, device)
        probe_feats_theta1 = extract_probe_features(model, probe_loader, device)
        eff_rank_theta1 = compute_effective_rank(probe_feats_theta1)

        t_p1_end = time.time()
        print(f"\n✓ Checkpoint θ₁ locked in {t_p1_end - t_p1_start:.1f}s (~{(t_p1_end - t_p1_start)/60:.1f} min)")
        print(f"  Validation Acc: {p_theta1_val:.2f}% | Test Acc: {p_theta1_test:.2f}%")
        print(f"  Linear Probe  : {p_probe_theta1:.2f}% | Effective Rank: {eff_rank_theta1:.2f}")

        torch.save({
            'model_state': model.state_dict(),
            'rng_states': get_rng_states(),
            'val_acc': p_theta1_val,
            'test_acc': p_theta1_test,
            'probe_acc': p_probe_theta1,
            'eff_rank': eff_rank_theta1,
        }, theta1_path)
        torch.save({
            'optimizer_state': optimizer.state_dict(),
            'scaler_state': scaler.state_dict(),
        }, opt1_path)

    results_dict = {
        'seed': args.seed,
        'p_theta1_val': p_theta1_val,
        'p_theta1_test': p_theta1_test,
        'p_probe_theta1': p_probe_theta1,
        'eff_rank_theta1': eff_rank_theta1,
    }

    # Helper function to run a 20k-update branch
    def run_branch(condition_id):
        print("\n" + "=" * 80)
        print(f" BRANCH: Condition {condition_id} (Seed {args.seed})")
        print("=" * 80)
        t_b_start = time.time()

        # Reload clean theta_1 checkpoint
        branch_model = CompactViT(img_size=32, patch_size=4, num_classes_a=100, embed_dim=192, depth=6, num_heads=4).to(device)
        ckpt = torch.load(theta1_path, map_location=device, weights_only=False)
        branch_model.load_state_dict(ckpt['model_state'])
        set_rng_states(ckpt['rng_states'])

        if condition_id == 'B0':
            # -------------------------------------------------------------
            # B0: Continuous Uninterrupted Training (+20,000 updates)
            # -------------------------------------------------------------
            opt_data = torch.load(opt1_path, map_location=device, weights_only=False)
            branch_opt = optim.AdamW(branch_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            branch_opt.load_state_dict(opt_data['optimizer_state'])
            branch_scaler = torch.amp.GradScaler('cuda')
            branch_scaler.load_state_dict(opt_data['scaler_state'])
            loader_iter = infinite_loader(train_loader_a)

            p_control2_val = None
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

                if step == 15000:  # Cumulative 30,000 updates (Control 2: Equal Task A updates)
                    p_control2_val = evaluate_accuracy_amp(branch_model, val_loader_a, device, head='A')
                    print(f"[B0] Step {step} (+15k -> 30k total): Control 2 Val Acc = {p_control2_val:.2f}%")

                if step % 2500 == 0 or step == 20000:
                    val_acc = evaluate_accuracy_amp(branch_model, val_loader_a, device, head='A')
                    elapsed = time.time() - t_b_start
                    print(f"[B0] Step {step}/20000 | Loss: {loss.item():.4f} | Val Acc: {val_acc:.2f}% | Elapsed: {elapsed/60:.1f}m")

            val_final = evaluate_accuracy_amp(branch_model, val_loader_a, device, head='A')
            test_final = evaluate_accuracy_amp(branch_model, test_loader_a, device, head='A')
            probe_final = run_linear_probe_amp(branch_model, train_loader_a, val_loader_a, device)
            probe_feats_final = extract_probe_features(branch_model, probe_loader, device)
            eff_rank_final = compute_effective_rank(probe_feats_final)

            return {
                'p_theta2_val': p_theta1_val,
                'p_probe_theta2': p_probe_theta1,
                'eff_rank_theta2': eff_rank_theta1,
                'p_control2_val': p_control2_val,
                'p_theta3_val': val_final,
                'p_theta3_test': test_final,
                'p_probe_theta3': probe_final,
                'eff_rank_theta3': eff_rank_final,
            }

        else:
            # -------------------------------------------------------------
            # SOTLI Schedule (B1 or B4): Phase 2 (5k) -> Phase 3 (15k)
            # -------------------------------------------------------------
            opt_data = torch.load(opt1_path, map_location=device, weights_only=False)
            opt_a = optim.AdamW(branch_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            opt_a.load_state_dict(opt_data['optimizer_state'])

            # Transition A -> B: Stash Head A parameters and optimizer state
            head_a_param_ids = {id(p) for p in branch_model.head_a.parameters()}
            stashed_opt_state = stash_head_optimizer_state(opt_a, head_a_param_ids)
            branch_model.freeze_head_a()

            # Prepare Phase 2 Auxiliary Loader & Head B
            if condition_id == 'B1':
                aux_loader_b, num_classes_b = train_loader_a, 100  # Clean on-task reset
            elif condition_id == 'B4':
                aux_loader_b, num_classes_b = get_auxiliary_loader('B4', data_dir=args.data_dir, batch_size=128, seed=args.seed)
            branch_model.init_auxiliary_head(num_classes_b, device=device)

            # Phase 2 Optimizer: Fresh AdamW on active parameters
            opt_b = optim.AdamW([p for p in branch_model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
            scaler_b = torch.amp.GradScaler('cuda')
            loader_b_iter = infinite_loader(aux_loader_b)

            print(f"[{condition_id}] Starting Phase 2 ({5000} updates)...")
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

                if step % 2500 == 0 or step == 5000:
                    print(f"[{condition_id} | Phase 2] Step {step}/5000 | Loss: {loss.item():.4f}")

            # Diagnostics at theta_2
            p_theta2_val = evaluate_accuracy_amp(branch_model, val_loader_a, device, head='A')
            p_probe_theta2 = run_linear_probe_amp(branch_model, train_loader_a, val_loader_a, device)
            probe_feats_theta2 = extract_probe_features(branch_model, probe_loader, device)
            eff_rank_theta2 = compute_effective_rank(probe_feats_theta2)
            print(f"[{condition_id}] Checkpoint θ₂ locked: Val Acc (Frozen Head A) = {p_theta2_val:.2f}% | Linear Probe = {p_probe_theta2:.2f}% | EffRank = {eff_rank_theta2:.2f}")

            # Transition B -> A: Discard Head B, unfreeze Head A, clear backbone momentum, restore Head A state
            branch_model.discard_auxiliary_head()
            branch_model.unfreeze_head_a()
            opt_p3 = optim.AdamW(branch_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            restore_head_optimizer_state(opt_p3, head_a_param_ids, stashed_opt_state)
            scaler_p3 = torch.amp.GradScaler('cuda')
            loader_a_iter = infinite_loader(train_loader_a)

            print(f"[{condition_id}] Starting Phase 3 Relearning ({15000} updates)...")
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
                    elapsed = time.time() - t_b_start
                    print(f"[{condition_id} | Phase 3] Step {step}/15000 | Loss: {loss.item():.4f} | Val Acc: {val_acc:.2f}% | Elapsed: {elapsed/60:.1f}m")

            val_final = evaluate_accuracy_amp(branch_model, val_loader_a, device, head='A')
            test_final = evaluate_accuracy_amp(branch_model, test_loader_a, device, head='A')
            probe_final = run_linear_probe_amp(branch_model, train_loader_a, val_loader_a, device)
            probe_feats_final = extract_probe_features(branch_model, probe_loader, device)
            eff_rank_final = compute_effective_rank(probe_feats_final)

            return {
                'p_theta2_val': p_theta2_val,
                'p_probe_theta2': p_probe_theta2,
                'eff_rank_theta2': eff_rank_theta2,
                'p_control2_val': 'N/A',
                'p_theta3_val': val_final,
                'p_theta3_test': test_final,
                'p_probe_theta3': probe_final,
                'eff_rank_theta3': eff_rank_final,
            }

    # Execute all 3 branches
    results_b0 = run_branch('B0')
    results_b1 = run_branch('B1')
    results_b4 = run_branch('B4')

    t_pilot_end = time.time()
    total_pilot_mins = (t_pilot_end - t_pilot_start) / 60.0

    # -------------------------------------------------------------------------
    # PILOT SUMMARY & HYPOTHESIS TESTING
    # -------------------------------------------------------------------------
    delta_b4_vs_b1_val = results_b4['p_theta3_val'] - results_b1['p_theta3_val']
    delta_b4_vs_b1_test = results_b4['p_theta3_test'] - results_b1['p_theta3_test']
    delta_total_b4 = results_b4['p_theta3_val'] - results_b0['p_theta3_val']
    delta_total_b1 = results_b1['p_theta3_val'] - results_b0['p_theta3_val']
    delta_a_b4 = results_b4['p_theta3_val'] - results_b0['p_control2_val']
    delta_a_b1 = results_b1['p_theta3_val'] - results_b0['p_control2_val']

    print("\n" + "=" * 80)
    print(" SOTLI VISION TRANSFORMER PILOT EXPERIMENT COMPLETE!")
    print(f" Total Runtime: {total_pilot_mins:.1f} minutes")
    print("=" * 80)
    print("\n[Empirical Validation Metrics]")
    print(f"  θ₁ Pre-Interval Baseline    : Val = {p_theta1_val:.2f}% | Test = {p_theta1_test:.2f}% | Probe = {p_probe_theta1:.2f}%")
    print(f"  B0 Continuous Final (35k)  : Val = {results_b0['p_theta3_val']:.2f}% | Test = {results_b0['p_theta3_test']:.2f}% (Control 2 @30k: {results_b0['p_control2_val']:.2f}%)")
    print(f"  B1 Clean Reset Final (35k) : Val = {results_b1['p_theta3_val']:.2f}% | Test = {results_b1['p_theta3_test']:.2f}% (θ₂ Probe: {results_b1['p_probe_theta2']:.2f}%)")
    print(f"  B4 STL-10 Transfer Final   : Val = {results_b4['p_theta3_val']:.2f}% | Test = {results_b4['p_theta3_test']:.2f}% (θ₂ Probe: {results_b4['p_probe_theta2']:.2f}%)")
    print("\n" + "-" * 80)
    print(" CORE HYPOTHESIS TESTS ON ViT:")
    print(f"  1. Decisive Paired Contrast Δ(B4 - B1): Val = {delta_b4_vs_b1_val:+.2f} pp | Test = {delta_b4_vs_b1_test:+.2f} pp")
    print(f"  2. Equal Target Updates Δ_A (vs 30k)  : B4 = {delta_a_b4:+.2f} pp | B1 = {delta_a_b1:+.2f} pp")
    print(f"  3. Equal Total Updates Δ_total (vs 35k): B4 = {delta_total_b4:+.2f} pp | B1 = {delta_total_b1:+.2f} pp")
    print("-" * 80 + "\n")

    summary_file = os.path.join(args.results_dir, "vit_pilot_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'seed': args.seed,
            'total_runtime_minutes': total_pilot_mins,
            'p_theta1_val': p_theta1_val,
            'p_theta1_test': p_theta1_test,
            'p_probe_theta1': p_probe_theta1,
            'eff_rank_theta1': eff_rank_theta1,
            'b0': results_b0,
            'b1': results_b1,
            'b4': results_b4,
            'deltas': {
                'delta_b4_vs_b1_val': delta_b4_vs_b1_val,
                'delta_b4_vs_b1_test': delta_b4_vs_b1_test,
                'delta_total_b4': delta_total_b4,
                'delta_total_b1': delta_total_b1,
                'delta_a_b4': delta_a_b4,
                'delta_a_b1': delta_a_b1,
            }
        }, f, indent=2)
    print(f"✓ Saved complete pilot metrics to {summary_file}")


if __name__ == "__main__":
    main()
