"""
Gate 2 Ablation Runner for SOTLI.
Executes the three pre-registered Gate 2 ablations for TMLR:
1. Schedule Timing Mini-Ablation (Early, Mid, Late insertion with Dual-Horizon evaluation).
2. Optimizer Momentum Buffer Ablation (Preserved vs. Zeroed momentum).
3. Replay Baseline (90% B + 10% A interleaved co-training during Phase 2).
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
    compute_linear_cka
)
from src.training.evaluator import RelearningDynamicsTracker, run_diagnostic_probe_at_theta2
from src.training.harness import (
    infinite_loader, stash_head_optimizer_state, restore_head_optimizer_state
)


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


def save_ablation_result(result_dict, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    fieldnames = list(result_dict.keys())
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_dict)


def get_or_train_phase1(seed, data_dir, checkpoint_dir, u1, lr, momentum, weight_decay, device, smoke_test=False, tag=""):
    """
    Loads Phase 1 checkpoint if it exists, otherwise trains Task A for u1 updates and caches it.
    Guarantees screening runs and all ablations branch from the EXACT same theta1 checkpoint (P1-A fix).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    suffix = "_smoke" if smoke_test else ""
    if tag:
        theta1_path = os.path.join(checkpoint_dir, f"seed_{seed}_{tag}_theta1{suffix}.pt")
        opt1_path = os.path.join(checkpoint_dir, f"seed_{seed}_{tag}_opt1{suffix}.pt")
        meta_path = os.path.join(checkpoint_dir, f"seed_{seed}_{tag}_meta{suffix}.pt")
    else:
        # Standard Gate 1 / screening naming (u1=15000): check checkpoint_dir then ./checkpoints
        theta1_path = os.path.join(checkpoint_dir, f"seed_{seed}_theta1{suffix}.pt")
        opt1_path = os.path.join(checkpoint_dir, f"seed_{seed}_opt_a_theta1{suffix}.pt")
        meta_path = os.path.join(checkpoint_dir, f"seed_{seed}_meta_theta1{suffix}.pt")
        if not (os.path.exists(theta1_path) and os.path.exists(opt1_path) and os.path.exists(meta_path)):
            alt_t = os.path.join("./checkpoints", f"seed_{seed}_theta1{suffix}.pt")
            alt_o = os.path.join("./checkpoints", f"seed_{seed}_opt_a_theta1{suffix}.pt")
            alt_m = os.path.join("./checkpoints", f"seed_{seed}_meta_theta1{suffix}.pt")
            if os.path.exists(alt_t) and os.path.exists(alt_o) and os.path.exists(alt_m):
                theta1_path, opt1_path, meta_path = alt_t, alt_o, alt_m

    if os.path.exists(theta1_path) and os.path.exists(opt1_path) and os.path.exists(meta_path):
        print(f"[Seed {seed}{f' | {tag}' if tag else ''}] Loaded shared Checkpoint θ₁: {theta1_path}")
        meta = torch.load(meta_path, map_location='cpu')
        p_theta1 = meta.get('p_theta1', meta.get('p_before', 0.0))
        return theta1_path, opt1_path, p_theta1

    print(f"\n[Seed {seed}{f' | {tag}' if tag else ''}] Training Phase 1 on Task A for {u1} updates...")
    set_seed(seed)
    train_loader_a, val_loader_a, _ = get_cifar100_loaders(data_dir=data_dir, batch_size=128, seed=seed)
    loader_a_iter = infinite_loader(train_loader_a)
    criterion = nn.CrossEntropyLoss()

    model = ResNet18GroupNorm(num_classes_a=100, num_groups=32).to(device)
    optimizer_a = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    log_freq = 10 if smoke_test else 2500
    for step in range(1, u1 + 1):
        model.train()
        imgs, labels = next(loader_a_iter)
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer_a.zero_grad()
        loss = criterion(model(imgs, head='A'), labels)
        loss.backward()
        optimizer_a.step()
        if step % log_freq == 0:
            val_acc = evaluate_accuracy(model, val_loader_a, device, head='A')
            print(f"[Seed {seed} | Phase 1] Step {step}/{u1} - Val Acc: {val_acc:.2f}%")

    p_theta1 = evaluate_accuracy(model, val_loader_a, device, head='A')
    torch.save(model.state_dict(), theta1_path)
    torch.save(optimizer_a.state_dict(), opt1_path)
    torch.save({'p_theta1': p_theta1, 'p_before': p_theta1}, meta_path)
    print(f"[Seed {seed}] Checkpoint θ₁ serialized. P_theta1: {p_theta1:.2f}%")
    return theta1_path, opt1_path, p_theta1


# =========================================================================
# 1. SCHEDULE TIMING MINI-ABLATION
# =========================================================================

def run_timing_ablation(condition_id, timing, seed, data_dir, checkpoint_dir, results_dir,
                        device, smoke_test=False):
    """
    Executes schedule timing ablation:
      timing in ['early', 'late']  (mid is already locked in Gate 1)
      - early: u1 = 5k (smoke: 20). Phase 3 runs for 25k (smoke: 40)
        -> Records acc at u2 = 15k (Fixed Recovery Horizon, U=25k total)
        -> Records acc at u2 = 25k (Fixed Total Budget, U=35k total)
      - late: u1 = 25k (smoke: 60). Phase 3 runs for 15k (smoke: 40)
        -> Records acc at u2 = 5k (Fixed Total Budget, U=35k total)
        -> Records acc at u2 = 15k (Fixed Recovery Horizon, U=45k total)
    """
    suffix = "_smoke" if smoke_test else ""
    run_id = f"timing_{timing}_{condition_id}_seed{seed}{suffix}"
    print(f"\n========================================================")
    print(f"[Gate 2 Timing] Running {condition_id} at Timing: {timing.upper()} | Seed: {seed}")
    print(f"========================================================")

    set_seed(seed)
    lr, momentum, weight_decay = 0.1, 0.9, 5e-4
    u_b = 20 if smoke_test else 5000

    if timing == 'early':
        u1 = 20 if smoke_test else 5000
        u2_max = 40 if smoke_test else 25000
        step_fixed_recovery = 20 if smoke_test else 15000
        step_fixed_total = 40 if smoke_test else 25000
    elif timing == 'late':
        u1 = 60 if smoke_test else 25000
        u2_max = 40 if smoke_test else 15000
        step_fixed_total = 10 if smoke_test else 5000
        step_fixed_recovery = 40 if smoke_test else 15000
    else:
        raise ValueError(f"Unknown timing: {timing}")

    train_loader_a, val_loader_a, test_loader_a = get_cifar100_loaders(data_dir=data_dir, batch_size=128, seed=seed)
    loader_a_iter = infinite_loader(train_loader_a)
    criterion = nn.CrossEntropyLoss()

    # P1-A fix: branch from shared Phase 1 checkpoint for this (seed, timing)
    theta1_path, opt1_path, p_theta1 = get_or_train_phase1(
        seed=seed, data_dir=data_dir, checkpoint_dir=checkpoint_dir,
        u1=u1, lr=lr, momentum=momentum, weight_decay=weight_decay,
        device=device, smoke_test=smoke_test, tag=f"timing_{timing}"
    )

    model = ResNet18GroupNorm(num_classes_a=100, num_groups=32).to(device)
    model.load_state_dict(torch.load(theta1_path, map_location=device))
    optimizer_a = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer_a.load_state_dict(torch.load(opt1_path, map_location=device))

    # ---------------- Phase 2: Auxiliary Interval ----------------
    head_a_param_ids = {id(p) for p in model.head_a.parameters()}
    stashed_opt_state = stash_head_optimizer_state(optimizer_a, head_a_param_ids)

    if condition_id == 'B1':
        model.init_auxiliary_head(num_classes_b=100)
    else:
        model.init_auxiliary_head(num_classes_b=10)

    model.freeze_head_a()
    model.to(device)

    aux_loader, _ = get_auxiliary_loader(condition_id, data_dir=data_dir, batch_size=128, seed=seed)
    aux_iter = infinite_loader(aux_loader)
    optimizer_b = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    print(f"[{run_id}] Phase 2: Training on {condition_id} for {u_b} updates...")
    for step in range(1, u_b + 1):
        model.train()
        imgs, labels = next(aux_iter)
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer_b.zero_grad()
        loss = criterion(model(imgs, head='B'), labels)
        loss.backward()
        optimizer_b.step()

    p_theta2_head_a = evaluate_accuracy(model, val_loader_a, device, head='A')
    f_b = p_theta1 - p_theta2_head_a
    print(f"[{run_id}] Checkpoint θ₂: Head A Acc = {p_theta2_head_a:.2f}% (Drop F_B: {f_b:.2f}%)")

    # ---------------- Phase 3: Relearning Task A ----------------
    model.discard_auxiliary_head()
    model.unfreeze_head_a()
    optimizer_a = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    restore_head_optimizer_state(optimizer_a, head_a_param_ids, stashed_opt_state)

    print(f"[{run_id}] Phase 3: Relearning Task A for {u2_max} updates...")
    acc_fixed_total = None
    acc_fixed_recovery = None
    test_fixed_total = None
    test_fixed_recovery = None
    tracker = RelearningDynamicsTracker(p_before=p_theta1)

    log_freq = 10 if smoke_test else 2500
    for step in range(1, u2_max + 1):
        model.train()
        imgs, labels = next(loader_a_iter)
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer_a.zero_grad()
        loss = criterion(model(imgs, head='A'), labels)
        loss.backward()
        optimizer_a.step()

        if step % (5 if smoke_test else 100) == 0 or step == u2_max:
            val_acc = evaluate_accuracy(model, val_loader_a, device, head='A')
            tracker.record_step(rel_update=step, val_acc=val_acc)
            if step % log_freq == 0:
                print(f"[{run_id} | Phase 3] Step {step}/{u2_max} - Val Acc: {val_acc:.2f}%")

        if step == step_fixed_total:
            acc_fixed_total = evaluate_accuracy(model, val_loader_a, device, head='A')
            test_fixed_total = evaluate_accuracy(model, test_loader_a, device, head='A')
        if step == step_fixed_recovery:
            acc_fixed_recovery = evaluate_accuracy(model, val_loader_a, device, head='A')
            test_fixed_recovery = evaluate_accuracy(model, test_loader_a, device, head='A')

    summary = tracker.get_summary()
    results_csv = os.path.join(results_dir, "timing_ablation_results.csv")
    result = {
        'run_id': run_id,
        'condition_id': condition_id,
        'timing': timing,
        'seed': seed,
        'u1': u1,
        'u_b': u_b,
        'p_theta1': p_theta1,
        'p_theta2': p_theta2_head_a,
        'f_b': f_b,
        'val_fixed_total_budget': acc_fixed_total,
        'val_fixed_recovery_horizon': acc_fixed_recovery,
        'test_fixed_total_budget': test_fixed_total,
        'test_fixed_recovery_horizon': test_fixed_recovery,
        'u_rec': summary['u_rec'],
        'aulc': summary['aulc'],
    }
    save_ablation_result(result, results_csv)
    print(f"✓ Saved timing results -> {results_csv}")
    return result


# =========================================================================
# 2. OPTIMIZER MOMENTUM BUFFER ABLATION
# =========================================================================

def run_momentum_ablation(condition_id, policy, seed, data_dir, checkpoint_dir, results_dir,
                          device, smoke_test=False):
    """
    Evaluates momentum buffer preservation policy across B -> A:
      policy in ['preserved', 'zeroed']
      - preserved: restores stashed momentum for Head A, retains backbone momentum.
      - zeroed: momentum buffer strictly reset to zero for all parameters upon returning to Task A.
    """
    suffix = "_smoke" if smoke_test else ""
    run_id = f"momentum_{policy}_{condition_id}_seed{seed}{suffix}"
    print(f"\n========================================================")
    print(f"[Gate 2 Momentum] Running {condition_id} | Policy: {policy.upper()} | Seed: {seed}")
    print(f"========================================================")

    set_seed(seed)
    lr, momentum, weight_decay = 0.1, 0.9, 5e-4
    u1 = 40 if smoke_test else 15000
    u_b = 20 if smoke_test else 5000
    u2 = 40 if smoke_test else 15000
    log_freq = 10 if smoke_test else 2500

    train_loader_a, val_loader_a, test_loader_a = get_cifar100_loaders(data_dir=data_dir, batch_size=128, seed=seed)
    loader_a_iter = infinite_loader(train_loader_a)
    criterion = nn.CrossEntropyLoss()

    # P1-A fix: branch from shared Gate 1 Phase 1 checkpoint
    theta1_path, opt1_path, p_theta1 = get_or_train_phase1(
        seed=seed, data_dir=data_dir, checkpoint_dir=checkpoint_dir,
        u1=u1, lr=lr, momentum=momentum, weight_decay=weight_decay,
        device=device, smoke_test=smoke_test
    )

    model = ResNet18GroupNorm(num_classes_a=100, num_groups=32).to(device)
    model.load_state_dict(torch.load(theta1_path, map_location=device))
    optimizer_a = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer_a.load_state_dict(torch.load(opt1_path, map_location=device))

    # Phase 2
    head_a_param_ids = {id(p) for p in model.head_a.parameters()}
    stashed_opt_state = stash_head_optimizer_state(optimizer_a, head_a_param_ids)

    if condition_id == 'B1':
        model.init_auxiliary_head(num_classes_b=100)
    else:
        model.init_auxiliary_head(num_classes_b=10)

    model.freeze_head_a()
    model.to(device)

    aux_loader, _ = get_auxiliary_loader(condition_id, data_dir=data_dir, batch_size=128, seed=seed)
    aux_iter = infinite_loader(aux_loader)
    optimizer_b = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    print(f"[{run_id}] Phase 2: Training on {condition_id} for {u_b} updates...")
    for step in range(1, u_b + 1):
        model.train()
        imgs, labels = next(aux_iter)
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer_b.zero_grad()
        loss = criterion(model(imgs, head='B'), labels)
        loss.backward()
        optimizer_b.step()
        if step % log_freq == 0:
            print(f"[{run_id} | Phase 2] Step {step}/{u_b} - Aux Loss: {loss.item():.4f}")

    p_theta2 = evaluate_accuracy(model, val_loader_a, device, head='A')
    f_b = p_theta1 - p_theta2
    print(f"[{run_id}] Checkpoint θ₂: Head A Acc = {p_theta2:.2f}% (Drop F_B: {f_b:.2f}%)")

    # Phase 3 transition
    model.discard_auxiliary_head()
    model.unfreeze_head_a()

    backbone_param_ids = {id(p) for name, p in model.named_parameters() if not name.startswith('head_')}
    optimizer_a = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    if policy == 'preserved':
        # P0-B fix: Carry forward backbone momentum from Phase 2 (optimizer_b)
        for group in optimizer_a.param_groups:
            for p in group['params']:
                if id(p) in backbone_param_ids and p in optimizer_b.state:
                    optimizer_a.state[p] = copy.deepcopy(optimizer_b.state[p])
        # Restore Head A momentum from Phase 1
        restore_head_optimizer_state(optimizer_a, head_a_param_ids, stashed_opt_state)
    elif policy == 'zeroed':
        # True zero baseline: both backbone and Head A start fresh with empty momentum buffers
        pass
    elif policy == 'preserved_head_only':
        # Historical Gate 1 behavior: only Head A momentum restored, backbone zeroed
        restore_head_optimizer_state(optimizer_a, head_a_param_ids, stashed_opt_state)
    else:
        raise ValueError(f"Unknown momentum policy: {policy}")

    print(f"[{run_id}] Phase 3: Relearning Task A (policy={policy}) for {u2} updates...")
    tracker = RelearningDynamicsTracker(p_before=p_theta1)
    dense_freq = 5 if smoke_test else 100

    for step in range(1, u2 + 1):
        model.train()
        imgs, labels = next(loader_a_iter)
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer_a.zero_grad()
        loss = criterion(model(imgs, head='A'), labels)
        loss.backward()
        optimizer_a.step()

        if step % dense_freq == 0 or step == u2:
            val_acc = evaluate_accuracy(model, val_loader_a, device, head='A')
            tracker.record_step(rel_update=step, val_acc=val_acc)
            if step % log_freq == 0 or step == u2:
                print(f"[{run_id} | Phase 3] Step {step}/{u2} - Val Acc: {val_acc:.2f}%")

    p_theta3 = evaluate_accuracy(model, val_loader_a, device, head='A')
    test_acc = evaluate_accuracy(model, test_loader_a, device, head='A')
    summary = tracker.get_summary()

    results_csv = os.path.join(results_dir, "momentum_ablation_results.csv")
    result = {
        'run_id': run_id,
        'condition_id': condition_id,
        'policy': policy,
        'seed': seed,
        'p_theta1': p_theta1,
        'p_theta2': p_theta2,
        'p_theta3_val': p_theta3,
        'p_theta3_test': test_acc,
        'u_rec': summary['u_rec'],
        'aulc': summary['aulc'],
    }
    save_ablation_result(result, results_csv)
    print(f"✓ Saved momentum results -> {results_csv}")
    return result


# =========================================================================
# 3. REPLAY BASELINE (90% B + 10% A CO-TRAINING)
# =========================================================================

def run_replay_baseline(condition_id, seed, data_dir, checkpoint_dir, results_dir,
                        device, smoke_test=False):
    """
    Executes Replay Baseline:
    During Phase 2 (5,000 updates), interleaves 90% auxiliary task B with 10% Task A updates.
    Tests whether temporal off-task isolation achieves an advantage over continuous joint regularization.
    """
    suffix = "_smoke" if smoke_test else ""
    run_id = f"replay_90_10_{condition_id}_seed{seed}{suffix}"
    print(f"\n========================================================")
    print(f"[Gate 2 Replay] Running 90% {condition_id} + 10% Task A Replay | Seed: {seed}")
    print(f"========================================================")

    set_seed(seed)
    lr, momentum, weight_decay = 0.1, 0.9, 5e-4
    u1 = 40 if smoke_test else 15000
    u_b = 20 if smoke_test else 5000
    u2 = 40 if smoke_test else 15000

    train_loader_a, val_loader_a, test_loader_a = get_cifar100_loaders(data_dir=data_dir, batch_size=128, seed=seed)
    loader_a_iter = infinite_loader(train_loader_a)
    criterion = nn.CrossEntropyLoss()

    # P1-A fix: branch from shared Gate 1 Phase 1 checkpoint
    theta1_path, opt1_path, p_theta1 = get_or_train_phase1(
        seed=seed, data_dir=data_dir, checkpoint_dir=checkpoint_dir,
        u1=u1, lr=lr, momentum=momentum, weight_decay=weight_decay,
        device=device, smoke_test=smoke_test
    )

    model = ResNet18GroupNorm(num_classes_a=100, num_groups=32).to(device)
    model.load_state_dict(torch.load(theta1_path, map_location=device))

    # Phase 2: 90% B + 10% A
    if condition_id == 'B1':
        model.init_auxiliary_head(num_classes_b=100)
    else:
        model.init_auxiliary_head(num_classes_b=10)
    model.to(device)

    aux_loader, _ = get_auxiliary_loader(condition_id, data_dir=data_dir, batch_size=128, seed=seed)
    aux_iter = infinite_loader(aux_loader)

    # Replay optimizer updates all parameters including both heads
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    print(f"[{run_id}] Phase 2: Training 90% {condition_id} + 10% A Replay for {u_b} updates...")
    log_freq = 10 if smoke_test else 2500
    for step in range(1, u_b + 1):
        model.train()
        optimizer.zero_grad()
        # 1 out of 10 steps trains on Task A (10% replay)
        if step % 10 == 0:
            imgs, labels = next(loader_a_iter)
            imgs, labels = imgs.to(device), labels.to(device)
            loss = criterion(model(imgs, head='A'), labels)
        else:
            imgs, labels = next(aux_iter)
            imgs, labels = imgs.to(device), labels.to(device)
            loss = criterion(model(imgs, head='B'), labels)
        loss.backward()
        optimizer.step()
        if step % log_freq == 0:
            print(f"[{run_id} | Phase 2] Step {step}/{u_b} - Loss: {loss.item():.4f}")

    p_theta2 = evaluate_accuracy(model, val_loader_a, device, head='A')
    f_b = p_theta1 - p_theta2
    print(f"[{run_id}] Checkpoint θ₂: Head A Acc = {p_theta2:.2f}% (Drop F_B: {f_b:.2f}%)")
    model.discard_auxiliary_head()

    # Phase 3
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    tracker = RelearningDynamicsTracker(p_before=p_theta1)
    dense_freq = 5 if smoke_test else 100
    print(f"[{run_id}] Phase 3: Relearning Task A for {u2} updates (Replay Baseline)...")

    for step in range(1, u2 + 1):
        model.train()
        imgs, labels = next(loader_a_iter)
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs, head='A'), labels)
        loss.backward()
        optimizer.step()

        if step % dense_freq == 0 or step == u2:
            val_acc = evaluate_accuracy(model, val_loader_a, device, head='A')
            tracker.record_step(rel_update=step, val_acc=val_acc)
            if step % log_freq == 0 or step == u2:
                print(f"[{run_id} | Phase 3] Step {step}/{u2} - Val Acc: {val_acc:.2f}%")

    p_theta3 = evaluate_accuracy(model, val_loader_a, device, head='A')
    test_acc = evaluate_accuracy(model, test_loader_a, device, head='A')
    summary = tracker.get_summary()

    results_csv = os.path.join(results_dir, "replay_ablation_results.csv")
    result = {
        'run_id': run_id,
        'condition_id': condition_id,
        'mode': 'replay_90_10',
        'seed': seed,
        'p_theta1': p_theta1,
        'p_theta2': p_theta2,
        'p_theta3_val': p_theta3,
        'p_theta3_test': test_acc,
        'u_rec': summary['u_rec'],
        'aulc': summary['aulc'],
    }
    save_ablation_result(result, results_csv)
    print(f"✓ Saved replay results -> {results_csv}")
    return result


def main():
    parser = argparse.ArgumentParser(description="SOTLI Gate 2 Ablation Runner")
    parser.add_argument("--experiment", type=str, choices=['timing', 'momentum', 'replay', 'all'],
                        default='all', help="Which ablation experiment to run")
    parser.add_argument("--conditions", nargs="+", default=['B4', 'B1'],
                        help="Conditions to test (default: B4 and B1)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456],
                        help="Random seeds to evaluate")
    parser.add_argument("--momentum_policies", nargs="+", default=['preserved', 'zeroed'],
                        choices=['preserved', 'zeroed', 'preserved_head_only'],
                        help="Optimizer momentum policies to evaluate")
    parser.add_argument("--data_dir", type=str, default="./data_cache")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--results_dir", type=str, default="./results/gate2")
    parser.add_argument("--smoke_test", action="store_true", help="High-speed verification mode")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_default_device()
    print("=================================================================")
    print(" SOTLI GATE 2 CONFOUND ELIMINATION & TIMING ABLATIONS")
    print(f" Target Venue: TMLR")
    print(f" Experiment: {args.experiment.upper()}")
    print(f" Conditions: {args.conditions}")
    print(f" Seeds: {args.seeds}")
    if args.experiment in ['momentum', 'all']:
        print(f" Momentum Policies: {args.momentum_policies}")
    print(f" Smoke Test: {args.smoke_test}")
    print(f" Execution Device: {device}")
    print("=================================================================")

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for seed in args.seeds:
        for cond in args.conditions:
            # 1. Timing Mini-Ablation
            if args.experiment in ['timing', 'all']:
                for timing in ['early', 'late']:
                    run_timing_ablation(
                        condition_id=cond, timing=timing, seed=seed,
                        data_dir=args.data_dir, checkpoint_dir=args.checkpoint_dir,
                        results_dir=args.results_dir, device=device, smoke_test=args.smoke_test
                    )

            # 2. Momentum Ablation
            if args.experiment in ['momentum', 'all']:
                for policy in args.momentum_policies:
                    run_momentum_ablation(
                        condition_id=cond, policy=policy, seed=seed,
                        data_dir=args.data_dir, checkpoint_dir=args.checkpoint_dir,
                        results_dir=args.results_dir, device=device, smoke_test=args.smoke_test
                    )

            # 3. Replay Baseline
            if args.experiment in ['replay', 'all']:
                run_replay_baseline(
                    condition_id=cond, seed=seed,
                    data_dir=args.data_dir, checkpoint_dir=args.checkpoint_dir,
                    results_dir=args.results_dir, device=device, smoke_test=args.smoke_test
                )

    print("\n=================================================================")
    print(" ALL REQUESTED GATE 2 ABLATIONS COMPLETED SUCCESSFULLY! 🎉")
    print(f" Results Saved In: {args.results_dir}")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
