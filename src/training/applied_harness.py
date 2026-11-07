"""
Training harness for the Applied Fine-Tuning Regime (Scenario B: ISIC 2019 + Imagenette).

Specifications (from MVA_12K_FINETUNE_PLAN.md):
- 12,000 updates total:
  * Phase 1 (u1 = 5,000 updates): Target Task A (ISIC 2019, 8 classes, Head A)
  * Phase 2 (uB = 2,000 updates): Arm-dependent interval
    - C0: Continuous on Task A (Head A)
    - C3: On-target reset control (Task A with temporary 8-class head + momentum zero)
    - S1: Auxiliary detour (Task B Imagenette, 10-class head + momentum zero)
    - S1-shuf: Task B with label-shuffled Imagenette
    - Factorial: F00, F01, F10, F11
  * Phase 3 (u2 = 5,000 updates): Target Task A relearning with restored Head A
- Optimization:
  * SGD, momentum 0.9, weight decay 1e-4, batch size 64
  * 500-step linear warmup applied at step 1 for all arms, then constant LR 0.01
  * PyTorch Automatic Mixed Precision (AMP)
  * BatchNorm running statistics preserved across boundaries
- Diagnostics recorded at theta1 (5k), theta2 (7k), theta3 (12k), and 10k (equal-target comparator):
  * Top-1 Accuracy, Macro one-vs-rest AUC, Balanced Accuracy
  * Expected Calibration Error (ECE) and Brier score
  * Effective Representation Rank (EffRank) on centered 2,000-image probe
  * Linear CKA representation drift vs theta1 on probe
"""

import os
import copy
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.utils.metrics import (
    evaluate_accuracy, evaluate_applied_metrics,
    extract_probe_features, compute_effective_rank, compute_linear_cka,
    train_and_eval_linear_probe
)


def infinite_loader(dataloader):
    """Yields batches infinitely from dataloader across epochs."""
    while True:
        for batch in dataloader:
            yield batch


def set_learning_rate(optimizer, lr):
    """Sets learning rate for all parameter groups in optimizer."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_warmup_lr(step, warmup_steps=500, base_lr=0.01):
    """Linear warmup from 0.0 to base_lr over warmup_steps, then constant base_lr."""
    if step < warmup_steps:
        return base_lr * (float(step) / float(max(1, warmup_steps)))
    return base_lr


def stash_head_optimizer_state(optimizer, head_param_ids):
    """Extracts and stashes optimizer momentum buffers belonging to Head A parameters."""
    stashed_state = {}
    for group in optimizer.param_groups:
        for p in group['params']:
            if id(p) in head_param_ids and p in optimizer.state:
                stashed_state[id(p)] = copy.deepcopy(optimizer.state[p])
                del optimizer.state[p]
    return stashed_state


def restore_head_optimizer_state(optimizer, head_param_ids, stashed_state):
    """Restores optimizer momentum buffers for Head A parameters upon returning to Task A."""
    for group in optimizer.param_groups:
        for p in group['params']:
            if id(p) in head_param_ids and id(p) in stashed_state:
                optimizer.state[p] = stashed_state[id(p)]


def zero_momentum_buffers(optimizer):
    """Zeros out momentum buffers across all parameters in optimizer."""
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state.get(p, None)
            if state is not None and 'momentum_buffer' in state:
                state['momentum_buffer'].zero_()


def run_applied_finetune_experiment(
    model,
    train_loader_a,
    val_loader_a,
    test_loader_a,
    probe_loader,
    aux_loader=None,
    condition='C0',
    seed=42,
    u1=5000,
    uB=2000,
    u2=5000,
    base_lr=0.01,
    warmup_steps=500,
    momentum=0.9,
    weight_decay=1e-4,
    device='cuda',
    checkpoint_dir='./checkpoints/applied',
    eval_probe=True,
    log_interval=500
):
    """
    Executes an applied fine-tuning run for condition under exact update budgeting.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device(device)
    model.to(device)

    total_updates = u1 + uB + u2
    boundary_1 = u1
    boundary_2 = u1 + uB
    equal_target_step = u1 + u2  # 10,000 for equal target exposure

    # Arm boundary configurations:
    # head_swap: whether Head A is replaced during Phase 2
    # momentum_reset: whether momentum buffers are zeroed at boundaries
    condition_configs = {
        'C0': {'head_swap': False, 'momentum_reset': False, 'phase2_task': 'A'},
        'C3': {'head_swap': True,  'momentum_reset': True,  'phase2_task': 'A_temp'},
        'S1': {'head_swap': True,  'momentum_reset': True,  'phase2_task': 'B'},
        'S1-shuf': {'head_swap': True, 'momentum_reset': True, 'phase2_task': 'B'},
        'F00': {'head_swap': False, 'momentum_reset': False, 'phase2_task': 'A'},
        'F01': {'head_swap': True,  'momentum_reset': False, 'phase2_task': 'A_temp'},
        'F10': {'head_swap': False, 'momentum_reset': True,  'phase2_task': 'A'},
        'F11': {'head_swap': True,  'momentum_reset': True,  'phase2_task': 'A_temp'},
    }

    if condition not in condition_configs:
        raise ValueError(f"Unknown condition: {condition}. Supported: {list(condition_configs.keys())}")

    cfg = condition_configs[condition]
    do_head_swap = cfg['head_swap']
    do_momentum_reset = cfg['momentum_reset']
    phase2_task = cfg['phase2_task']

    if phase2_task == 'B' and aux_loader is None:
        raise ValueError(f"Condition {condition} requires aux_loader (Task B) but None was passed.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0, momentum=momentum, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    iter_a = infinite_loader(train_loader_a)
    iter_b = infinite_loader(aux_loader) if aux_loader is not None else None

    # Stashed state for Head A
    stashed_head_opt_state = None
    head_a_param_ids = {id(p) for p in model.head_a.parameters()}

    run_metrics = {
        "condition": condition,
        "seed": seed,
        "total_updates": total_updates,
        "checkpoints": {},
        "step_times": [],
    }

    theta1_probe_features = None
    start_wall_time = time.time()

    print(f"\n========================================================")
    print(f"[{condition} | Seed {seed}] Starting Applied Fine-Tuning Run ({total_updates} updates)")
    print(f"Phase 1: 1..{boundary_1} | Phase 2: {boundary_1 + 1}..{boundary_2} | Phase 3: {boundary_2 + 1}..{total_updates}")
    print(f"Device: {device} | AMP: {scaler.is_enabled()} | Head Swap: {do_head_swap} | Momentum Reset: {do_momentum_reset}")
    print(f"========================================================\n")

    current_head = 'A'

    for step in range(1, total_updates + 1):
        step_start = time.time()
        model.train()

        # Update learning rate (warmup applied once from step 1..warmup_steps)
        current_lr = get_warmup_lr(step, warmup_steps=warmup_steps, base_lr=base_lr)
        set_learning_rate(optimizer, current_lr)

        # ----------------------------------------------------
        # Boundary 1 Transition (End of Phase 1 -> Start of Phase 2)
        # ----------------------------------------------------
        if step == boundary_1 + 1:
            print(f"\n--- [{condition} | Seed {seed}] Transitioning to Phase 2 at step {step} ---")
            if do_head_swap:
                model.freeze_head_a()
                stashed_head_opt_state = stash_head_optimizer_state(optimizer, head_a_param_ids)
                if phase2_task == 'B':
                    model.init_auxiliary_head(num_classes_b=10, device=device)
                    current_head = 'B'
                    optimizer.add_param_group({'params': model.head_b.parameters()})
                    print(f"[{condition}] Stashed Head A. Initialized Head B (10 classes).")
                else:
                    model.init_temp_head_a(device=device)
                    current_head = 'temp'
                    optimizer.add_param_group({'params': model.head_temp.parameters()})
                    print(f"[{condition}] Stashed Head A. Initialized temporary Head A ({model.num_classes_a} classes).")

            if do_momentum_reset:
                zero_momentum_buffers(optimizer)
                print(f"[{condition}] Zeroed momentum buffers at Boundary 1.")

        # ----------------------------------------------------
        # Boundary 2 Transition (End of Phase 2 -> Start of Phase 3)
        # ----------------------------------------------------
        if step == boundary_2 + 1:
            print(f"\n--- [{condition} | Seed {seed}] Transitioning to Phase 3 at step {step} ---")
            if do_head_swap:
                # Remove temporary head params from optimizer
                temp_head = model.head_b if model.head_b is not None else model.head_temp
                temp_param_ids = {id(p) for p in temp_head.parameters()}
                optimizer.param_groups = [g for g in optimizer.param_groups if not any(id(p) in temp_param_ids for p in g['params'])]
                
                model.discard_auxiliary_head()
                model.unfreeze_head_a()
                restore_head_optimizer_state(optimizer, head_a_param_ids, stashed_head_opt_state)
                current_head = 'A'
                print(f"[{condition}] Discarded temporary head. Restored Head A and its momentum state.")

            if do_momentum_reset:
                zero_momentum_buffers(optimizer)
                print(f"[{condition}] Zeroed momentum buffers at Boundary 2.")

        # ----------------------------------------------------
        # Fetch Batch based on active phase
        # ----------------------------------------------------
        if boundary_1 < step <= boundary_2 and phase2_task == 'B':
            images, labels = next(iter_b)
        else:
            images, labels = next(iter_a)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            outputs = model(images, head=current_head)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        step_elapsed = time.time() - step_start
        run_metrics["step_times"].append(step_elapsed)

        # ----------------------------------------------------
        # Milestone Checkpoint 1 (theta1 at step u1)
        # ----------------------------------------------------
        if step == boundary_1:
            ckpt_path = os.path.join(checkpoint_dir, f"{condition}_seed{seed}_theta1.pt")
            torch.save(model.state_dict(), ckpt_path)
            val_m = evaluate_applied_metrics(model, val_loader_a, device, num_classes=8, head='A')
            print(f"[{condition} | theta1 @ step {step}] Val Top-1: {val_m['top1_accuracy']:.2f}%, Macro AUC: {val_m['macro_auc']:.4f}, Bal Acc: {val_m['balanced_accuracy']:.2f}%")
            
            diag = {}
            if eval_probe and probe_loader is not None:
                theta1_probe_features = extract_probe_features(model, probe_loader, device)
                diag["eff_rank"] = compute_effective_rank(theta1_probe_features)
                diag["cka_drift"] = 0.0
                print(f"[{condition} | theta1] EffRank: {diag['eff_rank']:.2f}")

            run_metrics["checkpoints"]["theta1"] = {
                "step": step,
                "metrics": val_m,
                "diagnostics": diag,
                "checkpoint": ckpt_path,
            }

        # ----------------------------------------------------
        # Milestone Checkpoint 2 (theta2 at step u1 + uB)
        # ----------------------------------------------------
        if step == boundary_2:
            ckpt_path = os.path.join(checkpoint_dir, f"{condition}_seed{seed}_theta2.pt")
            torch.save(model.state_dict(), ckpt_path)
            diag = {}
            if eval_probe and probe_loader is not None and theta1_probe_features is not None:
                feat2 = extract_probe_features(model, probe_loader, device)
                diag["eff_rank"] = compute_effective_rank(feat2)
                diag["cka_drift"] = 1.0 - compute_linear_cka(theta1_probe_features, feat2)
                print(f"[{condition} | theta2 @ step {step}] EffRank: {diag['eff_rank']:.2f}, CKA Drift: {diag['cka_drift']:.4f}")

            run_metrics["checkpoints"]["theta2"] = {
                "step": step,
                "diagnostics": diag,
                "checkpoint": ckpt_path,
            }

        # ----------------------------------------------------
        # Equal-Target Comparator (Step u1 + u2 = 10,000 updates on Task A)
        # ----------------------------------------------------
        if step == equal_target_step and condition == 'C0':
            ckpt_path = os.path.join(checkpoint_dir, f"{condition}_seed{seed}_theta_10k_equal_target.pt")
            torch.save(model.state_dict(), ckpt_path)
            val_10k = evaluate_applied_metrics(model, val_loader_a, device, num_classes=8, head='A')
            print(f"[{condition} | Equal-Target @ step {step}] Val Top-1: {val_10k['top1_accuracy']:.2f}%, Macro AUC: {val_10k['macro_auc']:.4f}")
            run_metrics["checkpoints"]["theta_10k"] = {
                "step": step,
                "metrics": val_10k,
                "checkpoint": ckpt_path,
            }

        # Periodic progress logging
        if step % log_interval == 0:
            avg_step_ms = (sum(run_metrics["step_times"][-log_interval:]) / log_interval) * 1000.0
            print(f"[{condition} | Seed {seed}] Step {step}/{total_updates} | Loss: {loss.item():.4f} | LR: {current_lr:.5f} | Head: {current_head} | Step Time: {avg_step_ms:.1f}ms")

    # ----------------------------------------------------
    # Terminal Evaluation (theta3 at total_updates)
    # ----------------------------------------------------
    ckpt_path = os.path.join(checkpoint_dir, f"{condition}_seed{seed}_theta3.pt")
    torch.save(model.state_dict(), ckpt_path)

    val_terminal = evaluate_applied_metrics(model, val_loader_a, device, num_classes=8, head='A')
    diag = {}
    if eval_probe and probe_loader is not None and theta1_probe_features is not None:
        feat3 = extract_probe_features(model, probe_loader, device)
        diag["eff_rank"] = compute_effective_rank(feat3)
        diag["cka_drift"] = 1.0 - compute_linear_cka(theta1_probe_features, feat3)

    total_wall_time = time.time() - start_wall_time
    avg_step_time_ms = (sum(run_metrics["step_times"]) / len(run_metrics["step_times"])) * 1000.0

    print(f"\n========================================================")
    print(f"[{condition} | Seed {seed}] COMPLETED in {total_wall_time:.1f}s (Avg step: {avg_step_time_ms:.1f}ms)")
    print(f"Terminal Val Top-1: {val_terminal['top1_accuracy']:.2f}% | Macro AUC: {val_terminal['macro_auc']:.4f} | Bal Acc: {val_terminal['balanced_accuracy']:.2f}% | ECE: {val_terminal['ece']:.4f}")
    if diag:
        print(f"EffRank: {diag.get('eff_rank', 0.0):.2f} | CKA Drift vs theta1: {diag.get('cka_drift', 0.0):.4f}")
    print(f"========================================================\n")

    run_metrics["checkpoints"]["theta3"] = {
        "step": total_updates,
        "metrics": val_terminal,
        "diagnostics": diag,
        "checkpoint": ckpt_path,
    }
    run_metrics["total_wall_time_seconds"] = total_wall_time
    run_metrics["avg_step_time_ms"] = avg_step_time_ms

    # Save run record
    run_json_path = os.path.join(checkpoint_dir, f"{condition}_seed{seed}_summary.json")
    with open(run_json_path, "w") as f:
        json.dump(run_metrics, f, indent=2)

    return run_metrics
