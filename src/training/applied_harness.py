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
- Circuit Breakers:
  * Halts immediately on numerical divergence (loss > 5.0, NaN, Inf)
  * Halts on severe class collapse (balanced accuracy < 10.0% post-warmup)
- Diagnostics & Live Dashboard:
  * Real-time training & validation tracking exported to CSV every 250 updates
  * Automated 3-panel publication-grade trajectory plotting
"""

import os
import sys
import copy
import time
import json
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.utils.metrics import (
    evaluate_accuracy, evaluate_applied_metrics,
    extract_probe_features, compute_effective_rank, compute_linear_cka,
    train_and_eval_linear_probe
)

try:
    from scripts.plot_live_applied_progress import plot_applied_trajectory
except Exception:
    try:
        from plot_live_applied_progress import plot_applied_trajectory
    except Exception:
        plot_applied_trajectory = None


class CircuitBreakerError(RuntimeError):
    """Raised when training dynamics diverge or degenerate to save GPU compute."""
    pass


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
    log_interval=250
):
    """
    Executes an applied fine-tuning run for condition under exact update budgeting.
    Monitors convergence via live CSV logging and triggers circuit breaker on divergence.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device(device)
    model.to(device)

    total_updates = u1 + uB + u2
    boundary_1 = u1
    boundary_2 = u1 + uB
    equal_target_step = u1 + u2  # 10,000 for equal target exposure

    # Arm boundary configurations:
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

    # Initialize live trajectory CSV
    traj_csv_path = os.path.join(checkpoint_dir, "training_trajectory.csv")
    dashboard_png_path = os.path.join(checkpoint_dir, "training_trajectory_dashboard.png")
    traj_fields = ["step", "train_loss", "val_loss", "val_top1", "val_bal_acc", "val_macro_auc", "eff_rank", "lr", "elapsed_s"]

    with open(traj_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=traj_fields)
        writer.writeheader()

    theta1_probe_features = None
    start_wall_time = time.time()
    interval_train_loss = 0.0
    interval_steps = 0

    print(f"\n========================================================")
    print(f"[{condition} | Seed {seed}] Starting Applied Fine-Tuning Run ({total_updates} updates)")
    print(f"Phase 1: 1..{boundary_1} | Phase 2: {boundary_1 + 1}..{boundary_2} | Phase 3: {boundary_2 + 1}..{total_updates}")
    print(f"Device: {device} | AMP: {scaler.is_enabled()} | Head Swap: {do_head_swap} | Momentum Reset: {do_momentum_reset}")
    print(f"Trajectory CSV: {traj_csv_path}")
    print(f"Live Dashboard: {dashboard_png_path}")
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

        # Circuit Breaker 1: Numerical explosion check
        loss_val = float(loss.item())
        if torch.isnan(loss) or torch.isinf(loss) or loss_val > 5.0:
            raise CircuitBreakerError(
                f"Circuit breaker triggered: numerical instability/divergence at step {step} (loss={loss_val})"
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        step_elapsed = time.time() - step_start
        run_metrics["step_times"].append(step_elapsed)
        interval_train_loss += loss_val
        interval_steps += 1

        is_boundary_1 = (step == boundary_1)
        is_boundary_2 = (step == boundary_2)
        is_equal_target = (step == equal_target_step and condition == 'C0')
        is_periodic_log = (step % log_interval == 0)

        # ----------------------------------------------------
        # Periodic Diagnostics, Circuit Breakers & Trajectory Update
        # ----------------------------------------------------
        if is_periodic_log or is_boundary_1 or is_boundary_2 or is_equal_target:
            avg_train_loss = interval_train_loss / max(1, interval_steps)
            interval_train_loss = 0.0
            interval_steps = 0

            # Evaluate Target Task A validation performance
            val_m = evaluate_applied_metrics(model, val_loader_a, device, num_classes=8, head='A')

            # Circuit Breaker 2: Degenerate class collapse
            # Only checked during Phase 1 (post warmup) and Phase 3 (relearning)
            if (1000 <= step <= boundary_1) and (val_m['balanced_accuracy'] < 10.0):
                raise CircuitBreakerError(
                    f"Circuit breaker triggered: Phase 1 class collapse at step {step} "
                    f"(balanced accuracy {val_m['balanced_accuracy']:.2f}% < 10.0%)"
                )
            if (step > boundary_2 + 1000) and (val_m['balanced_accuracy'] < 10.0):
                raise CircuitBreakerError(
                    f"Circuit breaker triggered: Phase 3 class collapse at step {step} "
                    f"(balanced accuracy {val_m['balanced_accuracy']:.2f}% < 10.0%)"
                )

            # Representation EffRank diagnostics
            current_er = None
            if eval_probe and probe_loader is not None and (step % 1000 == 0 or is_boundary_1 or is_boundary_2):
                probe_feats = extract_probe_features(model, probe_loader, device)
                current_er = compute_effective_rank(probe_feats)

            # Append to trajectory CSV
            with open(traj_csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=traj_fields)
                writer.writerow({
                    "step": step,
                    "train_loss": f"{avg_train_loss:.4f}",
                    "val_loss": f"{val_m['val_loss']:.4f}",
                    "val_top1": f"{val_m['top1_accuracy']:.2f}",
                    "val_bal_acc": f"{val_m['balanced_accuracy']:.2f}",
                    "val_macro_auc": f"{val_m['macro_auc']:.4f}",
                    "eff_rank": f"{current_er:.2f}" if current_er is not None else "",
                    "lr": f"{current_lr:.6f}",
                    "elapsed_s": f"{time.time() - start_wall_time:.1f}",
                })

            # Regenerate live visual dashboard
            if plot_applied_trajectory is not None:
                try:
                    plot_applied_trajectory(
                        trajectory_csv=traj_csv_path,
                        output_png=dashboard_png_path,
                        condition=condition,
                        seed=seed,
                        u1=u1,
                        uB=uB,
                        u2=u2
                    )
                except Exception as pe:
                    print(f"[{condition} | Seed {seed}] Warning: dashboard plot failed: {pe}")

            # Console output
            avg_step_ms = (sum(run_metrics["step_times"][-log_interval:]) / max(1, min(log_interval, len(run_metrics["step_times"])))) * 1000.0
            print(f"[{condition} | Seed {seed}] Step {step}/{total_updates} | Loss: {avg_train_loss:.4f} | Val Loss: {val_m['val_loss']:.4f} | Top-1: {val_m['top1_accuracy']:.2f}% | Bal Acc: {val_m['balanced_accuracy']:.2f}% | AUC: {val_m['macro_auc']:.4f} | LR: {current_lr:.5f} | Step Time: {avg_step_ms:.1f}ms")

        # ----------------------------------------------------
        # Milestone Checkpoint 1 (theta1 at step u1)
        # ----------------------------------------------------
        if is_boundary_1:
            ckpt_path = os.path.join(checkpoint_dir, f"{condition}_seed{seed}_theta1.pt")
            torch.save(model.state_dict(), ckpt_path)
            
            diag = {}
            if eval_probe and probe_loader is not None:
                theta1_probe_features = extract_probe_features(model, probe_loader, device)
                diag["eff_rank"] = compute_effective_rank(theta1_probe_features)
                diag["cka_drift"] = 0.0
                print(f"[{condition} | theta1 @ step {step}] EffRank: {diag['eff_rank']:.2f}")

            run_metrics["checkpoints"]["theta1"] = {
                "step": step,
                "metrics": val_m,
                "diagnostics": diag,
                "checkpoint": ckpt_path,
            }

        # ----------------------------------------------------
        # Milestone Checkpoint 2 (theta2 at step u1 + uB)
        # ----------------------------------------------------
        if is_boundary_2:
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
        if is_equal_target:
            ckpt_path = os.path.join(checkpoint_dir, f"{condition}_seed{seed}_theta_10k_equal_target.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[{condition} | Equal-Target @ step {step}] Val Top-1: {val_m['top1_accuracy']:.2f}%, Macro AUC: {val_m['macro_auc']:.4f}")
            run_metrics["checkpoints"]["theta_10k"] = {
                "step": step,
                "metrics": val_m,
                "checkpoint": ckpt_path,
            }

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

    # Final trajectory row
    with open(traj_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=traj_fields)
        writer.writerow({
            "step": total_updates,
            "train_loss": f"{loss_val:.4f}",
            "val_loss": f"{val_terminal['val_loss']:.4f}",
            "val_top1": f"{val_terminal['top1_accuracy']:.2f}",
            "val_bal_acc": f"{val_terminal['balanced_accuracy']:.2f}",
            "val_macro_auc": f"{val_terminal['macro_auc']:.4f}",
            "eff_rank": f"{diag.get('eff_rank', 0.0):.2f}" if diag else "",
            "lr": f"{current_lr:.6f}",
            "elapsed_s": f"{time.time() - start_wall_time:.1f}",
        })

    # Final dashboard render
    if plot_applied_trajectory is not None:
        try:
            plot_applied_trajectory(
                trajectory_csv=traj_csv_path,
                output_png=dashboard_png_path,
                condition=condition,
                seed=seed,
                u1=u1,
                uB=uB,
                u2=u2
            )
        except Exception as pe:
            print(f"[{condition} | Seed {seed}] Final dashboard plot error: {pe}")

    total_wall_time = time.time() - start_wall_time
    avg_step_time_ms = (sum(run_metrics["step_times"]) / max(1, len(run_metrics["step_times"]))) * 1000.0

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
