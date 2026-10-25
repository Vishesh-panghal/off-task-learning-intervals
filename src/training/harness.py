"""
Core training harness for SOTLI (Structured Off-Task Learning Intervals).
Handles:
- Exact update-based budgeting (u_1, u_B, u_2)
- Clean Phase transition management:
  * Stashing Head A and its optimizer momentum buffers during Phase 2
  * Initializing temporary Head B for Task B
  * Restoring Head A and its momentum buffers for Phase 3
- Continuous baseline execution (Control 1: Equal Total Updates, Control 2: Equal Task-A Updates)
"""

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.metrics import (
    evaluate_accuracy, extract_probe_features, compute_effective_rank,
    compute_linear_cka, compute_gradient_cosine_backbone
)
from src.training.evaluator import RelearningDynamicsTracker, run_diagnostic_probe_at_theta2


def infinite_loader(dataloader):
    """Yields batches infinitely from dataloader across epochs"""
    while True:
        for batch in dataloader:
            yield batch


def stash_head_optimizer_state(optimizer, head_param_ids):
    """
    Extracts and stashes optimizer momentum buffers belonging to Head A parameters.
    """
    stashed_state = {}
    for group in optimizer.param_groups:
        for p in group['params']:
            if id(p) in head_param_ids:
                if p in optimizer.state:
                    stashed_state[id(p)] = copy.deepcopy(optimizer.state[p])
                    # Clear state for frozen param during interval
                    del optimizer.state[p]
    return stashed_state


def restore_head_optimizer_state(optimizer, head_param_ids, stashed_state):
    """
    Restores optimizer momentum buffers for Head A parameters upon returning to Task A.
    """
    for group in optimizer.param_groups:
        for p in group['params']:
            if id(p) in head_param_ids and id(p) in stashed_state:
                optimizer.state[p] = stashed_state[id(p)]


def train_continuous_baseline(
    model, train_loader_a, val_loader_a, test_loader_a, probe_loader,
    total_updates=35000, eval_intermediate_update=30000,
    lr=0.1, momentum=0.9, weight_decay=5e-4,
    device='cpu', checkpoint_dir='./checkpoints', run_id='continuous_seed42'
):
    """
    Executes Control 1 (Equal Total Updates: 35,000 updates) and exports intermediate
    checkpoint at update 30,000 (Control 2: Equal Task-A Updates).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    loader_iter = infinite_loader(train_loader_a)
    model.to(device)

    print(f"[{run_id}] Starting Continuous Baseline training for {total_updates} updates...")
    
    p_control2_val = None
    checkpoint_theta_c_a = None

    for step in range(1, total_updates + 1):
        model.train()
        images, labels = next(loader_iter)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, head='A')
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Intermediate checkpoint for Control 2 (Equal Task A exposure: u1 + u2 = 30,000)
        if step == eval_intermediate_update:
            p_control2_val = evaluate_accuracy(model, val_loader_a, device, head='A')
            checkpoint_theta_c_a = os.path.join(checkpoint_dir, f"{run_id}_theta_C_A_30k.pt")
            torch.save(model.state_dict(), checkpoint_theta_c_a)
            print(f"[{run_id}] Update {step}: Saved Control 2 Checkpoint (Val Acc: {p_control2_val:.2f}%)")

        if step % 2500 == 0:
            val_acc = evaluate_accuracy(model, val_loader_a, device, head='A')
            print(f"[{run_id}] Step {step}/{total_updates} - Loss: {loss.item():.4f} - Val Acc: {val_acc:.2f}%")

    # Final Checkpoint for Control 1 (Equal Total Updates: 35,000)
    p_control1_val = evaluate_accuracy(model, val_loader_a, device, head='A')
    p_control1_test = evaluate_accuracy(model, test_loader_a, device, head='A')
    checkpoint_theta_c_total = os.path.join(checkpoint_dir, f"{run_id}_theta_C_total_35k.pt")
    torch.save(model.state_dict(), checkpoint_theta_c_total)
    
    print(f"[{run_id}] Completed. Control 1 Val Acc: {p_control1_val:.2f}%, Test Acc: {p_control1_test:.2f}%")

    return {
        'checkpoint_theta_C_A': checkpoint_theta_c_a,
        'checkpoint_theta_C_total': checkpoint_theta_c_total,
        'p_control2_val': p_control2_val,
        'p_control1_val': p_control1_val,
        'p_control1_test': p_control1_test,
    }


def train_sotli_schedule(
    model, train_loader_a, val_loader_a, test_loader_a, aux_loader_b, num_classes_b,
    probe_loader, condition_id,
    u1=15000, u_b=5000, u2=15000,
    lr=0.1, momentum=0.9, weight_decay=5e-4,
    device='cpu', checkpoint_dir='./checkpoints', run_id='sotli_B4_seed42'
):
    """
    Executes the SOTLI 3-phase schedule: A^{u1} -> B^{u_B} -> A^{u2}.
    Standardized: u1 = 15k, u_B = 5k, u2 = 15k (Total = 35k updates).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    # -------------------------------------------------------------
    # PHASE 1: Initial Target Task A Training (u1 = 15,000 updates)
    # -------------------------------------------------------------
    print(f"[{run_id}] Phase 1: Training Task A for {u1} updates...")
    optimizer_a = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    loader_a_iter = infinite_loader(train_loader_a)

    for step in range(1, u1 + 1):
        model.train()
        images, labels = next(loader_a_iter)
        images, labels = images.to(device), labels.to(device)

        optimizer_a.zero_grad()
        outputs = model(images, head='A')
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_a.step()

        if step % 2500 == 0:
            val_acc = evaluate_accuracy(model, val_loader_a, device, head='A')
            print(f"[{run_id}] Phase 1 Step {step}/{u1} - Val Acc: {val_acc:.2f}%")

    # Checkpoint theta_1 (End of Phase 1)
    p_before = evaluate_accuracy(model, val_loader_a, device, head='A')
    theta1_path = os.path.join(checkpoint_dir, f"{run_id}_theta1.pt")
    torch.save(model.state_dict(), theta1_path)
    
    probe_feats_theta1 = extract_probe_features(model, probe_loader, device)
    eff_rank_theta1 = compute_effective_rank(probe_feats_theta1)
    print(f"[{run_id}] Checkpoint θ₁ locked. P_before: {p_before:.2f}%, EffRank: {eff_rank_theta1:.2f}")

    # -------------------------------------------------------------
    # TRANSITION A -> B
    # -------------------------------------------------------------
    head_a_param_ids = {id(p) for p in model.head_a.parameters()}
    stashed_opt_state = stash_head_optimizer_state(optimizer_a, head_a_param_ids)
    model.freeze_head_a()
    model.init_auxiliary_head(num_classes_b, device=device)

    # Compute pre-interval Gradient Cosine S_g on shared backbone
    batch_a_sample = next(loader_a_iter)
    loader_b_iter = infinite_loader(aux_loader_b)
    batch_b_sample = next(loader_b_iter)
    s_g_theta1 = compute_gradient_cosine_backbone(model, batch_a_sample, batch_b_sample, criterion, device)
    print(f"[{run_id}] Computed Shared-Backbone Gradient Cosine S_g(A, B): {s_g_theta1:.4f}")

    # -------------------------------------------------------------
    # PHASE 2: Auxiliary Interval Training (u_b = 5,000 updates)
    # -------------------------------------------------------------
    print(f"[{run_id}] Phase 2: Training Auxiliary Task {condition_id} for {u_b} updates...")
    # Optimizer for Phase 2 covers shared backbone + new head_b
    optimizer_b = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    for step in range(1, u_b + 1):
        model.train()
        images, labels = next(loader_b_iter)
        images, labels = images.to(device), labels.to(device)

        optimizer_b.zero_grad()
        outputs = model(images, head='B')
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_b.step()

        if step % 1000 == 0:
            print(f"[{run_id}] Phase 2 Step {step}/{u_b} - Aux Loss: {loss.item():.4f}")

    # Checkpoint theta_2 (End of Phase 2)
    theta2_path = os.path.join(checkpoint_dir, f"{run_id}_theta2.pt")
    torch.save(model.state_dict(), theta2_path)

    # Immediate Target Performance & Forgetting
    p_after_b = evaluate_accuracy(model, val_loader_a, device, head='A')
    f_b = p_before - p_after_b

    # Diagnostic probe at theta_2 (tests feature retention on frozen backbone)
    p_diagnostic_probe = run_diagnostic_probe_at_theta2(
        model, train_loader_a, val_loader_a, device, num_classes_a=100, probe_epochs=5
    )

    # Representation Drift & Rank at theta_2
    probe_feats_theta2 = extract_probe_features(model, probe_loader, device)
    cka_drift = 1.0 - compute_linear_cka(probe_feats_theta1, probe_feats_theta2)
    eff_rank_theta2 = compute_effective_rank(probe_feats_theta2)

    print(f"[{run_id}] Checkpoint θ₂ locked.")
    print(f"       P_after_B (Frozen Head A): {p_after_b:.2f}% (Drop F_B: {f_b:.2f}%)")
    print(f"       Diagnostic Probe (Task A on Frozen Backbone): {p_diagnostic_probe:.2f}%")
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
    # PHASE 3: Target Task Relearning with Dense Tracking (u2 = 15,000 updates)
    # -------------------------------------------------------------
    print(f"[{run_id}] Phase 3: Relearning Task A for {u2} updates with dense logging...")
    dynamics_tracker = RelearningDynamicsTracker(p_before=p_before)

    for step in range(1, u2 + 1):
        model.train()
        images, labels = next(loader_a_iter)
        images, labels = images.to(device), labels.to(device)

        optimizer_a.zero_grad()
        outputs = model(images, head='A')
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_a.step()

        # Dense validation logging every 100 updates
        if step % 100 == 0 or step == u2:
            val_acc = evaluate_accuracy(model, val_loader_a, device, head='A')
            dynamics_tracker.record_step(rel_update=step, val_acc=val_acc)
            if step % 2500 == 0:
                print(f"[{run_id}] Phase 3 Step {step}/{u2} - Val Acc: {val_acc:.2f}%")

    # Checkpoint theta_3 (End of Phase 3 / Final Model)
    theta3_path = os.path.join(checkpoint_dir, f"{run_id}_theta3.pt")
    torch.save(model.state_dict(), theta3_path)

    p_relearned = evaluate_accuracy(model, val_loader_a, device, head='A')
    p_relearned_test = evaluate_accuracy(model, test_loader_a, device, head='A')
    r_a = p_relearned - p_after_b

    probe_feats_theta3 = extract_probe_features(model, probe_loader, device)
    eff_rank_theta3 = compute_effective_rank(probe_feats_theta3)

    dynamics_summary = dynamics_tracker.get_summary()

    print(f"[{run_id}] Final Checkpoint θ₃ locked.")
    print(f"       P_relearned (Val): {p_relearned:.2f}%, Test Acc: {p_relearned_test:.2f}%")
    print(f"       Recovery Gain R_A: {r_a:.2f}%, u_rec: {dynamics_summary['u_rec']} updates, AULC: {dynamics_summary['aulc']:.2f}")

    return {
        'run_id': run_id,
        'condition_id': condition_id,
        'checkpoint_theta1': theta1_path,
        'checkpoint_theta2': theta2_path,
        'checkpoint_theta3': theta3_path,
        'p_before': p_before,
        'p_after_b': p_after_b,
        'f_b': f_b,
        'p_diagnostic_probe': p_diagnostic_probe,
        'p_relearned': p_relearned,
        'p_relearned_test': p_relearned_test,
        'r_a': r_a,
        's_g': s_g_theta1,
        'cka_drift': cka_drift,
        'eff_rank_theta1': eff_rank_theta1,
        'eff_rank_theta2': eff_rank_theta2,
        'eff_rank_theta3': eff_rank_theta3,
        **dynamics_summary
    }
