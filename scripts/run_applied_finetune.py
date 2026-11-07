"""
Production Training Runner for the Applied Fine-Tuning Regime (Scenario B: ISIC 2019 + Imagenette).

Runs pilot (N=2), primary confirmatory (N=10), or factorial (N=6) blocks under
the prespecified 12,000-update protocol. Outputs seed-level CSV rows and summary JSONs.
"""

import os
import sys
import argparse
import time
import json
import csv
import torch
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.resnet_pretrained import PretrainedResNet50DualHead
from src.data.datasets import (
    get_applied_isic2019_loaders, get_applied_imagenette_loader, get_applied_isic_probe_loader
)
from src.training.applied_harness import run_applied_finetune_experiment, CircuitBreakerError


CSV_FIELDNAMES = [
    "condition", "seed", "u1", "uB", "u2", "total_updates",
    "theta1_top1", "theta1_auc", "theta1_bal_acc", "theta1_eff_rank",
    "theta2_eff_rank", "theta2_cka_drift",
    "theta_10k_top1", "theta_10k_auc",
    "theta3_top1", "theta3_auc", "theta3_bal_acc", "theta3_ece", "theta3_brier",
    "theta3_eff_rank", "theta3_cka_drift",
    "total_wall_time_s", "avg_step_ms", "timestamp"
]


def append_result_to_csv(csv_path, row_data):
    """Atomically appends a completed run result row to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row_data)
        f.flush()


def main():
    parser = argparse.ArgumentParser(description="Run applied fine-tuning experiments.")
    parser.add_argument("--conditions", nargs="+", default=["C0", "C3", "S1"],
                        help="List of conditions to run (e.g. C0 C3 S1 or F00 F01 F10 F11)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1001],
                        help="List of integer seeds to run")
    parser.add_argument("--u1", type=int, default=5000, help="Phase 1 updates (default: 5000)")
    parser.add_argument("--uB", type=int, default=2000, help="Phase 2 updates (default: 2000)")
    parser.add_argument("--u2", type=int, default=5000, help="Phase 3 updates (default: 5000)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--base_lr", type=float, default=0.01, help="Base learning rate (default: 0.01)")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup updates (default: 500)")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (cuda or cpu)")
    parser.add_argument("--data_dir", type=str, default="./data_cache")
    parser.add_argument("--splits_dir", type=str, default="./data_cache/splits")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/applied")
    parser.add_argument("--csv_output", type=str, default="./results/applied_runs.csv")
    parser.add_argument("--log_interval", type=int, default=250, help="Updates between trajectory logs (default: 250)")
    args = parser.parse_args()

    isic_splits = os.path.join(args.splits_dir, "isic2019_splits.json")
    isic_images = os.path.join(args.data_dir, "isic2019_224/train-image/image")
    imagenette_splits = os.path.join(args.splits_dir, "imagenette_splits.json")
    imagenette_dir = os.path.join(args.data_dir, "imagenette2")
    probe_json = os.path.join(args.splits_dir, "isic2019_probe_2000.json")

    print("\n" + "="*65)
    print(" SOTLI APPLIED VISION FINE-TUNING RUNNER (12k Updates)")
    print(f" Conditions ({len(args.conditions)}): {args.conditions}")
    print(f" Seeds ({len(args.seeds)}): {args.seeds}")
    print(f" Total planned runs: {len(args.conditions) * len(args.seeds)}")
    print(f" Update schedule: u1={args.u1}, uB={args.uB}, u2={args.u2} (Total: {args.u1 + args.uB + args.u2})")
    print(f" Batch size: {args.batch_size} | Warmup: {args.warmup_steps} | Base LR: {args.base_lr}")
    print(f" Results CSV: {args.csv_output}")
    print("="*65 + "\n")

    print("[Runner] Initializing Target and Probe DataLoaders...")
    train_l, val_l, test_l = get_applied_isic2019_loaders(
        isic_splits, isic_images, batch_size=args.batch_size, num_workers=4
    )
    probe_l = get_applied_isic_probe_loader(
        probe_json, isic_images, batch_size=args.batch_size, num_workers=4
    )
    print("[Runner] DataLoaders initialized successfully.\n")

    for seed in args.seeds:
        for cond in args.conditions:
            run_start_time = time.time()
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Fresh ImageNet-pretrained ResNet-50 instance
            model = PretrainedResNet50DualHead(num_classes_a=8)
            seed_ckpt_dir = os.path.join(args.checkpoint_dir, f"seed_{seed}", cond)

            # Build condition-specific auxiliary loader if needed
            if cond == 'S1':
                aux_loader, _ = get_applied_imagenette_loader(
                    imagenette_splits, imagenette_dir, batch_size=args.batch_size,
                    num_workers=4, shuffle_labels=False, seed=seed
                )
            elif cond == 'S1-shuf':
                aux_loader, _ = get_applied_imagenette_loader(
                    imagenette_splits, imagenette_dir, batch_size=args.batch_size,
                    num_workers=4, shuffle_labels=True, seed=seed
                )
            else:
                aux_loader = None

            try:
                res = run_applied_finetune_experiment(
                    model=model,
                    train_loader_a=train_l,
                    val_loader_a=val_l,
                    test_loader_a=test_l,
                    probe_loader=probe_l,
                    aux_loader=aux_loader,
                    condition=cond,
                    seed=seed,
                    u1=args.u1,
                    uB=args.uB,
                    u2=args.u2,
                    base_lr=args.base_lr,
                    warmup_steps=args.warmup_steps,
                    device=args.device,
                    checkpoint_dir=seed_ckpt_dir,
                    eval_probe=True,
                    log_interval=args.log_interval
                )

                # Extract metrics for CSV
                t1 = res["checkpoints"].get("theta1", {})
                t1_m = t1.get("metrics", {})
                t1_d = t1.get("diagnostics", {})

                t2 = res["checkpoints"].get("theta2", {})
                t2_d = t2.get("diagnostics", {})

                t10k = res["checkpoints"].get("theta_10k", {})
                t10k_m = t10k.get("metrics", {})

                t3 = res["checkpoints"].get("theta3", {})
                t3_m = t3.get("metrics", {})
                t3_d = t3.get("diagnostics", {})

                row = {
                    "condition": cond,
                    "seed": seed,
                    "u1": args.u1,
                    "uB": args.uB,
                    "u2": args.u2,
                    "total_updates": args.u1 + args.uB + args.u2,
                    "theta1_top1": t1_m.get("top1_accuracy", ""),
                    "theta1_auc": t1_m.get("macro_auc", ""),
                    "theta1_bal_acc": t1_m.get("balanced_accuracy", ""),
                    "theta1_eff_rank": t1_d.get("eff_rank", ""),
                    "theta2_eff_rank": t2_d.get("eff_rank", ""),
                    "theta2_cka_drift": t2_d.get("cka_drift", ""),
                    "theta_10k_top1": t10k_m.get("top1_accuracy", ""),
                    "theta_10k_auc": t10k_m.get("macro_auc", ""),
                    "theta3_top1": t3_m.get("top1_accuracy", ""),
                    "theta3_auc": t3_m.get("macro_auc", ""),
                    "theta3_bal_acc": t3_m.get("balanced_accuracy", ""),
                    "theta3_ece": t3_m.get("ece", ""),
                    "theta3_brier": t3_m.get("brier_score", ""),
                    "theta3_eff_rank": t3_d.get("eff_rank", ""),
                    "theta3_cka_drift": t3_d.get("cka_drift", ""),
                    "total_wall_time_s": round(res["total_wall_time_seconds"], 2),
                    "avg_step_ms": round(res["avg_step_time_ms"], 2),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }

                append_result_to_csv(args.csv_output, row)
                print(f"[Runner] Saved result row for {cond} (Seed {seed}) to {args.csv_output}\n")

            except CircuitBreakerError as cbe:
                elapsed_abort = time.time() - run_start_time
                print(f"\n{'!'*65}")
                print(f"[CIRCUIT BREAKER ABORT] {cbe}")
                print(f"Aborting {cond} (Seed {seed}) after {elapsed_abort:.1f}s to preserve GPU compute.")
                print(f"{'!'*65}\n")

                row = {
                    "condition": cond,
                    "seed": seed,
                    "u1": args.u1,
                    "uB": args.uB,
                    "u2": args.u2,
                    "total_updates": args.u1 + args.uB + args.u2,
                    "theta1_top1": f"ABORTED_CIRCUIT_BREAKER: {cbe}",
                    "total_wall_time_s": round(elapsed_abort, 2),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                append_result_to_csv(args.csv_output, row)


if __name__ == "__main__":
    main()

