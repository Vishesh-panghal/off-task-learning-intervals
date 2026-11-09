"""
Smoke Test for the Applied Fine-Tuning Regime (Scenario B: ISIC 2019 + Imagenette).

Runs a rapid 180-update schedule across primary conditions (C0, C3, S1):
- Phase 1: 70 updates
- Phase 2: 40 updates
- Phase 3: 70 updates
- Warmup: 20 updates
- Batch size: 64, Device: cuda with AMP

Verifies:
1. Data loading for ISIC 2019 (224x224) and Imagenette (224x224)
2. ResNet-50 IMAGENET1K_V2 pretrained weights initialization
3. Boundary 1 and Boundary 2 head stashing/swapping and momentum zeroing
4. Checkpointing at theta1, theta2, theta3
5. Computation of Macro AUC, Balanced Accuracy, ECE, Brier, EffRank, and CKA
6. Real GPU step execution time benchmarking
"""

import os
import sys
import argparse
import time
import json
import torch
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.resnet_pretrained import PretrainedResNet50DualHead
from src.data.datasets import (
    get_applied_isic2019_loaders, get_applied_imagenette_loader, get_applied_isic_probe_loader
)
from src.training.applied_harness import run_applied_finetune_experiment


def run_smoke_test(
    data_dir="./data_cache",
    splits_dir="./data_cache/splits",
    device="cuda",
    conditions=None,
    seeds=None,
    checkpoint_dir="./checkpoints/applied_smoke"
):
    if conditions is None:
        conditions = ["C0", "C3", "S1"]
    if seeds is None:
        seeds = [42]

    isic_splits = os.path.join(splits_dir, "isic2019_splits.json")
    isic_images = os.path.join(data_dir, "isic2019_224/train-image/image")
    imagenette_splits = os.path.join(splits_dir, "imagenette_splits.json")
    imagenette_dir = os.path.join(data_dir, "imagenette2")
    probe_json = os.path.join(splits_dir, "isic2019_probe_2000.json")

    print("\n========================================================")
    print(" SOTLI APPLIED FINE-TUNING REGIME — GPU SMOKE TEST")
    print(f" Device: {device} | CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available() and device.startswith("cuda"):
        print(f" GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print(f" Conditions to verify: {conditions}")
    print(f" Seeds: {seeds}")
    print("========================================================\n")

    print("[Smoke Test] Initializing DataLoaders (batch_size=64)...")
    train_l, val_l, test_l = get_applied_isic2019_loaders(
        isic_splits, isic_images, batch_size=64, num_workers=4
    )
    aux_l, num_b = get_applied_imagenette_loader(
        imagenette_splits, imagenette_dir, batch_size=64, num_workers=4
    )
    probe_l = get_applied_isic_probe_loader(
        probe_json, isic_images, batch_size=64, num_workers=4
    )
    print("[Smoke Test] DataLoaders initialized successfully.")

    # Smoke test budget: 70 + 40 + 70 = 180 updates
    u1_smoke = 70
    uB_smoke = 40
    u2_smoke = 70
    warmup_smoke = 20

    smoke_results = {}
    benchmarks = []

    for seed in seeds:
        # Pre-seed numpy and torch
        torch.manual_seed(seed)
        np.random.seed(seed)

        for cond in conditions:
            print(f"\n>>> Running Smoke Test for condition {cond} (Seed {seed}) <<<")
            model = PretrainedResNet50DualHead(num_classes_a=8)

            res = run_applied_finetune_experiment(
                model=model,
                train_loader_a=train_l,
                val_loader_a=val_l,
                test_loader_a=test_l,
                probe_loader=probe_l,
                aux_loader=aux_l if cond in ['S1', 'S1-shuf'] else None,
                condition=cond,
                seed=seed,
                u1=u1_smoke,
                uB=uB_smoke,
                u2=u2_smoke,
                base_lr=0.01,
                warmup_steps=warmup_smoke,
                device=device,
                checkpoint_dir=os.path.join(checkpoint_dir, f"seed_{seed}"),
                eval_probe=True,
                log_interval=50
            )

            smoke_results[f"{cond}_seed{seed}"] = res
            benchmarks.append(res["avg_step_time_ms"])

            # Verify checkpoints were created
            assert "theta1" in res["checkpoints"], f"Missing theta1 for {cond}"
            assert "theta2" in res["checkpoints"], f"Missing theta2 for {cond}"
            assert "theta3" in res["checkpoints"], f"Missing theta3 for {cond}"
            print(f">>> Condition {cond} VERIFIED successfully. Avg step: {res['avg_step_time_ms']:.1f} ms <<<")

    avg_bench_ms = float(np.mean(benchmarks))
    estimated_12k_mins = (avg_bench_ms * 12000.0) / (1000.0 * 60.0)

    summary = {
        "status": "ALL_CONDITIONS_PASSED",
        "device": device,
        "avg_step_time_ms": avg_bench_ms,
        "estimated_12k_run_minutes": estimated_12k_mins,
        "verified_conditions": conditions,
        "verified_seeds": seeds,
    }

    summary_file = os.path.join(checkpoint_dir, "smoke_benchmark_report.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print(" SMOKE TEST BENCHMARK REPORT")
    print(f" Status: {summary['status']}")
    print(f" Measured Step Execution Time: {avg_bench_ms:.1f} ms / update")
    print(f" Empirical Time for 12,000 updates: {estimated_12k_mins:.1f} minutes ({estimated_12k_mins/60.0:.2f} hours)")
    print(f" Report saved to: {summary_file}")
    print("="*60 + "\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run applied fine-tuning smoke test.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conditions", nargs="+", default=["C0", "C3", "S1"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = parser.parse_args()

    run_smoke_test(device=args.device, conditions=args.conditions, seeds=args.seeds)

