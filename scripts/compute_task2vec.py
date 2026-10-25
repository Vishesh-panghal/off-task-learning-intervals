"""
Task2Vec Fisher Distance Extraction Engine for SOTLI (Gate 3).
Computes the diagonal Fisher Information Matrix embedding for each dataset
using a fixed, external probe network (ImageNet-pretrained ResNet-34).
Computes pairwise Task2Vec cosine distance d_task(A, B) and similarity S_task(A, B).
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import resnet34, ResNet34_Weights
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.datasets import get_cifar100_loaders, get_auxiliary_loader


def get_probe_network(device):
    """Loads a fixed ImageNet-pretrained ResNet-34 probe network."""
    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights)
    model.eval()
    return model.to(device)


def compute_fisher_embedding(model, loader, device, max_batches=40):
    """
    Computes diagonal Fisher Information Matrix embedding:
    F_i = E_{x, y} [ (nabla_{theta_i} log p(y|x))^2 ]
    Evaluated across classifier and penultimate layer parameters.
    """
    model.eval()
    fisher_diag = {}
    
    # Target parameter layers: fc classifier and layer4 convs
    target_params = [
        (name, p) for name, p in model.named_parameters()
        if "layer4" in name or "fc" in name
    ]
    
    for name, p in target_params:
        fisher_diag[name] = torch.zeros_like(p, device=device)
        
    total_samples = 0
    
    for batch_idx, (images, _) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        images = images.to(device)
        # ResNet34 expects 224x224 or at least bilinear upsampling from 32x32
        if images.shape[-1] < 224:
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            
        batch_size = images.size(0)
        total_samples += batch_size
        
        logits = model(images)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Sample pseudo-labels from model's predicted categorical distribution
        probs = F.softmax(logits, dim=-1)
        pseudo_labels = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        loss = F.nll_loss(log_probs, pseudo_labels)
        model.zero_grad()
        loss.backward(retain_graph=False)
        
        for name, p in target_params:
            if p.grad is not None:
                fisher_diag[name] += (p.grad.data ** 2) * batch_size

    # Concatenate normalized Fisher diagonal vector
    vec_parts = []
    for name, _ in target_params:
        f_val = fisher_diag[name] / max(total_samples, 1)
        vec_parts.append(f_val.view(-1))
        
    fisher_vector = torch.cat(vec_parts).detach().cpu().numpy()
    # Normalize
    norm = np.linalg.norm(fisher_vector)
    if norm > 0:
        fisher_vector = fisher_vector / norm
    return fisher_vector


def cosine_distance(u, v):
    """Computes cosine distance: 1 - (u . v) / (||u|| ||v||)"""
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 1.0
    cosine_sim = np.clip(np.dot(u, v) / (norm_u * norm_v), -1.0, 1.0)
    return float(1.0 - cosine_sim)


def main():
    parser = argparse.ArgumentParser(description="Task2Vec Distance Engine")
    parser.add_argument("--data_dir", type=str, default="./data_cache")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--max_batches", type=int, default=30)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=================================================================")
    print(" SOTLI GATE 3: TASK2VEC FISHER DISTANCE ENGINE")
    print(f" Target Venue: TMLR")
    print(f" Probe Network: ImageNet-pretrained ResNet-34")
    print(f" Device: {device}")
    print("=================================================================\n")

    os.makedirs(args.results_dir, exist_ok=True)
    probe_model = get_probe_network(device)

    # 1. Compute Target Task A (CIFAR-100) Embedding
    print("[Task2Vec] Extracting Fisher embedding for Target Task A (CIFAR-100)...")
    train_loader_a, _, _ = get_cifar100_loaders(data_dir=args.data_dir, batch_size=64, seed=42)
    fim_target_a = compute_fisher_embedding(probe_model, train_loader_a, device, max_batches=args.max_batches)
    print(f"✓ Target Task A embedding extracted (dim={len(fim_target_a)}).")

    # 2. Extract for each Auxiliary Condition
    conditions = {
        'B0': 'CIFAR-100 (Self-Reference)',
        'B1': 'CIFAR-100 (Head-Switch Control)',
        'B3': 'CIFAR-10 (Related Real Natural Images)',
        'B4': 'STL-10 (Distinct Real Natural Images)',
        'B5': 'SVHN (Distant Real Cropped Digits)',
        'B7': 'Synthetic Gaussian Noise Images'
    }

    embeddings = {'A': fim_target_a}
    results = []

    for cond_id, desc in conditions.items():
        print(f"\n[Task2Vec] Processing Condition {cond_id}: {desc}...")
        if cond_id in ['B0', 'B1']:
            fim_b = fim_target_a
        else:
            aux_loader, _ = get_auxiliary_loader(cond_id, data_dir=args.data_dir, batch_size=64, seed=42)
            fim_b = compute_fisher_embedding(probe_model, aux_loader, device, max_batches=args.max_batches)
            
        embeddings[cond_id] = fim_b
        d_task = cosine_distance(fim_target_a, fim_b)
        s_task = float(np.exp(-d_task))
        
        print(f"  -> d_task(A, {cond_id}): {d_task:.4f}")
        print(f"  -> S_task(A, {cond_id}): {s_task:.4f}")
        
        results.append({
            'condition_id': cond_id,
            'description': desc,
            'd_task': round(d_task, 4),
            's_task': round(s_task, 4)
        })

    # Save to CSV
    output_csv = os.path.join(args.results_dir, "task2vec_distances.csv")
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False)
    print(f"\n✓ Saved Task2Vec distance matrix -> {output_csv}")
    print("\nSummary Table:")
    print(df_out.to_string(index=False))

    # Also save raw embeddings for downstream modeling
    emb_path = os.path.join(args.results_dir, "task2vec_embeddings.pt")
    torch.save(embeddings, emb_path)
    print(f"✓ Saved raw Fisher vectors -> {emb_path}")
    print("\n=================================================================")
    print(" TASK2VEC EXTRACTION COMPLETED SUCCESSFULLY! 🎉")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
