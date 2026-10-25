"""
Evaluation metrics and analytical tools for SOTLI:
- Classification accuracy
- Effective Representation Rank (EffRank)
- Linear Centered Kernel Alignment (CKA) representation drift
- Shared-backbone gradient cosine similarity (S_g)
"""

import torch
import torch.nn.functional as F
import numpy as np


def evaluate_accuracy(model, loader, device, head='A'):
    """Computes Top-1 accuracy over a DataLoader"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images, head=head)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / max(1, total)


def extract_probe_features(model, probe_loader, device):
    """
    Extracts centered penultimate representations over the fixed probe set X_probe.
    Returns: Tensor of shape (N, 512).
    """
    model.eval()
    features = []
    with torch.no_grad():
        for images, _ in probe_loader:
            images = images.to(device)
            feat = model(images, head='features')
            features.append(feat.cpu())
    features = torch.cat(features, dim=0)
    # Center the features
    features = features - features.mean(dim=0, keepdim=True)
    return features


def compute_effective_rank(features):
    """
    Computes the Effective Representation Rank (EffRank) of a feature matrix.
    EffRank(Z) = exp( - sum_i p_i ln(p_i) ), where p_i = sigma_i / sum_j sigma_j.
    features: Tensor of shape (N, D).
    """
    if isinstance(features, torch.Tensor):
        features = features.float().numpy()

    # Singular value decomposition
    _, s, _ = np.linalg.svd(features, full_matrices=False)
    s_sum = s.sum()
    if s_sum == 0:
        return 0.0

    p = s / s_sum
    # Avoid log(0)
    p = p[p > 1e-12]
    entropy = -np.sum(p * np.log(p))
    eff_rank = float(np.exp(entropy))
    return eff_rank


def compute_linear_cka(feat1, feat2):
    """
    Computes Linear Centered Kernel Alignment (CKA) between two feature representations
    evaluated on the EXACT SAME fixed probe set X_probe.
    feat1, feat2: Tensors of shape (N, D).
    """
    if isinstance(feat1, torch.Tensor):
        X = feat1.float().numpy()
    else:
        X = feat1

    if isinstance(feat2, torch.Tensor):
        Y = feat2.float().numpy()
    else:
        Y = feat2

    # Center matrices
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Gram matrices
    K = X @ X.T
    L = Y @ Y.T

    # HSIC formulation
    # tr(K L) / sqrt( tr(K K) tr(L L) )
    hsic_kl = np.sum(K * L)
    hsic_kk = np.sum(K * K)
    hsic_ll = np.sum(L * L)

    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom == 0:
        return 0.0
    cka = float(hsic_kl / denom)
    return np.clip(cka, 0.0, 1.0)


def compute_gradient_cosine_backbone(model, batch_a, batch_b, criterion, device):
    """
    Computes Gradient Cosine Similarity (S_g) strictly over the shared backbone parameters w.
    Excludes classification heads head_a and head_b.
    """
    model.train()
    
    # 1. Gradient for Task A
    model.zero_grad()
    images_a, labels_a = batch_a
    images_a, labels_a = images_a.to(device), labels_a.to(device)
    out_a = model(images_a, head='A')
    loss_a = criterion(out_a, labels_a)
    loss_a.backward()

    # Collect shared backbone gradients
    grad_a = []
    for param in model.get_shared_parameters():
        if param.grad is not None:
            grad_a.append(param.grad.detach().view(-1))
    grad_a = torch.cat(grad_a)

    # 2. Gradient for Task B
    model.zero_grad()
    images_b, labels_b = batch_b
    images_b, labels_b = images_b.to(device), labels_b.to(device)
    out_b = model(images_b, head='B')
    loss_b = criterion(out_b, labels_b)
    loss_b.backward()

    grad_b = []
    for param in model.get_shared_parameters():
        if param.grad is not None:
            grad_b.append(param.grad.detach().view(-1))
    grad_b = torch.cat(grad_b)

    model.zero_grad()

    # Compute cosine similarity
    norm_a = grad_a.norm(p=2)
    norm_b = grad_b.norm(p=2)
    if norm_a == 0 or norm_b == 0:
        return 0.0

    cos_sim = float((grad_a @ grad_b) / (norm_a * norm_b))
    return cos_sim
