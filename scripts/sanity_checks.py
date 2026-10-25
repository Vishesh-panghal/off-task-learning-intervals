"""
Sanity Checks & Verification Suite for SOTLI Stage 1.
Tests:
1. GroupNorm representation invariance (zero running-statistic leakage).
2. Head-A gradient freezing during Phase 2 (zero gradient leakage).
3. Metric implementations: EffRank, Linear CKA, Gradient Cosine S_g.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.resnet_gn import ResNet18GroupNorm
from src.utils.metrics import (
    compute_effective_rank, compute_linear_cka, compute_gradient_cosine_backbone
)


def test_groupnorm_invariance():
    """
    Asserts that GroupNorm has ZERO running statistics.
    Processing an arbitrary Batch B must NOT alter the feature activations of Batch A.
    """
    print("[Test 1/3] Verifying GroupNorm Representation Invariance...")
    model = ResNet18GroupNorm(num_classes_a=100)
    model.eval()

    batch_a = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        feat_a_initial = model(batch_a, head='features')

    # Now pass a drastically different distribution (e.g. huge variance, different mean)
    batch_b = torch.randn(16, 3, 32, 32) * 100.0 + 50.0
    with torch.no_grad():
        _ = model(batch_b, head='features')

    # Pass batch_a again
    with torch.no_grad():
        feat_a_second = model(batch_a, head='features')

    diff = (feat_a_initial - feat_a_second).abs().max().item()
    assert diff == 0.0, f"GroupNorm failed invariance! Max diff: {diff}"
    print("   ✓ Passed: GroupNorm has zero running-statistic drift (max diff = 0.000000).")


def test_head_isolation():
    """
    Asserts that when Head A is frozen during Phase 2, its gradients are strictly zero.
    """
    print("[Test 2/3] Verifying Head A Gradient Isolation...")
    model = ResNet18GroupNorm(num_classes_a=100)
    model.init_auxiliary_head(num_classes_b=10)
    model.freeze_head_a()

    batch_b = torch.randn(4, 3, 32, 32)
    targets_b = torch.tensor([0, 1, 2, 3])

    outputs = model(batch_b, head='B')
    loss = nn.CrossEntropyLoss()(outputs, targets_b)
    loss.backward()

    # Verify Head A parameters have requires_grad=False and grad is None
    for name, param in model.head_a.named_parameters():
        assert not param.requires_grad, f"Param {name} still has requires_grad=True!"
        assert param.grad is None, f"Param {name} received non-zero gradient during Phase 2!"

    # Verify Head B and Backbone DID receive gradients
    backbone_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.get_shared_parameters())
    assert backbone_has_grad, "Backbone failed to receive gradients from Head B!"
    assert model.head_b.weight.grad is not None, "Head B failed to receive gradients!"

    print("   ✓ Passed: Head A received strictly 0 gradients while Backbone and Head B updated.")


def test_metrics():
    """
    Verifies that EffRank, Linear CKA, and Gradient Cosine S_g execute correctly without errors.
    """
    print("[Test 3/3] Verifying Metric Functions...")
    
    # 1. EffRank
    # Identical features -> rank 1
    features_rank1 = torch.ones(50, 512)
    rank1 = compute_effective_rank(features_rank1)
    assert 0.9 <= rank1 <= 1.1, f"EffRank of identical vectors should be ~1, got {rank1}"

    # Random features -> high rank
    features_random = torch.randn(100, 512)
    rank_rand = compute_effective_rank(features_random)
    assert rank_rand > 10.0, f"EffRank of random vectors should be high, got {rank_rand}"

    # 2. Linear CKA
    # Identical representations -> CKA = 1.0
    cka_self = compute_linear_cka(features_random, features_random)
    assert abs(cka_self - 1.0) < 1e-4, f"Self-CKA must be 1.0, got {cka_self}"

    # Independent random representations (N >> D so Gram matrices don't collapse to isotropic identities) -> CKA ~ 0.0
    feat_x = torch.randn(1000, 10)
    feat_y = torch.randn(1000, 10)
    cka_diff = compute_linear_cka(feat_x, feat_y)
    assert cka_diff < 0.1, f"Different representations should have low CKA, got {cka_diff}"

    # 3. Gradient Cosine Backbone
    model = ResNet18GroupNorm(num_classes_a=100)
    model.init_auxiliary_head(num_classes_b=10)
    batch_a = (torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 3]))
    batch_b = (torch.randn(4, 3, 32, 32), torch.tensor([4, 5, 6, 7]))
    criterion = nn.CrossEntropyLoss()
    s_g = compute_gradient_cosine_backbone(model, batch_a, batch_b, criterion, 'cpu')
    assert -1.0 <= s_g <= 1.0, f"Cosine similarity out of bounds: {s_g}"

    print(f"   ✓ Passed: EffRank (rank1={rank1:.2f}, rank_rand={rank_rand:.2f}), CKA (self={cka_self:.4f}, diff={cka_diff:.4f}), S_g={s_g:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING SOTLI GATE 1 SANITY CHECKS")
    print("=" * 60)
    test_groupnorm_invariance()
    test_head_isolation()
    test_metrics()
    print("=" * 60)
    print("ALL GATE 1 SANITY CHECKS PASSED SUCCESSFULLY! ✅")
    print("=" * 60)
