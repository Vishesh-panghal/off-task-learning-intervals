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


# =====================================================================
# Applied Fine-Tuning Diagnostics & Metrics (ISIC 2019 Applied Vision)
# =====================================================================

def compute_ece(probs, labels, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) with equal-width probability bins.
    probs: Array/Tensor of shape (N, C) containing predicted softmax probabilities.
    labels: Array/Tensor of shape (N,) containing true integer class indices.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier_score(probs, labels, num_classes=8):
    """
    Computes standard multiclass Brier score: (1/N) * sum_i sum_c (P_ic - Y_ic)^2.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    n = len(labels)
    one_hot = np.zeros((n, num_classes), dtype=np.float64)
    one_hot[np.arange(n), labels] = 1.0

    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return float(brier)


def evaluate_applied_metrics(model, loader, device, num_classes=8, head='A'):
    """
    Comprehensive applied evaluation over a DataLoader.
    Computes:
    - top1_accuracy (%): standard top-1 classification accuracy
    - balanced_accuracy (%): macro mean recall across all classes (crucial for ISIC imbalance)
    - macro_auc: macro one-vs-rest Area Under the ROC Curve
    - ece: Expected Calibration Error
    - brier: Multiclass Brier score
    """
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images, head=head)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_probs.append(probs.cpu())
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # 1. Top-1 Accuracy
    top1 = float(np.mean(all_preds == all_labels) * 100.0)

    # 2. Balanced Accuracy
    recalls = []
    for c in range(num_classes):
        mask = (all_labels == c)
        if np.sum(mask) > 0:
            recalls.append(np.mean(all_preds[mask] == c))
    bal_acc = float(np.mean(recalls) * 100.0) if len(recalls) > 0 else 0.0

    # 3. Macro AUC
    try:
        from sklearn.metrics import roc_auc_score
        # Only evaluate over classes present in labels
        present_classes = np.unique(all_labels)
        if len(present_classes) > 1 and len(present_classes) == num_classes:
            macro_auc = float(roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro'))
        elif len(present_classes) > 1:
            # Subset probs to present classes
            sub_probs = all_probs[:, present_classes]
            sub_probs = sub_probs / sub_probs.sum(axis=1, keepdims=True)
            macro_auc = float(roc_auc_score(all_labels, sub_probs, multi_class='ovr', average='macro', labels=present_classes))
        else:
            macro_auc = 0.5
    except Exception:
        macro_auc = 0.5

    # 4. Calibration
    ece = compute_ece(all_probs, all_labels)
    brier = compute_brier_score(all_probs, all_labels, num_classes=num_classes)

    return {
        "top1_accuracy": top1,
        "balanced_accuracy": bal_acc,
        "macro_auc": macro_auc,
        "ece": ece,
        "brier_score": brier,
        "num_samples": len(all_labels),
    }


def train_and_eval_linear_probe(
    model, train_loader, val_loader, device,
    feat_dim=2048, num_classes=8, epochs=10, lr=1e-3, weight_decay=1e-4, batch_size=256
):
    """
    Fits a linear classifier on frozen penultimate representations extracted by `model`.
    Trained with Adam for `epochs` on train_loader features, evaluated on val_loader features.
    Returns: dict of validation metrics for the linear probe.
    """
    model.eval()

    def extract_dataset_features(loader):
        feats = []
        lbls = []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                f = model(images, head='features')
                feats.append(f.cpu())
                lbls.append(labels)
        return torch.cat(feats, dim=0), torch.cat(lbls, dim=0)

    train_feats, train_labels = extract_dataset_features(train_loader)
    val_feats, val_labels = extract_dataset_features(val_loader)

    probe = torch.nn.Linear(feat_dim, num_classes).to(device)
    torch.nn.init.kaiming_normal_(probe.weight, nonlinearity='linear')
    torch.nn.init.constant_(probe.bias, 0.0)

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    train_dataset = torch.utils.data.TensorDataset(train_feats, train_labels)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    probe.train()
    for _ in range(epochs):
        for bx, by in train_dl:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = probe(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    probe.eval()
    val_dataset = torch.utils.data.TensorDataset(val_feats, val_labels)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    all_probs = []
    all_preds = []
    all_lbls = []
    with torch.no_grad():
        for bx, by in val_dl:
            bx = bx.to(device)
            out = probe(bx)
            probs = F.softmax(out, dim=1)
            _, pred = out.max(1)
            all_probs.append(probs.cpu())
            all_preds.append(pred.cpu())
            all_lbls.append(by)

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_lbls = torch.cat(all_lbls, dim=0).numpy()

    top1 = float(np.mean(all_preds == all_lbls) * 100.0)
    try:
        from sklearn.metrics import roc_auc_score
        macro_auc = float(roc_auc_score(all_lbls, all_probs, multi_class='ovr', average='macro'))
    except Exception:
        macro_auc = 0.5

    return {
        "probe_top1": top1,
        "probe_macro_auc": macro_auc,
    }

