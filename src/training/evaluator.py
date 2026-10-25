"""
Diagnostic evaluators and dynamics tracking for SOTLI:
- Dense validation logger during Phase 3 relearning
- Diagnostic Linear Probing at checkpoint theta_2 (disentangling head misalignment from representation damage)
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.metrics import evaluate_accuracy


class RelearningDynamicsTracker:
    """
    Tracks dense validation trajectory during Phase 3 (Updates 20,000 to 35,000).
    Computes:
    - u_rec: updates required to recover P_before
    - AULC: Area Under the Relearning Curve
    - Fixed-horizon recoveries (+500, +1000, +2000 updates)
    """
    def __init__(self, p_before):
        self.p_before = p_before
        self.trajectory = []  # List of (relative_update, val_acc)
        self.u_rec = None
        self.horizon_500 = None
        self.horizon_1000 = None
        self.horizon_2000 = None

    def record_step(self, rel_update, val_acc):
        self.trajectory.append((rel_update, val_acc))
        
        # Check recovery threshold
        if self.u_rec is None and val_acc >= self.p_before:
            self.u_rec = rel_update

        # Track fixed horizons
        if self.horizon_500 is None and rel_update >= 500:
            self.horizon_500 = val_acc
        if self.horizon_1000 is None and rel_update >= 1000:
            self.horizon_1000 = val_acc
        if self.horizon_2000 is None and rel_update >= 2000:
            self.horizon_2000 = val_acc

    def compute_aulc(self):
        """Computes Area Under the Relearning Curve via trapezoidal integration"""
        if len(self.trajectory) < 2:
            return 0.0
        updates = [step for step, acc in self.trajectory]
        accs = [acc for step, acc in self.trajectory]
        # Trapezoidal rule normalized by total update span
        total_span = max(1, updates[-1] - updates[0])
        integral = 0.0
        for i in range(len(updates) - 1):
            dx = updates[i+1] - updates[i]
            avg_y = (accs[i+1] + accs[i]) / 2.0
            integral += avg_y * dx
        return float(integral / total_span)

    def get_summary(self):
        return {
            'u_rec': self.u_rec if self.u_rec is not None else -1,  # -1 indicates unrecovered
            'aulc': self.compute_aulc(),
            'acc_horizon_500': self.horizon_500 if self.horizon_500 is not None else 0.0,
            'acc_horizon_1000': self.horizon_1000 if self.horizon_1000 is not None else 0.0,
            'acc_horizon_2000': self.horizon_2000 if self.horizon_2000 is not None else 0.0,
        }


def run_diagnostic_probe_at_theta2(model, train_loader, val_loader, device, num_classes_a=100, probe_epochs=5):
    """
    Fits a standardized linear probe on the FROZEN backbone w_2 using Task A training data.
    Evaluates whether representations at theta_2 retained discriminative power for Task A.
    If probe accuracy >> P_after_B (on frozen head_a), then F_B was primarily head misalignment.
    """
    model.eval()
    
    # Linear probe layer
    probe = nn.Linear(model.feat_dim, num_classes_a).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(probe_epochs):
        probe.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                features = model(images, head='features')
            
            outputs = probe(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate probe on validation set
    probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            features = model(images, head='features')
            outputs = probe(features)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    probe_acc = 100.0 * correct / max(1, total)
    return probe_acc
